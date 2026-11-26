"""Deadline queue for native 200ms packets. No game semantics or action rewriting."""
import math
import threading
import time
from collections import deque


class CadenceScheduler:
    def __init__(self, apply, release, emit, period_ms=200, max_outstanding=2):
        self.apply, self.release, self.emit = apply, release, emit
        self.period = period_ms
        self.max_outstanding = max_outstanding
        self.cv = threading.Condition()
        self.queue = deque()
        self.active = None
        self.cancelled = False
        self.closed = False
        self.fault = None
        self.previous_end = None
        self.seen = set()
        self.thread = threading.Thread(target=self._run, name='gamepad-cadence', daemon=True)
        self.thread.start()

    @staticmethod
    def now():
        return time.perf_counter() * 1000

    def enqueue(self, packet):
        states, seq = packet.get('states'), packet.get('packageSeq')
        requested = float(packet['startAtHelperMs'])
        if not math.isfinite(requested) or not seq or not isinstance(states, list) or len(states) != 6:
            raise ValueError('cadence requires a finite deadline, packageSeq and six states')
        with self.cv:
            if self.closed or self.fault:
                raise RuntimeError(self.fault or 'cadence scheduler closed')
            if seq in self.seen:
                raise ValueError('duplicate cadence packageSeq')
            if len(self.queue) + bool(self.active) >= self.max_outstanding:
                raise ValueError('cadence queue full')
            # A late arrival moves to a later beat, never compresses a packet or
            # preempts its predecessor. All deadlines stay on the same grid.
            earliest = self.now()
            tail = self.queue[-1] if self.queue else self.active
            if tail:
                earliest = max(earliest, tail['scheduledStartHelperMs'] + self.period)
            skipped = max(0, math.ceil((earliest - requested - 0.001) / self.period))
            if skipped and packet.get('strictDeadline'):
                self.seen.add(seq)
                self.emit(dict(event='gamepad_package_cancelled', packageSeq=seq, started=False,
                               events=[], reason='fixed_execution_deadline_missed', missedSlots=skipped,
                               completedAtUnixMs=time.time()*1000))
                return {'packageSeq': seq, 'scheduledStartHelperMs': requested, 'missedSlots': skipped,
                        'receivedHelperMs': self.now(), 'rejected': True}
            p = dict(packet, scheduledStartHelperMs=requested + skipped * self.period,
                     missedSlots=skipped, receivedHelperMs=self.now())
            self.seen.add(seq)
            self.queue.append(p)
            self.cv.notify_all()
            return {k: p[k] for k in ('packageSeq', 'scheduledStartHelperMs', 'missedSlots', 'receivedHelperMs')}

    def busy(self):
        with self.cv:
            return bool(self.active or self.queue)

    def cancel(self, pending_only=False):
        with self.cv:
            discarded = list(self.queue)
            self.queue.clear()
            for p in discarded:
                self.emit(dict(event='gamepad_package_cancelled', packageSeq=p['packageSeq'],
                               started=False, events=[], reason='queue_cancelled', completedAtUnixMs=time.time()*1000))
            release_result = {}
            if not pending_only:
                self.cancelled = True
                self.cv.notify_all()
                while self.active is not None:
                    self.cv.wait()
                release_result = self.release() or {}
                self.cancelled = False
                self.previous_end = None
            self.cv.notify_all()
        return {'cancelledPackages': len(discarded), **release_result}

    def close(self):
        self.cancel()
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('cadence worker did not stop')

    def _wait(self, deadline):
        # Called with cv held; cancel is a barrier against all subsequent writes.
        while not self.cancelled and not self.closed:
            remaining = deadline - self.now()
            if remaining <= 0:
                return True
            self.cv.wait(remaining / 1000)
        return False

    def _run(self):
        with self.cv:
            while not self.closed:
                if not self.queue:
                    self.cv.wait()
                    continue
                # Keep future packets in the pending queue so pending-only drain
                # can cancel them even before their deadline.
                p = self.queue[0]
                remaining = p['scheduledStartHelperMs'] - self.now()
                if remaining > 0:
                    self.cv.wait(remaining / 1000)
                    continue
                self.active = self.queue.popleft()
                events = []
                start = self.now()
                start_wall = time.time() * 1000
                scheduled = p['scheduledStartHelperMs']
                reason = None
                previous_gap = max(0, start - self.previous_end) if self.previous_end is not None else None
                try:
                    # A stalled host must not replay stale chunks in a burst.
                    if start - scheduled > 50:
                        reason = 'scheduler_deadline_missed'
                    else:
                        self.emit(dict(event='gamepad_package_started', packageSeq=p['packageSeq'],
                                       startedAtUnixMs=start_wall, startedHelperMs=start,
                                       scheduledStartHelperMs=scheduled))
                        for i, state in enumerate(p['states']):
                            if not self._wait(scheduled + i * self.period / 6):
                                reason = 'cancelled'
                                break
                            if self.now() - (scheduled + i * self.period / 6) > 50:
                                reason = 'scheduler_deadline_missed'
                                break
                            self.apply(state)
                            events.append({'actualMs': self.now() - start, 'state': state})
                        if not reason and not self._wait(scheduled + self.period):
                            reason = 'cancelled'
                    end = self.now()
                    contiguous = bool(not reason and self.queue and
                                      self.queue[0]['scheduledStartHelperMs'] <= scheduled + self.period + 0.001)
                    if not contiguous:
                        self.release()
                    self.emit(dict(event='gamepad_package_cancelled' if reason else 'gamepad_package_completed',
                                   packageSeq=p['packageSeq'], started=bool(events), reason=reason,
                                   startedAtUnixMs=start_wall, completedAtUnixMs=time.time()*1000,
                                   scheduledStartHelperMs=scheduled, requestedStartHelperMs=p['startAtHelperMs'],
                                   startedHelperMs=start, completedHelperMs=end,
                                   actualDurationMs=end-start, helperDurationMs=end-start,
                                   deadlineLatenessMs=start-scheduled, missedSlots=p['missedSlots'],
                                   previousBoundaryGapMs=previous_gap, events=events,
                                   firstStateApplyMs=events[0]['actualMs'] if events else None,
                                   lastStateApplyMs=events[-1]['actualMs'] if events else None,
                                   idleStateApplied=not contiguous, successorQueued=contiguous))
                    self.previous_end = end
                except Exception as error:
                    self.fault = str(error)
                    try:
                        self.release()
                    finally:
                        self.emit(dict(event='gamepad_package_cancelled', packageSeq=p['packageSeq'],
                                       started=bool(events), reason='helper_error', error=str(error), events=events,
                                       startedAtUnixMs=start_wall, completedAtUnixMs=time.time()*1000))
                        # Report every reserved successor as cancelled too.
                        self.cancel(pending_only=True)
                finally:
                    self.active = None
                    self.cv.notify_all()

"""Fixed-grid streamed control with clipped tails and an independent input watchdog.

No guessed actions: missing states hold the last *applied* state. Only a fresh
packet's first complete state refreshes the watchdog. All writes, cancellation
and watchdog decisions serialize under one lock. No timer callback replays data.
"""
import math
import threading
import time


class StreamScheduler:
    def __init__(self, apply, release, emit, now=None, threaded=True, pause=None):
        self.apply, self.release, self.emit, self.pause = apply, release, emit, pause
        self.now = now or (lambda: time.perf_counter()*1000)
        self.cv = threading.Condition()
        self.epoch = None
        self.armed = False
        self.closed = False
        self.packets = {}
        self.active = None
        self.last_valid = None
        self.last_sequence = -1
        self.last_end = None
        self.thread = None
        self.owner = 'A'
        self.ownership_version = 0
        self.update_version = 0
        self.ownership_required = False
        self.last_applied_state = None
        if threaded:
            self.thread = threading.Thread(target=self._run, name='gamepad-stream', daemon=True)
            self.thread.start()

    def arm(self, epoch, timeout_ms=1000, pause_enabled=False, reference_ms=None, owner="A", ownership_version=0, update_version=0):
        if not isinstance(epoch, str) or not epoch or type(timeout_ms) not in (int, float) or not 500 <= timeout_ms <= 1000:
            raise ValueError('Stream requires an epoch and a watchdog in [500,1000]ms')
        with self.cv:
            self.cancel('rearm')
            self.epoch = epoch
            self.owner = owner
            self.ownership_version = ownership_version
            self.update_version = update_version
            self.pause_enabled = bool(pause_enabled)
            reference = self.now() if reference_ms is None else float(reference_ms)
            if not math.isfinite(reference) or reference > self.now()+1000: raise ValueError('Invalid watchdog reference')
            self.last_valid = max(self.now(), reference)
            self.timeout = timeout_ms
            self.last_sequence = -1
            self.armed = True
            self.cv.notify_all()
            return {'armed': True, 'epoch': epoch, 'helperPerfMs': self.now()}

    def update(self, msg):
        with self.cv:
            now = self.now()
            if not self.armed or msg.get('epoch') != self.epoch:
                return {'accepted': False, 'reason': 'inactive_or_stale_epoch'}
            if (msg.get('ownerId', None if self.ownership_required else self.owner) != self.owner
                    or msg.get('ownershipVersion', None if self.ownership_required else self.ownership_version) != self.ownership_version
                    or msg.get('updateVersion', None if self.ownership_required else self.update_version) != self.update_version):
                return {'accepted': False, 'reason': 'stale_owner_or_version'}
            # Check expiry before accepting: a late event cannot revive a watchdog.
            self.tick()
            if not self.armed: return {'accepted': False, 'reason': 'watchdog_latched'}
            seq, index = msg.get('sequence'), msg.get('index')
            start = float(msg['startAtHelperMs'])
            if type(seq) is not int or type(index) is not int or not 0 <= index < 6 or not math.isfinite(start):
                raise ValueError('Invalid stream state envelope')
            if start + 200 <= now or seq < self.last_sequence:
                return {'accepted': False, 'reason': 'expired_packet'}
            if start > now + 400: raise ValueError('Stream packet too far in future')
            p = self.packets.get(seq)
            if p is None:
                if index != 0: return {'accepted': False, 'reason': 'missing_first_state'}
                if len(self.packets) >= 4: raise ValueError('Stream queue full')
                p = dict(sequence=seq, packageSeq=msg['packageSeq'], start=start, end=start+200,
                         states={}, events=[], started=None, index=-1, valid=False,
                         epoch=self.epoch, ownerId=self.owner, ownershipVersion=self.ownership_version,
                         updateVersion=self.update_version)
                self.packets[seq] = p
                self.last_valid = now
            if p['packageSeq'] != msg['packageSeq'] or p['start'] != start:
                raise ValueError('Changed stream packet identity')
            if index in p['states']:
                if p['states'][index] != msg['state']: raise ValueError('Conflicting stream state')
                return {'accepted': False, 'reason': 'duplicate_state'}
            p['states'][index] = msg['state']
            self.cv.notify_all()
            self.tick()
            return {'accepted': True, 'receivedHelperMs': now}

    @staticmethod
    def _complete_state(state):
        if not isinstance(state, dict): return False
        def number(value, low, high):
            return type(value) in (int, float) and math.isfinite(value) and low <= value <= high
        for stick in ('leftStick', 'rightStick'):
            if not isinstance(state.get(stick), dict): return False
            if not all(number(state[stick].get(axis), -1, 1) for axis in ('x', 'y')): return False
        if not all(number(state.get(key), 0, 1) for key in ('leftTrigger', 'rightTrigger')): return False
        buttons = state.get('buttons')
        allowed = {'a', 'b', 'x', 'y', 'lb', 'rb', 'back', 'start',
                   'left_thumb', 'right_thumb', 'dpad_up', 'dpad_down', 'dpad_left', 'dpad_right', 'guide'}
        return isinstance(buttons, list) and all(isinstance(b, str) and b in allowed for b in buttons)

    def handoff(self, msg):
        """One locked ownership transaction; never calls release or re-arms a watchdog.

        A fully parsed successor is installed before revoking the old owner. The
        old physical state remains held until the successor's absolute window.
        Invalid standby replies leave the active queue and watchdog untouched.
        """
        with self.cv:
            now = self.now()
            reject = lambda reason: {'accepted': False, 'reason': reason, 'helperPerfMs': now}
            if not self.armed or msg.get('epoch') != self.epoch:
                return reject('inactive_or_stale_epoch')
            if (msg.get('expectedOwner') != self.owner
                    or msg.get('ownershipVersion') != self.ownership_version + 1):
                return reject('stale_owner_or_version')
            if (msg.get('nextOwner') not in ('A', 'B') or msg.get('nextOwner') == self.owner
                    or type(msg.get('updateVersion')) is not int or msg['updateVersion'] <= self.update_version):
                return reject('stale_update_version')
            if not isinstance(msg.get('nextEpoch'), str) or not msg['nextEpoch'] or msg['nextEpoch'] == self.epoch:
                return reject('invalid_next_epoch')
            states = msg.get('states')
            if not isinstance(states, list) or len(states) != 6 or not all(self._complete_state(s) for s in states):
                return reject('incomplete_or_invalid_action')
            if type(msg.get('sequence')) is not int or not isinstance(msg.get('packageSeq'), str):
                return reject('invalid_packet_identity')
            start, observed, age_limit = msg.get('startAtHelperMs'), msg.get('observedAtHelperMs'), msg.get('maxObservationAgeMs')
            if not all(type(x) in (int, float) and math.isfinite(x) for x in (start, observed, age_limit)):
                return reject('invalid_handoff_timing')
            if age_limit <= 0 or age_limit > 400 or observed > now + 5 or max(now, start) - observed > age_limit:
                return reject('stale_observation')
            if start + 200 <= now or start > now + 400:
                return reject('expired_or_future_packet')
            # A watchdog timeout is authoritative even if a standby action races it.
            if now - self.last_valid >= self.timeout:
                self.tick()
                return reject('watchdog_latched')
            previous_owner, previous_epoch = self.owner, self.epoch
            held = self.last_applied_state
            for p in list(self.packets.values()):
                reason = 'service_handoff' if now < p['end'] else ('window_end' if p['started'] is not None else 'expired')
                self._finish(p, min(now, p['end']), reason)
            self.owner, self.epoch = msg['nextOwner'], msg['nextEpoch']
            self.ownership_required = True
            self.ownership_version, self.update_version = msg['ownershipVersion'], msg['updateVersion']
            self.last_sequence = -1
            p = dict(sequence=msg['sequence'], packageSeq=msg['packageSeq'], start=start, end=start+200,
                     states=dict(enumerate(states)), events=[], started=None, index=-1, valid=True,
                     epoch=self.epoch, ownerId=self.owner, ownershipVersion=self.ownership_version,
                     updateVersion=self.update_version, handoff=True, handoffPreviousState=held)
            self.packets[p['sequence']] = p
            self.last_valid = now  # This is a fresh, validated action, not a standby heartbeat.
            receipt = dict(accepted=True, epoch=self.epoch, ownerId=self.owner,
                           previousOwner=previous_owner, previousEpoch=previous_epoch,
                           ownershipVersion=self.ownership_version, updateVersion=self.update_version,
                           packageSeq=p['packageSeq'], handoffHelperMs=now,
                           observedAtHelperMs=observed, observationAgeMs=now-observed,
                           plannedExecutionObservationAgeMs=max(now, start)-observed,
                           scheduledStartHelperMs=start, systemReleased=False, previousState=held)
            self.emit(dict(event='gamepad_stream_handoff', **receipt))
            self.cv.notify_all()
            self.tick()
            return receipt

    def finalize(self, msg):
        with self.cv:
            if msg.get('epoch') != self.epoch: return {'accepted': False}
            if not msg.get('valid'):
                self.cancel('invalid_stream_output')
                return {'accepted': False}
            p = self.packets.get(msg.get('sequence'))
            if p: p['valid'] = True
            return {'accepted': bool(p)}

    def _finish(self, p, now, reason):
        # The final state remains physically held until a successor or cancellation.
        if p['events']:
            p['events'][-1]['endedHelperMs'] = now
        self.emit(dict(event='gamepad_stream_finished', packageSeq=p['packageSeq'], sequence=p['sequence'],
            epoch=p['epoch'], ownerId=p['ownerId'], ownershipVersion=p['ownershipVersion'], updateVersion=p['updateVersion'],
            started=p['started'] is not None, startedHelperMs=p['started'], endedHelperMs=now,
            scheduledStartHelperMs=p['start'], scheduledEndHelperMs=p['end'],
            actualDurationMs=now-p['started'] if p['started'] is not None else 0,
            deadlineLatenessMs=max(0, (p['started'] or now)-p['start']),
            events=p['events'], reason=reason, streamValidated=p['valid'],
            completedAtUnixMs=time.time()*1000, lastStateHeld=reason in ('window_end', 'service_handoff')))
        self.packets.pop(p['sequence'], None)
        if self.active is p: self.active = None
        if p['started'] is not None: self.last_end = now

    def tick(self):
        with self.cv:
            now = self.now()
            if not self.armed: return
            if now-self.last_valid >= self.timeout:
                elapsed = now-self.last_valid
                self.cancel('valid_action_timeout')
                released = self.now()
                notice = dict(epoch=self.epoch,
                    ownerId=self.owner, ownershipVersion=self.ownership_version, updateVersion=self.update_version,
                    elapsedMs=elapsed, releasedHelperMs=released, timeoutMs=self.timeout,
                    watchdogId=f'{self.epoch}:{released:.6f}')
                # Flush the latch notification before the synchronous P key hold.
                # emit only writes stdout; it never waits for a JS/native command.
                # The later completion remains the barrier for pause verification.
                self.emit(dict(event='gamepad_stream_watchdog_triggered', **notice,
                    pausePending=bool(self.pause_enabled and self.pause)))
                try:
                    pause_receipt = self.pause() if self.pause_enabled and self.pause else {}
                except Exception as error:
                    self.emit(dict(event='gamepad_stream_watchdog', **notice,
                        pauseCompleted=False, pauseError=str(error)))
                    raise
                self.emit(dict(event='gamepad_stream_watchdog', **notice,
                    pauseCompleted=True, **pause_receipt))
                return
            for p in list(self.packets.values()):
                if now >= p['end']: self._finish(p, min(now, p['end']), 'window_end' if p['started'] is not None else 'expired')
            eligible = [p for p in self.packets.values() if p['start'] <= now < p['end'] and 0 in p['states']]
            if not eligible: return
            p = max(eligible, key=lambda x: x['sequence'])
            if self.active is not p:
                if self.active: self._finish(self.active, now, 'successor_takeover')
                self.active = p
                p['started'] = now
                p['startedUnixMs'] = time.time()*1000
                self.last_sequence = p['sequence']
                self.emit(dict(event='gamepad_stream_started', packageSeq=p['packageSeq'],
                    epoch=p['epoch'], ownerId=p['ownerId'], ownershipVersion=p['ownershipVersion'], updateVersion=p['updateVersion'],
                    startedHelperMs=now, startedAtUnixMs=p['startedUnixMs'],
                    carryGapMs=None if self.last_end is None else now-self.last_end))
            due = min(5, int((now-p['started'])/(200/6)+1e-8))
            ready = [i for i in p['states'] if p['index'] < i <= due]
            if ready:
                index = max(ready)
                previous_state = self.last_applied_state
                applied = self.apply(p['states'][index])
                actual = self.now()
                self.last_applied_state = p['states'][index]
                if p.get('handoff') and not p['events']:
                    self.emit(dict(event='gamepad_stream_handoff_applied', packageSeq=p['packageSeq'],
                                   epoch=p['epoch'], ownerId=p['ownerId'], ownershipVersion=p['ownershipVersion'],
                                   updateVersion=p['updateVersion'], appliedHelperMs=actual,
                                   previousState=previous_state, state=p['states'][index], systemReleased=False))
                if p['events']: p['events'][-1]['endedHelperMs'] = actual
                p['events'].append(dict(index=index, state=p['states'][index], appliedHelperMs=actual, applyReceipt=applied))
                p['index'] = index

    def cancel(self, reason='cancelled'):
        with self.cv:
            self.armed = False
            now = self.now()
            for p in list(self.packets.values()): self._finish(p, now, reason)
            self.release()
            self.last_applied_state = None
            self.last_end = None
            self.cv.notify_all()
            return {'cancelled': True, 'releasedHelperMs': self.now()}

    def busy(self):
        with self.cv: return self.armed or bool(self.packets)

    def _run(self):
        while True:
            with self.cv:
                if self.closed: return
                try: self.tick()
                except Exception as error:
                    self.cancel('helper_error')
                    self.emit(dict(event='gamepad_stream_error', error=str(error)))
                self.cv.wait(0.001 if self.armed else 0.1)

    def close(self):
        self.cancel('closed')
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        if self.thread: self.thread.join(timeout=2)

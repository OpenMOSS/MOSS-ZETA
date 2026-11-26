const { EventEmitter } = require('node:events');
const { randomUUID } = require('node:crypto');
const { ActionRuntime, postprocessPausedControllerState } = require('./actionRuntime');

const HANDOFF_BUTTONS = new Set(['a', 'b', 'x', 'y', 'lb', 'rb', 'back', 'start', 'guide',
  'left_thumb', 'right_thumb', 'dpad_up', 'dpad_down', 'dpad_left', 'dpad_right']);
function completeControllerState(state) {
  const number = (value, low, high) => typeof value === 'number' && Number.isFinite(value) && value >= low && value <= high;
  return state && ['leftStick', 'rightStick'].every(stick => state[stick]
    && ['x', 'y'].every(axis => number(state[stick][axis], -1, 1)))
    && ['leftTrigger', 'rightTrigger'].every(key => number(state[key], 0, 1))
    && Array.isArray(state.buttons) && state.buttons.every(button => HANDOFF_BUTTONS.has(button));
}

class StreamCadenceExecutor extends EventEmitter {
  constructor(input, options = {}) {
    super(); this.input = input; this.actionRuntime = options.actionRuntime || new ActionRuntime({ inputController: input });
    this.isPackageValid = options.isPackageValid || (() => true);
    this.watchdogPause = options.watchdogPause === true;
    this.watchdogTimeoutMs = options.hfValidActionTimeoutMs ?? 1000;
    this.fixedCadence = this.streaming = true; this.items = new Map(); this.sequence = 0;
    this.completedPackages = this.droppedPackages = this.preemptedPackages = 0;
    this.submissionTail = Promise.resolve(); this.waiters = []; this.armed = false;
    this.onInputEvent = e => this.nativeEvent(e);
    this.ownerId = options.policyOwnerId || 'A'; this.ownershipVersion = 0; this.updateVersion = 0;
    this.ownershipRequired = options.policyHotSwapEnabled === true; this.handoffInProgress = false;
  }
  start() {
    if (this.running) return;
    this.running = this.accepting = true; this.stopPromise = null; this.epoch = randomUUID();
    this.input.on('event', this.onInputEvent);
    this.loopPromise = new Promise((resolve, reject) => { this.resolveLoop = resolve; this.rejectLoop = reject; });
    this.readyPromise = (async () => {
      await this.actionRuntime.reset('stream_cadence_start');
      if (!this.input.streamCadenceSupported) throw Error('Input helper lacks streaming cadence');
      this.clock = await this.input.calibrateGamepadClock();
    })();
    this.readyPromise.catch(e => this.fatal(e));
  }
  async arm(referencePerfMs = null) {
    await this.readyPromise;
    if (this.armed) return;
    if (this.watchdogNotice && !this.watchdogNotice.completed) throw Error('Native watchdog pause is not complete');
    this.epoch = randomUUID();
    this.watchdogNotice = null;
    await this.input.send('gamepad_stream_arm', { epoch: this.epoch, ownerId: this.ownerId,
      ownershipVersion: this.ownershipVersion, updateVersion: this.updateVersion, timeoutMs: this.watchdogTimeoutMs, watchdogPause: this.watchdogPause,
      referenceAtHelperMs: referencePerfMs == null ? null : referencePerfMs + this.clock.offsetMs });
    this.armed = true;
  }
  hasCapacity() { return this.accepting && this.items.size < 4; }
  getStatus() { return { scheduling: 'stream_fixed_cadence', periodMs: 200, running: this.running,
    accepting: this.accepting, ownerId: this.ownerId, ownershipVersion: this.ownershipVersion,
    updateVersion: this.updateVersion, handoffInProgress: this.handoffInProgress, watchdogTimeoutMs: this.watchdogTimeoutMs, active: [...this.items.values()].some(i => i.started),
    activePackageSeq: [...this.items.values()].find(i => i.started)?.seq || null,
    pendingQueueLength: this.items.size, completedPackages: this.completedPackages,
    preemptedPackages: this.preemptedPackages, droppedPackages: this.droppedPackages, clock: this.clock }; }
  emitStatus() { this.emit('status', this.getStatus()); }
  submit(command, payload) {
    const task = this.submissionTail.then(() => this.input.send(command, payload));
    this.submissionTail = task.catch(e => { this.fatal(e); });
    return task;
  }
  matchesOwner(metadata) {
    if (metadata.policyOwnerId == null && metadata.policyOwnershipVersion == null && !this.ownershipRequired) return true;
    return metadata.policyOwnerId === this.ownerId && metadata.policyOwnershipVersion === this.ownershipVersion
      && metadata.policyUpdateVersion === this.updateVersion;
  }
  streamState(event, metadata) {
    if (this.handoffInProgress || !this.matchesOwner(metadata)) return;
    if (!this.accepting || !this.armed || !this.isPackageValid(metadata) || event.brainCall) return;
    let item = this.items.get(metadata.requestId);
    if (!item) {
      if (event.index !== 0 || !this.hasCapacity()) return;
      item = { ...metadata, seq: ++this.sequence, requestedStates: [], states: [], chunks: [],
        firstExecutablePerfMs: event.receivedPerfMs, enqueuedPerfMs: performance.now(), streamEpoch: this.epoch };
      item.helperPackageId = `${this.epoch}:${item.seq}`;
      this.items.set(item.requestId, item);
    }
    item.requestedStates[event.index] = event.state;
    item.states[event.index] = postprocessPausedControllerState(event.state);
    item.appliedStates = item.states;
    item.chunks[event.index] = { index: event.index, raw: event.raw, state: item.states[event.index] };
    const payload = { epoch: item.streamEpoch, ownerId: this.ownerId,
      ownershipVersion: this.ownershipVersion, updateVersion: this.updateVersion, sequence: item.seq, packageSeq: item.helperPackageId,
      startAtHelperMs: item.scheduledStartPerfMs + this.clock.offsetMs,
      index: event.index, state: item.states[event.index] };
    // Keep the pipe ordered, but never await completion of the previous packet.
    void this.submit('gamepad_stream_state', payload).then(ack => {
      if (event.index === 0) {
        item.firstStateAck = ack;
        if (!ack.accepted && !item.receipt) {
          item.receipt = { event: 'gamepad_stream_finished', started: false, reason: ack.reason, events: [], completedAtUnixMs: Date.now() };
          this.flush(item);
        }
      }
    });
  }
  async handoff({ expectedOwner, nextOwner, updateVersion, ownershipVersion, observationPerfMs,
    maxObservationAgeMs = 400, firstExecutablePerfMs, pkg, commit, isCurrent = () => true }) {
    const reject = reason => ({ accepted: false, reason, ownerId: this.ownerId, ownershipVersion: this.ownershipVersion });
    if (!this.running || !this.accepting || !this.armed) return reject('inactive_executor');
    if (this.handoffInProgress) return reject('handoff_in_progress');
    if (!this.input.streamHandoffSupported) return reject('helper_lacks_atomic_handoff');
    if (expectedOwner !== this.ownerId || !['A', 'B'].includes(nextOwner) || nextOwner === expectedOwner
      || ownershipVersion !== this.ownershipVersion + 1) return reject('stale_owner_or_version');
    if (!Number.isInteger(updateVersion) || updateVersion <= this.updateVersion) return reject('stale_update_version');
    if (!pkg || !pkg.requestId || typeof commit !== 'function' || !Array.isArray(pkg.states)
      || pkg.states.length !== 6 || !pkg.states.every(completeControllerState)) return reject('incomplete_or_invalid_action');
    if ((pkg.policyOwnerId != null && pkg.policyOwnerId !== nextOwner)
      || (pkg.policyOwnershipVersion != null && pkg.policyOwnershipVersion !== ownershipVersion)
      || (pkg.policyUpdateVersion != null && pkg.policyUpdateVersion !== updateVersion)) return reject('package_owner_mismatch');
    if (![observationPerfMs, maxObservationAgeMs, firstExecutablePerfMs, pkg.scheduledStartPerfMs].every(Number.isFinite)
      || maxObservationAgeMs <= 0 || maxObservationAgeMs > 400) return reject('invalid_handoff_timing');
    const now = performance.now();
    if (Math.max(now, pkg.scheduledStartPerfMs) - observationPerfMs > maxObservationAgeMs
      || observationPerfMs > now + 5) return reject('stale_observation');
    if (pkg.scheduledStartPerfMs + 200 <= now || pkg.scheduledStartPerfMs > now + 400) return reject('expired_or_future_packet');
    if (!isCurrent()) return reject('superseded_update');
    if (this.items.has(pkg.requestId)) return reject('duplicate_packet');
    const oldEpoch = this.epoch, nextEpoch = randomUUID(), oldItems = [...this.items.values()];
    const requestedStates = pkg.requestedStates || pkg.states;
    const states = pkg.states.map(postprocessPausedControllerState);
    const item = { ...pkg, seq: ++this.sequence, requestedStates, states, appliedStates: states,
      chunks: states.map((state, index) => ({ ...pkg.chunks?.[index], index, state })), firstExecutablePerfMs,
      enqueuedPerfMs: now, streamEpoch: nextEpoch, finalized: true,
      policyOwnerId: nextOwner, policyOwnershipVersion: ownershipVersion, policyUpdateVersion: updateVersion };
    item.helperPackageId = `${nextEpoch}:${item.seq}`;
    this.handoffInProgress = true;
    this.items.set(item.requestId, item); // Native events can arrive before the IPC acknowledgement.
    const payload = { epoch: oldEpoch, nextEpoch, expectedOwner, nextOwner, ownershipVersion, updateVersion,
      sequence: item.seq, packageSeq: item.helperPackageId, states,
      observedAtHelperMs: observationPerfMs + this.clock.offsetMs, maxObservationAgeMs,
      startAtHelperMs: item.scheduledStartPerfMs + this.clock.offsetMs };
    try {
      // Reserve the same ordered IPC stream as ordinary state submissions. A
      // superseded standby may not gain authority merely by waiting in this queue.
      const task = this.submissionTail.then(() => {
        if (!this.accepting || !this.armed || this.epoch !== oldEpoch || !isCurrent()) return reject('superseded_update');
        return this.input.send('gamepad_stream_handoff', payload);
      });
      this.submissionTail = task.catch(error => this.fatal(error));
      const ack = await task;
      if (!ack.accepted) { this.items.delete(item.requestId); return ack; }
      this.epoch = nextEpoch; this.ownerId = nextOwner;
      this.ownershipVersion = ownershipVersion; this.updateVersion = updateVersion; this.ownershipRequired = true;
      item.firstStateAck = ack;
      // The callback commits framework state and routing in this same JS turn.
      // It must be synchronous: no old owner may enqueue with the new epoch.
      const result = commit({ ...ack, package: item });
      if (result && typeof result.then === 'function') throw Error('Handoff commit must be synchronous');
      for (const old of oldItems) {
        old.cancelled = true;
        old.receipt ||= { started: false, reason: 'service_handoff', events: [], completedAtUnixMs: Date.now() };
        if (this.items.has(old.requestId)) this.flush(old);
      }
      this.emit('handoff', { ...ack, package: item, committedPerfMs: performance.now() });
      this.emitStatus();
      return { ...ack, package: item };
    } catch (error) { this.fatal(error); throw error; }
    finally {
      this.handoffInProgress = false;
      if (!this.items.size) this.waiters.splice(0).forEach(resolve => resolve());
    }
  }
  finalize(requestId, pkg) {
    const item = this.items.get(requestId);
    if (!item) return false;
    const states = item.states, requestedStates = item.requestedStates, firstExecutablePerfMs = item.firstExecutablePerfMs;
    Object.assign(item, pkg, { states, appliedStates: states, requestedStates, firstExecutablePerfMs, finalized: true });
    void this.submit('gamepad_stream_finalize', { epoch: item.streamEpoch, sequence: item.seq, valid: true });
    this.flush(item); return item;
  }
  nativeEvent(event) {
    if (event.event === 'gamepad_stream_watchdog_triggered' || event.event === 'gamepad_stream_watchdog') {
      if (event.epoch !== this.epoch) return; // A retired helper epoch cannot pause a newly armed owner.
      const id = event.watchdogId || `${event.epoch}:${event.releasedHelperMs}`;
      if (!this.watchdogNotice || this.watchdogNotice.id !== id) {
        this.watchdogNotice = { id, completed: false };
        this.armed = this.accepting = false;
        this.emit('watchdogTriggered', event);
      }
      if (event.event === 'gamepad_stream_watchdog' && !this.watchdogNotice.completed) {
        this.watchdogNotice.completed = true;
        this.emit('watchdog', event);
      }
      return;
    }
    if (event.event === 'gamepad_stream_handoff' || event.event === 'gamepad_stream_handoff_applied') {
      this.emit('nativeHandoff', event); return;
    }
    if (event.event === 'gamepad_stream_error') { this.fatal(Error(event.error)); return; }
    if (!event.event?.startsWith('gamepad_stream_')) return;
    const item = [...this.items.values()].find(i => i.helperPackageId === event.packageSeq);
    if (!item) return;
    if (event.event === 'gamepad_stream_started') { item.started = true; item.startReceipt = event; this.emitStatus(); }
    if (event.event === 'gamepad_stream_finished') { item.receipt = event; this.flush(item); }
  }
  flush(item) {
    if (!item.receipt || (!item.finalized && !item.cancelled)) return;
    const r = item.receipt;
    this.items.delete(item.requestId);
    const timing = { executionMode: 'stream_fixed_cadence_helper', configuredDurationMs: 200,
      clockCalibration: this.clock, realtimeHelperCompletion: r,
      scheduledStartPerfMs: item.scheduledStartPerfMs, actualDurationMs: r.actualDurationMs || 0,
      executableToEnqueueMs: item.enqueuedPerfMs-item.firstExecutablePerfMs,
      firstStateApplyFromExecutableMs: r.events?.[0]?.appliedHelperMs-this.clock.offsetMs-item.firstExecutablePerfMs,
      deadlineLatenessMs: r.deadlineLatenessMs, previousBoundaryGapMs: item.startReceipt?.carryGapMs,
      firstStateAck: item.firstStateAck, chunkCount: r.events?.length || 0,
      tailClippedMs: r.started ? Math.max(0, 200-r.actualDurationMs) : 200 };
    const event = { package: item, startedAt: new Date(item.startReceipt?.startedAtUnixMs || r.completedAtUnixMs).toISOString(),
      completedAt: new Date(r.completedAtUnixMs).toISOString(), durationMs: r.actualDurationMs || 0, timing };
    if (!r.started) { this.droppedPackages++; this.emit('packageDiscarded', { ...event, reason: r.reason }); }
    else if (item.finalized && r.reason === 'window_end' && r.actualDurationMs >= 198 && r.events.length === 6) {
      this.completedPackages++; this.emit('packageComplete', event);
    } else { this.preemptedPackages++; this.emit('packagePreempted', event); }
    if (!this.items.size) this.waiters.splice(0).forEach(r => r());
    this.emitStatus();
  }
  waitForIdle() { return this.items.size ? new Promise(resolve => this.waiters.push(resolve)) : Promise.resolve(); }
  async clearQueue(reason = 'cancelled') {
    this.armed = false;
    for (const item of this.items.values()) item.cancelled = true;
    await this.submit('gamepad_stream_cancel', { reason });
    for (const item of this.items.values()) {
      item.receipt ||= { started: false, reason, events: [], completedAtUnixMs: Date.now() };
      this.flush(item);
    }
  }
  async drainToBoundary(reason) { this.accepting = false; await this.clearQueue(reason); await this.waitForIdle(); }
  resumeAccepting() {
    if (this.watchdogNotice && !this.watchdogNotice.completed) return false;
    this.accepting = true; return true;
  }
  fatal(error) { this.rejectLoop?.(error); void this.stop({ reason: 'stream_error' }).catch(() => {}); }
  stop({ reason = 'stop' } = {}) {
    if (this.stopPromise) return this.stopPromise;
    this.accepting = false;
    return this.stopPromise = (async () => {
      await this.clearQueue(reason); await this.actionRuntime.reset(reason);
      this.input.off('event', this.onInputEvent); this.running = false; this.resolveLoop?.();
    })();
  }
}
module.exports = { StreamCadenceExecutor };

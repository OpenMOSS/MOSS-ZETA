// Controller timing shared by the local action execution paths.
(function (root) {
  const neutral = () => ({ leftStick: { x: 0, y: 0 }, rightStick: { x: 0, y: 0 }, leftTrigger: 0, rightTrigger: 0, buttons: [] });
  const copy = value => structuredClone(value);
  const active = value => typeof value === 'object' ? value.x !== 0 || value.y !== 0 : value > 0;
  const equal = (a, b) => typeof a === 'object' && typeof b === 'object' ? a.x === b.x && a.y === b.y : a === b;
  const zero = key => key.endsWith('Stick') ? { x: 0, y: 0 } : 0;
  function controls(state) {
    const result = new Map(['leftStick', 'rightStick', 'leftTrigger', 'rightTrigger'].map(key => [key, copy(state[key] ?? zero(key))]));
    for (const button of state.buttons || []) result.set('button:' + button, 1);
    return result;
  }

  class ControllerContinuity {
    constructor({ write, reset, recordEvent = async () => {}, now = () => performance.now(), wallNow = () => Date.now() }) {
      Object.assign(this, { write, resetDevice: reset, recordEvent, now, wallNow });
      this.items = new Map();
      this.pending = new Map();
      this.epoch = 0;
      this.tail = Promise.resolve();
      this.timer = null;
      this.timerDeadline = null;
      this.fullHold = null;
      this.enabled = false;
      this.hasAppliedChunk = false;
      this.releasing = false;
      this.error = null;
      this.sequence = 0;
      this.transitions = [];
      this.configure({ thresholdChunks: 12, releaseMs: 0 });
    }

    configure({ thresholdChunks = this.thresholdChunks, releaseMs = this.releaseMs } = {}) {
      if (!Number.isInteger(thresholdChunks) || thresholdChunks < 1 || thresholdChunks > 100000) throw Error('Invalid longHoldThresholdChunks');
      if (!Number.isFinite(releaseMs) || releaseMs < 0 || releaseMs > 60000) throw Error('Invalid longHoldReleaseMs');
      this.thresholdChunks = thresholdChunks;
      this.releaseMs = releaseMs;
    }

    state() {
      const result = neutral();
      for (const [key, item] of this.items) {
        if (key.startsWith('button:')) { if (active(item.value)) result.buttons.push(key.slice(7)); }
        else result[key] = copy(item.value);
      }
      return result;
    }

    status() {
      return {
        packageEndHold: this.fullHold ? copy(this.fullHold) : null,
        packageEndHoldReleasing: this.releasing,
        packageEndHoldError: this.error?.message || null,
        continuity: { enabled: this.enabled, thresholdChunks: this.thresholdChunks, releaseMs: this.releaseMs,
          controls: Object.fromEntries([...this.items].map(([key, item]) => [key, { value: copy(item.value), chunks: item.chunks }])),
          pending: [...this.pending].map(([control, p]) => ({ control, deadlinePerfMs: p.deadline, kind: p.kind, source: p.context })) }
      };
    }

    audit(type, data) {
      // Persistence must not block an input transition. Framework's writer also
      // observes its own failures; standalone runtimes retain them in status.
      Promise.resolve().then(() => this.recordEvent(type, data)).catch(error => { this.error = error; });
    }

    run(operation, epoch = this.epoch) {
      const task = this.tail.then(async () => {
        if (epoch !== this.epoch) return { cancelled: true };
        return operation();
      });
      this.tail = task;
      task.catch(error => { this.error = error; });
      return task;
    }

    item(key) {
      if (!this.items.has(key)) this.items.set(key, { value: zero(key), chunks: 0, version: 0 });
      return this.items.get(key);
    }

    cancel(key, reason) {
      const pending = this.pending.get(key);
      if (!pending) return;
      this.pending.delete(key);
      this.audit('controller_delayed_release_cancelled', { control: key, reason, deadlinePerfMs: pending.deadline,
        oldValue: pending.value, actualPendingMs: Math.max(0, this.now() - pending.startedPerfMs), source: pending.context });
    }

    setValue(key, value) {
      const item = this.item(key);
      if (!equal(item.value, value)) { item.value = copy(value); item.chunks = 0; item.version++; item.activeSincePerfMs = active(value) ? this.now() : null; }
    }

    schedule(key, delayMs, kind, context, now) {
      if (this.pending.has(key)) return; // Never move an existing release deadline.
      const item = this.item(key);
      const pending = { deadline: now + delayMs, startedPerfMs: now, heldSincePerfMs: item.activeSincePerfMs,
        value: copy(item.value), chunks: item.chunks, kind,
        context: { candidateId: context.candidateId, actionSeq: context.actionSeq }, epoch: this.epoch };
      this.pending.set(key, pending);
      this.audit('controller_delayed_release_started', { control: key, ...copy(pending), delayMs });
    }

    arm() {
      const deadline = this.pending.size ? Math.min(...[...this.pending.values()].map(p => p.deadline)) : null;
      if (this.timer && this.timerDeadline === deadline) return;
      clearTimeout(this.timer); this.timer = null; this.timerDeadline = deadline;
      if (deadline == null) return;
      const epoch = this.epoch;
      this.timer = setTimeout(() => {
        this.timer = null; this.timerDeadline = null;
        this.releasing = true;
        const task = this.run(async () => {
          const now = this.now(), due = [...this.pending].filter(([, p]) => p.deadline <= now);
          if (!due.length) return;
          for (const [key] of due) { this.pending.delete(key); this.setValue(key, zero(key)); }
          await this.writeState('release_deadline');
          for (const [control, p] of due) this.audit('controller_delayed_release_completed', {
            control, ...copy(p), actualReleasedPerfMs: this.now(), actualDelayMs: this.now() - p.startedPerfMs,
            actualHeldMs: this.now() - p.heldSincePerfMs, lateMs: Math.max(0, this.now() - p.deadline) });
          if (!this.pending.size) this.endFullHold('timeout');
        }, epoch);
        task.finally(() => { this.releasing = false; if (epoch === this.epoch) this.arm(); }).catch(() => {});
      }, Math.max(0, deadline - this.now()));
      this.timer.unref?.();
    }

    async writeState(reason, requestedState = null) {
      const state = this.state(), startedPerfMs = this.now();
      const result = await this.write(copy(state));
      const event = this.recordTransition(state, reason, startedPerfMs, requestedState);
      return { result, state, event };
    }

    recordTransition(state, reason, startedPerfMs, requestedState = null) {
      const event = { sequence: ++this.sequence, startedPerfMs, completedPerfMs: this.now(), state: copy(state), reason,
        ...(requestedState ? { requestedState: copy(requestedState) } : {}) };
      this.transitions.push(event); if (this.transitions.length > 256) this.transitions.shift();
      return event;
    }

    traceSince(sequence, originPerfMs) {
      return this.transitions.filter(event => event.sequence > sequence).map(event => ({
        ...copy(event), startedOffsetMs: event.startedPerfMs - originPerfMs,
        completedOffsetMs: event.completedPerfMs - originPerfMs
      }));
    }

    endFullHold(reason) {
      const hold = this.fullHold;
      this.fullHold = null;
      if (hold) this.audit('controller_package_end_hold_released', { ...hold, reason,
        endedAt: new Date(this.wallNow()).toISOString(), actualHoldMs: Math.max(0, this.now() - hold.startedPerfMs) });
    }

    applyChunk(requested, context = {}) {
      const epoch = this.epoch;
      return this.run(async () => {
        if (context.shouldStop?.()) return { cancelled: true };
        this.endFullHold('next_package');
        const values = controls(requested), now = this.now();
        for (const key of new Set([...this.items.keys(), ...values.keys()])) {
          const next = values.get(key) ?? zero(key), item = this.item(key), p = this.pending.get(key);
          if (active(next)) {
            this.cancel(key, 'new_executed_input'); this.setValue(key, next);
          } else if (active(item.value)) {
            if (p && (p.kind === 'long_hold' || (this.releaseMs > 0 && item.chunks >= this.thresholdChunks))) continue;
            if (item.chunks >= this.thresholdChunks && this.releaseMs > 0) this.schedule(key, this.releaseMs, 'long_hold', context.source || {}, now);
            else { this.cancel(key, 'model_zero'); this.setValue(key, next); }
          }
        }
        const applied = await this.writeState('model_chunk', requested);
        this.hasAppliedChunk = true;
        this.arm();
        return { ...applied, epoch, tokens: [...this.items].map(([key, item]) => [key, item.version, copy(item.value)]) };
      }, epoch);
    }

    async completeChunk(receipt, shouldStop = () => false) {
      if (!receipt || receipt.completed) return;
      receipt.completed = true;
      await this.tail;
      if (!receipt || receipt.cancelled || receipt.epoch !== this.epoch || shouldStop()) return;
      for (const [key, version, value] of receipt.tokens) {
        const item = this.item(key);
        if (item.version === version && equal(item.value, value) && active(value)) item.chunks++;
      }
    }

    finishPackage(lastState, holdMs, context = {}) {
      return this.run(async () => {
        if (context.shouldStop?.()) return;
        if (!this.hasAppliedChunk) for (const [key, value] of controls(lastState)) this.setValue(key, value);
        const now = this.now();
        for (const [key, item] of this.items) {
          if (!active(item.value) || this.pending.has(key)) continue;
          const delay = Math.max(holdMs, item.chunks >= this.thresholdChunks ? this.releaseMs : 0);
          if (delay > 0) this.schedule(key, delay, 'package_end', context, now);
          else this.setValue(key, zero(key));
        }
        if (this.pending.size) {
          const wall = this.wallNow();
          this.fullHold = { state: this.state(), maxMs: holdMs, startedPerfMs: now, startedAt: new Date(wall).toISOString(),
            expiresAt: new Date(wall + Math.max(...[...this.pending.values()].map(p => p.deadline - now))).toISOString(),
            sourceCandidateId: context.candidateId, sourceActionSeq: context.actionSeq };
          this.audit('controller_package_end_hold_started', copy(this.fullHold));
          this.arm();
        } else {
          this.endFullHold('package_complete');
          const startedPerfMs = this.now();
          await this.resetDevice();
          this.recordTransition(neutral(), 'package_complete_reset', startedPerfMs);
        }
      });
    }

    invalidate(reason) {
      this.epoch++;
      clearTimeout(this.timer); this.timer = null; this.timerDeadline = null;
      for (const key of this.pending.keys()) this.cancel(key, reason);
      this.endFullHold(reason);
    }

    async clear(reason) {
      this.invalidate(reason);
      await this.tail;
      this.tail = Promise.resolve(); this.error = null; this.releasing = false;
      this.items.clear(); this.hasAppliedChunk = false; this.enabled = false;
    }

    async reset(reason) {
      // Synchronously revoke queued work, then drain a write already in flight.
      this.invalidate(reason);
      const task = this.tail.catch(() => {}).then(async () => {
        const startedPerfMs = this.now();
        const result = await this.resetDevice();
        this.recordTransition(neutral(), reason, startedPerfMs);
        this.items.clear(); this.hasAppliedChunk = false; this.enabled = false; this.error = null;
        return result;
      });
      this.tail = task; task.catch(error => { this.error = error; });
      return task;
    }
  }
  const api = { ControllerContinuity };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ControllerContinuity = api;
})(globalThis);

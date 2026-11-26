const { EventEmitter } = require('node:events');
const { randomUUID } = require('node:crypto');
const { ActionRuntime, postprocessPausedControllerState } = require('./actionRuntime');

// The native helper owns the deadline queue. Node never waits for a predecessor's
// completion before sending a ready successor, and never resets between packets.
class FixedCadenceExecutor extends EventEmitter {
  constructor(input, options = {}) {
    super();
    this.input = input;
    this.actionRuntime = options.actionRuntime || new ActionRuntime({ inputController: input });
    this.isPackageValid = options.isPackageValid || (() => true);
    this.items = new Map();
    this.jobs = new Set();
    this.packageSeq = 0;
    this.helperSession = randomUUID();
    this.completedPackages = 0;
    this.droppedPackages = 0;
    this.preemptedPackages = 0;
    this.generation = 0;
    this.accepting = false;
    this.running = false;
    this.fixedCadence = true;
    this.submissionTail = Promise.resolve();
    this.onInputEvent = (event) => {
      if (event.event !== 'gamepad_package_started') return;
      const item = [...this.items.values()].find((p) => p.helperPackageId === event.packageSeq);
      if (item) { item.started = true; this.emitStatus(); }
    };
  }

  start() {
    if (this.running) return;
    this.running = this.accepting = true;
    this.stopPromise = null;
    this.generation++;
    this.input.on('event', this.onInputEvent);
    this.loopPromise = new Promise((resolve, reject) => { this.resolveLoop = resolve; this.rejectLoop = reject; });
    this.readyPromise = (async () => {
      await this.actionRuntime.reset('fixed_cadence_start');
      this.clock = await this.input.calibrateGamepadClock();
    })();
    this.readyPromise.catch((error) => this.fatal(error));
    this.emitStatus();
  }

  fatal(error) {
    this.accepting = false;
    this.rejectLoop?.(error);
    // Cleanup must also run if the owner has not yet observed loopPromise.
    void this.stop({ reason: 'fixed_cadence_error' }).catch(() => {});
  }

  hasCapacity() { return this.accepting && this.items.size < 2; }

  getStatus() {
    const active = [...this.items.values()].find((p) => p.started);
    return { scheduling: 'fixed_cadence', periodMs: 200, running: this.running,
      accepting: this.accepting, active: Boolean(active), activePackageSeq: active?.seq || null,
      pendingQueueLength: this.items.size - Number(Boolean(active)),
      completedPackages: this.completedPackages, droppedPackages: this.droppedPackages,
      preemptedPackages: this.preemptedPackages, clock: this.clock || null,
      lastExecution: this.lastExecution || null };
  }

  emitStatus() { this.emit('status', this.getStatus()); }

  enqueue(pkg) {
    if (!this.hasCapacity()) return false;
    if (!Number.isFinite(pkg.scheduledStartPerfMs)) throw Error('Fixed cadence package requires a capture-slot deadline');
    const requestedStates = structuredClone(pkg.requestedStates || pkg.states);
    const appliedStates = requestedStates.map(postprocessPausedControllerState);
    const item = { ...pkg, seq: ++this.packageSeq, requestedStates, appliedStates, states: appliedStates,
      enqueuedPerfMs: performance.now(), generation: this.generation, started: false, submitted: false };
    item.helperPackageId = `${this.helperSession}:${item.seq}`;
    this.items.set(item.seq, item);
    // Serialize only submission ACKs, never the duration/completion of a packet.
    const submission = this.submissionTail.then(async () => {
      await this.readyPromise;
      if (item.cancelled || !this.accepting || item.generation !== this.generation || !this.isPackageValid(item)) return null;
      item.submitted = true;
      return this.input.gamepadPackageCadence(item.states, { packageSeq: item.helperPackageId,
        startAtHelperMs: item.scheduledStartPerfMs + this.clock.offsetMs, strictDeadline: Boolean(item.strictDeadline) });
    });
    this.submissionTail = submission.catch(() => {});
    const job = (async () => {
      try {
        const ack = await submission;
        if (!ack) {
          this.droppedPackages++;
          this.emit('packageDiscarded', { package: item, reason: 'invalid_or_cancelled_before_submission' });
          return;
        }
        const receipt = await this.input.waitForRealtimeGamepadPackageCompletion(item.helperPackageId, 5000);
        const timing = { executionMode: 'fixed_cadence_helper', configuredDurationMs: 200,
          scheduledStartPerfMs: item.scheduledStartPerfMs,
          executableToEnqueueMs: item.enqueuedPerfMs - item.firstExecutablePerfMs,
          queueWaitMs: receipt.startedHelperMs - this.clock.offsetMs - item.enqueuedPerfMs,
          executableToStartMs: receipt.startedHelperMs - this.clock.offsetMs - item.firstExecutablePerfMs,
          firstStateApplyFromExecutableMs: receipt.startedHelperMs + receipt.firstStateApplyMs - this.clock.offsetMs - item.firstExecutablePerfMs,
          clockCalibration: this.clock, enqueueReceipt: ack, realtimeHelperCompletion: receipt,
          actualDurationMs: receipt.actualDurationMs, chunkCount: receipt.events?.length || 0,
          missedSlots: receipt.missedSlots || 0, previousBoundaryGapMs: receipt.previousBoundaryGapMs,
          deadlineLatenessMs: receipt.deadlineLatenessMs };
        const event = { package: item, startedAt: new Date(receipt.startedAtUnixMs || receipt.completedAtUnixMs).toISOString(),
          completedAt: new Date(receipt.completedAtUnixMs).toISOString(), durationMs: receipt.actualDurationMs || 0, timing };
        this.lastExecution = { seq: item.seq, ...event, package: undefined };
        // Free the reservation before the synchronous completed-action commit.
        this.items.delete(item.seq);
        if (receipt.event === 'gamepad_package_completed') {
          this.completedPackages++;
          this.emit('packageComplete', event);
        } else if (receipt.started) {
          this.preemptedPackages++;
          this.emit('packagePreempted', event);
        } else {
          this.droppedPackages++;
          this.emit('packageDiscarded', { package: item, reason: receipt.reason, timing });
        }
        if (receipt.reason === 'helper_error' || receipt.reason === 'scheduler_deadline_missed') {
          throw Error(`Cadence helper failed: ${receipt.error || receipt.reason}`);
        }
      } catch (error) { this.fatal(error); }
      finally { this.items.delete(item.seq); this.emitStatus(); }
    })();
    this.jobs.add(job);
    void job.finally(() => this.jobs.delete(job));
    this.emitStatus();
    return item;
  }

  clearQueue(reason = 'clear') {
    return this.clearPending(reason).catch((error) => this.fatal(error));
  }

  async clearPending(reason) {
    for (const p of this.items.values()) if (!p.submitted) p.cancelled = true;
    // Wait for any already-dispatched ACK before cancelling the helper queue.
    await this.submissionTail;
    await this.input.cancelGamepadCadence(true);
    this.emit('log', `fixed cadence pending queue cleared: ${reason}`);
  }

  async waitForIdle() { await Promise.all([...this.jobs]); }

  async drainToBoundary(reason = 'boundary') {
    this.accepting = false;
    await this.clearQueue(reason);
    await this.waitForIdle();
    await this.actionRuntime.reset(reason);
    return this.getStatus();
  }

  resumeAccepting() {
    if (!this.running || this.stopPromise) return false;
    this.accepting = true;
    this.emitStatus();
    return true;
  }

  stop({ reason = 'stop', drain = false } = {}) {
    if (this.stopPromise) return this.stopPromise;
    this.accepting = false;
    this.generation++;
    this.stopPromise = (async () => {
      try {
        await this.submissionTail;
        if (!drain) await this.input.cancelGamepadCadence(false);
        await this.waitForIdle();
      } finally {
        await this.actionRuntime.reset(reason);
        this.input.off('event', this.onInputEvent);
        this.running = false;
        this.resolveLoop?.();
        this.emitStatus();
      }
      return this.getStatus();
    })();
    return this.stopPromise;
  }
}

module.exports = { FixedCadenceExecutor };

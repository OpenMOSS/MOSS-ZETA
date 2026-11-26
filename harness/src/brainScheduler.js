const { EventEmitter } = require("node:events");

const STATES = Object.freeze({
  WAITING_FOR_ACTIONS: "WAITING_FOR_ACTIONS",
  INFERENCING: "INFERENCING",
  RESPONSE_BUFFERED: "RESPONSE_BUFFERED"
});

function integer(value, fallback, min = 0) {
  const number = Math.trunc(Number(value));
  return Number.isFinite(number) ? Math.max(min, number) : fallback;
}

function observationSequence(value) {
  const match = String(value ?? "").match(/(\d+)$/);
  return match ? Number(match[1]) : null;
}

/**
 * Single-flight Brain scheduler.
 *
 * `runInference` receives an immutable request containing the trajectory
 * cutoff. It may resolve before the simulated action lag is over; the result
 * remains pending until the release condition is met.
 */
class BrainScheduler extends EventEmitter {
  constructor(options = {}) {
    super();
    this.barrierMode = options.barrierMode === true;
    this.deferCommit = false;
    this.minNewActionsBeforeStart = integer(options.minNewActionsBeforeStart, 5);
    this.simulatedLagActions = this.barrierMode ? 0 : integer(options.simulatedLagActions, 0);
    this.maxActionsWithoutMemoryUpdate = integer(options.maxActionsWithoutMemoryUpdate, 20, 1);
    this.maxActionsWithoutInstructionUpdate = integer(options.maxActionsWithoutInstructionUpdate, 40, 1);
    this.instructionFallbackEnabled = options.instructionFallbackEnabled === true;
    this.memoryFallbackIntervalMs = integer(options.memoryFallbackIntervalMs, 10000);
    // This clock measures unpaused gameplay, including gaps without a completed
    // package. Keep the existing action-duration clock for latency diagnostics.
    this.monotonicNow = options.monotonicNow || (() => performance.now());
    this.fallbackActiveMs = 0;
    this.fallbackClockStartedAt = null;
    this.lastMemoryRefreshActiveMs = 0;
    this.lastRecoveryCompletedActiveMs = 0;
    this.runInference = options.runInference;
    if (typeof this.runInference !== "function") {
      throw new TypeError("BrainScheduler requires a runInference function.");
    }
    this.onCommit = typeof options.onCommit === "function" ? options.onCommit : null;
    this.state = STATES.WAITING_FOR_ACTIONS;
    this.latestActionSeq = 0;
    this.latestObservationId = null;
    this.latestActiveGameMs = 0;
    this.lastCommittedCutoffActionSeq = 0;
    this.pending = null;
    this.revisionCount = 0;
    this.inferencePromise = null;
    this.generation = 0;
    this.stopped = false;
    this.replanRequested = false;
    this.replanMetadata = null;
  }

  getStatus() {
    return {
      state: this.state,
      latestActionSeq: this.latestActionSeq,
      latestObservationId: this.latestObservationId,
      latestActiveGameMs: this.latestActiveGameMs,
      lastCommittedCutoffActionSeq: this.lastCommittedCutoffActionSeq,
      minNewActionsBeforeStart: this.minNewActionsBeforeStart,
      simulatedLagActions: this.simulatedLagActions,
      maxActionsWithoutMemoryUpdate: this.maxActionsWithoutMemoryUpdate,
      maxActionsWithoutInstructionUpdate: this.maxActionsWithoutInstructionUpdate,
      instructionFallbackEnabled: this.instructionFallbackEnabled,
      memoryFallbackIntervalMs: this.memoryFallbackIntervalMs,
      fallbackClockRunning: this.fallbackClockStartedAt !== null,
      fallbackActiveGameMs: this.getFallbackActiveGameMs(),
      lastMemoryRefreshActiveMs: this.lastMemoryRefreshActiveMs,
      lastRecoveryCompletedActiveMs: this.lastRecoveryCompletedActiveMs,
      replanRequested: this.replanRequested,
      revisionCount: this.revisionCount,
      pending: this.pending
        ? {
            basedOnObservationId: this.pending.basedOnObservationId,
            cutoffActionSeq: this.pending.cutoffActionSeq,
            computedAtActionSeq: this.pending.computedAtActionSeq,
            virtualReleaseActionSeq: this.pending.virtualReleaseActionSeq
          }
        : null
    };
  }

  seedInitialRevision(result) {
    if (!this.barrierMode || this.stopped || this.latestObservationId === null
      || this.latestActionSeq !== 0 || this.revisionCount !== 0 || this.pending || this.inferencePromise
      || this.state !== STATES.WAITING_FOR_ACTIONS || this.replanRequested) {
      throw new Error("Brain seed requires a fresh paused session with its initial observation.");
    }
    this.revisionCount = 1;
    this.lastCommittedCutoffActionSeq = 0;
    this.replanMetadata = null;
    const revision = {
      revision: 1,
      basedOnObservationId: this.latestObservationId,
      cutoffActionSeq: 0,
      computedAtActionSeq: 0,
      actualPublishActionSeq: 0,
      configuredLagActions: 0,
      synthetic: true,
      result: JSON.parse(JSON.stringify(result))
    };
    this.onCommit?.(revision);
    this.lastMemoryRefreshActiveMs = this.getFallbackActiveGameMs();
    this.emit("revision_committed", revision);
    return revision;
  }

  observe(observationId) {
    this.latestObservationId = observationId;
    this.tryStart();
  }

  advanceActiveGameTime(durationMs) {
    const duration = Number(durationMs);
    if (Number.isFinite(duration) && duration > 0) {
      this.latestActiveGameMs += duration;
    }
  }

  getFallbackActiveGameMs() {
    return this.fallbackActiveMs + (this.fallbackClockStartedAt === null ? 0
      : Math.max(0, this.monotonicNow() - this.fallbackClockStartedAt));
  }

  setFallbackClockRunning(running) {
    const shouldRun = running === true && !this.stopped;
    if (shouldRun === (this.fallbackClockStartedAt !== null)) return;
    if (shouldRun) this.fallbackClockStartedAt = this.monotonicNow();
    else {
      this.fallbackActiveMs = this.getFallbackActiveGameMs();
      this.fallbackClockStartedAt = null;
    }
  }

  executedAction(action = {}) {
    this.latestActionSeq += 1;
    const incoming = observationSequence(action.observationId);
    const current = observationSequence(this.latestObservationId);
    if (this.latestObservationId == null || (incoming !== null && (current === null || incoming > current))) {
      this.latestObservationId = action.observationId ?? this.latestObservationId;
    }
    this.emit("action", {
      actionSeq: this.latestActionSeq,
      observationId: this.latestObservationId,
      action
    });
    this.tryRelease();
    this.tryStart();
    return this.latestActionSeq;
  }

  canStart() {
    const newActions = this.latestActionSeq - this.lastCommittedCutoffActionSeq;
    const initial = this.replanMetadata?.initial === true && this.latestActionSeq === 0;
    const requested = this.replanRequested;
    return !this.stopped
      && this.state === STATES.WAITING_FOR_ACTIONS
      && this.latestObservationId !== null
      && ((!this.barrierMode && newActions >= this.minNewActionsBeforeStart) || requested);
  }

  requestReplan(reason = "policy_request", metadata = {}) {
    if (this.stopped) return false;
    this.replanRequested = true;
    this.replanMetadata = { ...metadata, updateType: require("./brainProtocol").updateType({ reason, ...metadata }) };
    this.emit("replan_requested", {
      reason: String(reason),
      latestActionSeq: this.latestActionSeq,
      latestObservationId: this.latestObservationId,
      state: this.state
    });
    this.tryStart();
    return true;
  }

  invalidate(reason = "context_invalidated") {
    this.lastError = null;
    this.generation += 1;
    this.pending = null;
    this.inferencePromise = null;
    this.replanRequested = false;
    this.lastCommittedCutoffActionSeq = this.latestActionSeq;
    this.state = STATES.WAITING_FOR_ACTIONS;
    this.emit("context_invalidated", {
      reason: String(reason),
      latestActionSeq: this.latestActionSeq,
      latestObservationId: this.latestObservationId
    });
    this.emitStatus?.();
  }

  tryStart() {
    if (!this.canStart()) {
      return false;
    }

    const cutoffActionSeq = this.latestActionSeq;
    const request = Object.freeze({
      basedOnObservationId: this.latestObservationId,
      cutoffActionSeq,
      initial: this.replanMetadata?.initial === true && cutoffActionSeq === 0,
      startedAt: new Date().toISOString(),
      startedWallMs: Date.now(),
      activeGameMsAtRequest: this.latestActiveGameMs
      ,
      policyTriggerThinking: this.replanMetadata?.thinking || "",
      policyEvidence: this.replanMetadata?.policyEvidence ? Object.freeze({ ...this.replanMetadata.policyEvidence }) : null,
      policyProposal: this.replanMetadata?.policyProposal ? Object.freeze({ ...this.replanMetadata.policyProposal }) : null,
      updateType: this.replanMetadata?.updateType || "full"
    });
    this.state = STATES.INFERENCING;
    this.replanRequested = false;
    this.replanMetadata = null;
    this.lastError = null;
    this.emit("inference_started", request);
    const generation = this.generation;
    const promise = Promise.resolve().then(() => this.runInference(request));
    this.inferencePromise = promise;
    promise.then(
      (result) => {
        if (generation === this.generation) this.receiveResponse(request, result);
      },
      (error) => {
        if (generation === this.generation) this.receiveError(request, error);
      }
    );
    return true;
  }

  receiveResponse(request, result) {
    if (this.inferencePromise === null) {
      return;
    }
    if (result.recoveryCompleted && (!result.recoveryContext || result.recoveryReceipt?.status !== 'inputs_completed')) {
      this.receiveError(request, Error('Recovery completion requires confirmed execution and its evidence boundary'));
      return;
    }
    if (result.recoveryContext) {
      const c = result.recoveryContext;
      if (!Number.isSafeInteger(c.cutoffActionSeq) || c.cutoffActionSeq < request.cutoffActionSeq
        || c.cutoffActionSeq > this.latestActionSeq || !c.basedOnObservationId || c.updateType !== 'full') {
        this.receiveError(request, Error('Invalid post-recovery Brain evidence boundary'));
        return;
      }
      request = Object.freeze({ ...request, ...c });
    }
    this.pending = {
      ...request,
      result,
      computedAtActionSeq: this.latestActionSeq,
      virtualReleaseActionSeq: request.cutoffActionSeq + (request.initial ? 0 : this.simulatedLagActions),
      responseReadyAt: new Date().toISOString(),
      brainWallLatencyMs: Date.now() - request.startedWallMs,
      activeMsAtReceive: this.latestActiveGameMs - request.activeGameMsAtRequest,
      actionsAtReceive: this.latestActionSeq - request.cutoffActionSeq,
      observationGapAtReceive: (() => {
        const latest = observationSequence(this.latestObservationId);
        const based = observationSequence(request.basedOnObservationId);
        return latest === null || based === null ? null : latest - based;
      })()
    };
    this.state = STATES.RESPONSE_BUFFERED;
    this.emit("response_buffered", {
      ...this.getStatus().pending,
      result
    });
    this.tryRelease();
  }

  receiveError(request, error) {
    this.lastError = error;
    this.inferencePromise = null;
    this.state = STATES.WAITING_FOR_ACTIONS;
    this.emit("inference_error", {
      ...request,
      error: error?.message || String(error),
      ...(error?.brainResponse ? { response: error.brainResponse } : {})
    });
  }

  canRelease() {
    return !this.deferCommit && this.state === STATES.RESPONSE_BUFFERED
      && this.pending
      && this.latestActionSeq >= this.pending.virtualReleaseActionSeq;
  }

  tryRelease() {
    if (!this.canRelease()) {
      return false;
    }
    const pending = this.pending;
    this.pending = null;
    this.inferencePromise = null;
    if (pending.result.recoveryCompleted) {
      this.lastCommittedCutoffActionSeq = pending.cutoffActionSeq;
      this.lastRecoveryCompletedActiveMs = this.getFallbackActiveGameMs();
      this.state = STATES.WAITING_FOR_ACTIONS;
      this.emit('recovery_completed', { basedOnObservationId: pending.basedOnObservationId,
        cutoffActionSeq: pending.cutoffActionSeq, actualPublishActionSeq: this.latestActionSeq,
        result: pending.result });
      this.tryStart();
      return true;
    }
    this.revisionCount += 1;
    this.lastCommittedCutoffActionSeq = pending.cutoffActionSeq;
    const revision = {
      revision: this.revisionCount,
      basedOnObservationId: pending.basedOnObservationId,
      cutoffActionSeq: pending.cutoffActionSeq,
      computedAtActionSeq: pending.computedAtActionSeq,
      actualPublishActionSeq: this.latestActionSeq,
      configuredLagActions: this.simulatedLagActions,
      brainRevisionMetrics: {
        brainWallLatencyMs: pending.brainWallLatencyMs,
        gameActiveMsSinceRequest: this.latestActiveGameMs - pending.activeGameMsAtRequest,
        executedActionsSinceRequest: this.latestActionSeq - pending.cutoffActionSeq,
        observationGap: (() => {
          const latest = observationSequence(this.latestObservationId);
          const based = observationSequence(pending.basedOnObservationId);
          return latest === null || based === null ? null : latest - based;
        })()
      },
      brainRevisionReleaseMetrics: {
        activeMsAtReceive: pending.activeMsAtReceive,
        actionsAtReceive: pending.actionsAtReceive,
        observationGapAtReceive: pending.observationGapAtReceive,
        activeMsAtRelease: this.latestActiveGameMs - pending.activeGameMsAtRequest,
        actionsAtRelease: this.latestActionSeq - pending.cutoffActionSeq,
        observationGapAtRelease: (() => {
          const latest = observationSequence(this.latestObservationId);
          const based = observationSequence(pending.basedOnObservationId);
          return latest === null || based === null ? null : latest - based;
        })()
      },
      updateType: pending.updateType,
      policyTriggerThinking: pending.policyTriggerThinking || '',
      result: pending.result
    };
    this.state = STATES.WAITING_FOR_ACTIONS;
    if (this.onCommit) {
      this.onCommit(revision);
    }
    // Only an accepted state revision fulfils the refresh obligation. Starting,
    // superseding or failing a request (or accepting Policy think) never does.
    this.lastMemoryRefreshActiveMs = this.getFallbackActiveGameMs();
    this.emit("revision_committed", revision);
    this.tryStart();
    return true;
  }

  stop() {
    this.setFallbackClockRunning(false);
    this.stopped = true;
    this.generation += 1;
    this.pending = null;
    this.inferencePromise = null;
    this.state = STATES.WAITING_FOR_ACTIONS;
  }
}

module.exports = {
  BrainScheduler,
  BRAIN_SCHEDULER_STATES: STATES
};

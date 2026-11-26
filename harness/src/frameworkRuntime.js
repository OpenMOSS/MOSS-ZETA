const { isTrivialPolicy } = require("./trivialPolicy");
const { BrainRuntime } = require("./brainRuntime");
const { PolicyActor } = require("./policyActor");
const { ActionRuntime } = require("./actionRuntime");
const { PauseHarness } = require("./pauseHarness");

class FrameworkRuntime {
  constructor(options = {}) {
    this.config = options.config || {};
    this.sessionState = options.sessionState;
    this.history = options.history;
    this.brainActor = options.brainActor;
    this.brainEnabled = !isTrivialPolicy(options.config);
    this.host = options.host;
    this.stopped = false;
    this.automaticBrainBarrierPending = false;
    this.automaticBrainTrigger = null;
    this.automaticBrainTriggers = new WeakSet();
    this.frameworkFullBrainTriggers = new WeakSet();
    this.policyActor = options.policyActor || new PolicyActor({ sessionState: this.sessionState });
    this.actionRuntime = new ActionRuntime({
      sessionState: this.sessionState,
      inputController: options.inputController,
      recordEvent: (type, payload) => this.recordEvent(type, payload)
    });
    this.pauseHarness = new PauseHarness({
      sessionState: this.sessionState,
      recordEvent: (type, payload) => this.recordEvent(type, payload)
    });
    this.brainRuntime = new BrainRuntime({
      config: options.config,
      monotonicNow: options.monotonicNow,
      runInference: (request) => this.brainActor.infer({ ...request,
        previous: require("./brainSchema").brainView(this.sessionState),
        trajectory: JSON.parse(JSON.stringify(this.sessionState.brainObservationWindow(request.cutoffActionSeq,
          this.config.brainPrefillEnabled ? Math.min(8, this.config.brainTrajectoryObservationLimit || 8)
            : this.config.brainTrajectoryObservationLimit || 8))),
        evidenceRecords: JSON.parse(JSON.stringify(this.sessionState.timeline.readRecords({fromActionSeq:this.sessionState.brain.lastBrainCommitActionSeq||0,toActionSeq:request.cutoffActionSeq}))),
        previousActionSeq: this.sessionState.brain.lastBrainCommitActionSeq || 0
      }),
      setPendingRevision: (pending) => this.sessionState.setPendingBrainRevision(pending),
      commitRevision: (revision) => this.commitBrainRevision(revision),
      completeRecovery: (completion) => {
        this.sessionState.pendingBrainRevision = null;
        // Track the execution boundary separately: no memory refresh or Brain
        // revision is invented, and the old due counter cannot immediately loop.
        this.sessionState.brain.lastRecoveryActionSeq = completion.cutoffActionSeq;
        this.automaticBrainBarrierPending = false;
        this.automaticBrainTrigger = null;
        this.host?.onBrainRecoveryCompleted?.(completion);
      },
      recordEvent: (type, payload) => this.recordEvent(type, payload),
      onStatus: (status) => {
        this.sessionState.brain.scheduler = status;
      },
      onBrainEvent: (event) => this.host?.onBrainEvent?.(event)
    });
  }

  observeHistoryWrite(write) {
    // Some scheduler callbacks deliberately do not await audit persistence.
    // Observe their failure without hiding it from callers that do await it.
    void write.catch(error => this.host?.onHistoryError?.(error));
    return write;
  }

  recordEvent(type, payload) {
    if (!this.history) return Promise.resolve();
    return this.observeHistoryWrite(this.history.appendEvent(type, payload, {
      ...this.sessionState.historySnapshot()
    }));
  }

  commitBrainRevision(revision) {
    this.sessionState.commitBrainRevision(revision);
    this.brainActor?.invalidatePrefill?.('brain_revision_committed');
    if (this.history) {
      void this.observeHistoryWrite(this.history.appendRecord("brainRevisions", revision, "brain_revision", {
        revision: revision.revision,
        basedOnObservationId: revision.basedOnObservationId,
        cutoffActionSeq: revision.cutoffActionSeq,
        publishActionSeq: revision.actualPublishActionSeq
      }, {
        ...this.sessionState.historySnapshot()
      }));
    }
    this.host?.onBrainRevision?.(revision);
  }

  observation(record, { deferBrainObservation = false } = {}) {
    const timelineRecord = this.sessionState.addObservation(deferBrainObservation
      ? { ...record, brainEligible: false } : record);
    if (!deferBrainObservation) this.publishBrainObservation(timelineRecord.observationId);
    return timelineRecord;
  }

  publishBrainObservation(observationId) {
    const record = this.sessionState.timeline.records.find(item => item.type === 'observation'
      && item.observationId === observationId);
    if (this.stopped || !record || record.captureInvalidated) return false;
    record.brainEligible = true;
    if (this.brainEnabled) this.brainRuntime.observe(record.observationId);
    this.offerBrainPrefill();
    return true;
  }

  offerBrainPrefill() {
    const eligible = () => this.brainEnabled && !this.stopped && this.config.agentMode === 'high_frequency'
      // The host is a callback adapter, not the agent instance. Session state
      // carries the real pause/recovery flags; revision zero is synthetic startup.
      && this.sessionState.brain.revision > 0 && !this.sessionState.pauseHarness?.gamePaused
      && this.sessionState.pauseHarness?.phase !== 'PAUSE_REQUESTED' && !this.sessionState.recoveryMode
      && !this.host?.gamePaused && !this.host?.paused && !this.host?.recoveryMode
      && !this.brainRuntime.scheduler.inferencePromise && !this.brainRuntime.scheduler.pending;
    if (!eligible() || !this.brainActor?.canOfferPrefill?.()) return;
    this.brainActor.offerPrefill(() => {
      // The actor runs this factory off the capture/submission callback, once
      // per admitted warmup. Freeze one causal cutoff rather than every frame.
      if (!eligible()) return null;
      const started = performance.now();
      const state = this.sessionState;
      const cutoffActionSeq = state.timeline.actionSeq;
      const basedOnObservationId = this.brainRuntime.scheduler.latestObservationId;
      const previousActionSeq = state.brain.lastBrainCommitActionSeq || 0;
      const snapshot = {
        sessionId: state.sessionId, brainRevision: state.brain.revision,
        cutoffActionSeq, basedOnObservationId, previousActionSeq,
        trajectory: structuredClone(state.brainObservationWindow(cutoffActionSeq,
          Math.min(8, this.config.brainTrajectoryObservationLimit || 8))),
        evidenceRecords: structuredClone(state.timeline.readRecords({
          fromActionSeq: previousActionSeq, toActionSeq: cutoffActionSeq }))
      };
      snapshot.snapshotPreparationMs = performance.now() - started;
      return snapshot;
    });
  }

  policyCandidate(record) {
    return this.policyActor.createCandidate(record);
  }

  async inferPolicy(observation, requestActor = this.policyActor) {
    // A high-frequency request captures its owner before asynchronous image
    // preparation. Keep the framework entry point without rerouting that request
    // if a standby service takes ownership while preparation is in progress.
    return requestActor.infer(observation);
  }

  executedAction(record) {
    if (!this.brainEnabled) return this.actionRuntime.recordExecutedAction(record);
    this.brainRuntime.advanceActiveGameTime(record.execution?.durationMs || 0);
    const executed = this.actionRuntime.recordExecutedAction(record);
    this.brainRuntime.executedAction({
      ...record,
      observationId: record.basedOnObservationId
    });
    this.checkAutomaticBrainFallback();
    return executed;
  }

  setBrainFallbackClockRunning(running) {
    this.brainRuntime.setFallbackClockRunning(this.brainEnabled && !this.stopped && running === true);
  }

  checkAutomaticBrainFallback() {
    if (this.stopped || !this.brainEnabled) return null;
    const scheduler = this.sessionState.brain;
    const timing = this.brainRuntime.scheduler;
    const actionSeq = this.sessionState.timeline.actionSeq;
    const timedMemoryFallback = timing.memoryFallbackIntervalMs > 0;
    const memoryDue = timedMemoryFallback
      ? timing.fallbackClockStartedAt !== null && timing.getFallbackActiveGameMs()
        - Math.max(timing.lastMemoryRefreshActiveMs, timing.lastRecoveryCompletedActiveMs) >= timing.memoryFallbackIntervalMs
      : actionSeq - Math.max(Number(scheduler.lastMemoryUpdateActionSeq || 0), Number(scheduler.lastRecoveryActionSeq || 0))
      >= Number(timing.maxActionsWithoutMemoryUpdate
        || this.sessionState.brain.scheduler?.maxActionsWithoutMemoryUpdate || 20);
    const instructionDue = timing.instructionFallbackEnabled !== false
      && ['high_frequency', 'paused_package'].includes(this.config.agentMode)
      && actionSeq - Math.max(Number(scheduler.lastInstructionUpdateActionSeq || 0), Number(scheduler.lastRecoveryActionSeq || 0))
        >= Number(this.brainRuntime.scheduler.maxActionsWithoutInstructionUpdate || 40);
    if (instructionDue || (memoryDue && !this.brainRuntime.scheduler.inferencePromise
      && !this.brainRuntime.scheduler.pending)) {
      this.automaticBrainBarrierPending = true;
      this.automaticBrainTrigger = Object.freeze(instructionDue
        ? { reason: 'instruction_age_safety_barrier', updateType: 'full' }
        : { reason: timedMemoryFallback ? 'active_time_safety_barrier' : 'action_count_safety_barrier', updateType: 'fallback' });
      this.automaticBrainTriggers.add(this.automaticBrainTrigger);
    }
    return this.automaticBrainBarrierPending ? this.automaticBrainTrigger : null;
  }

  consumeAutomaticBrainTrigger(policyTrigger) {
    let trigger = this.automaticBrainBarrierPending
      ? this.automaticBrainTrigger || { reason: 'action_count_safety_barrier', updateType: 'fallback' } : null;
    if (trigger?.updateType === 'full' && policyTrigger) {
      trigger = Object.freeze({ ...trigger, thinking: policyTrigger.thinking || '', policyEvidence: policyTrigger.policyEvidence || null });
      this.automaticBrainTriggers.add(trigger);
    }
    this.automaticBrainBarrierPending = false;
    this.automaticBrainTrigger = null;
    return trigger;
  }

  isAutomaticInstructionRefresh(trigger) {
    return Boolean(this.brainRuntime.scheduler.instructionFallbackEnabled
      && trigger && this.automaticBrainTriggers.has(trigger) && trigger.updateType === 'full');
  }

  createFrameworkFullBrainTrigger(reason) {
    if(typeof reason!=='string'||!reason.trim())throw Error('Explicit framework Brain event requires a reason');
    const trigger=Object.freeze({reason,updateType:'full'});
    this.frameworkFullBrainTriggers.add(trigger);return trigger;
  }

  isFrameworkFullBrainTrigger(trigger) {
    return this.isAutomaticInstructionRefresh(trigger)||Boolean(trigger&&this.frameworkFullBrainTriggers.has(trigger));
  }

  automaticInstructionRefreshIsDue(trigger) {
    const brain = this.sessionState.brain;
    return this.isAutomaticInstructionRefresh(trigger)
      && this.sessionState.timeline.actionSeq - Math.max(Number(brain.lastInstructionUpdateActionSeq || 0), Number(brain.lastRecoveryActionSeq || 0))
        >= this.brainRuntime.scheduler.maxActionsWithoutInstructionUpdate;
  }

  resumeSettle(durationMs) {
    if (this.brainEnabled) this.brainRuntime.advanceActiveGameTime(durationMs);
  }

  requestBrainReplan(reason = "policy_request", metadata = {}) {
    if (!this.brainEnabled) return false;
    const accepted = this.brainRuntime.requestReplan(reason, metadata);
    if (accepted) {
      void this.recordEvent("brain_replan_requested", { reason });
    }
    return accepted;
  }

  async waitForBrainBarrier(timeoutMs) {
    return this.brainRuntime.waitForSettled(timeoutMs);
  }

  invalidateBrainContext(reason = "context_invalidated") {
    if (!['superseded by full update', 'newer Policy think'].includes(reason)) {
      this.brainActor?.invalidatePrefill?.(reason);
    }
    this.brainActor?.stop?.(reason);
    this.brainRuntime.invalidate(reason);
    void this.recordEvent("brain_context_invalidated", { reason });
  }

  getStatus() {
    return {
      session: this.sessionState.snapshot({ includeTimeline: false }),
      brain: this.brainRuntime.getStatus(),
      brainPrefill: this.brainActor?.getPrefillStatus?.() || null,
      actionRuntime: this.actionRuntime.getStatus()
    };
  }

  stop(reason = "session stopped") {
    this.stopped = true;
    this.brainActor?.invalidatePrefill?.(reason);
    this.brainRuntime.stop();
    this.policyActor.stop?.(reason);
    this.brainActor.stop(reason);
    void this.actionRuntime.reset(reason).catch(() => {});
  }
}

module.exports = {
  FrameworkRuntime
};

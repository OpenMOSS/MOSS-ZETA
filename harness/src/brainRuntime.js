const { BrainScheduler } = require("./brainScheduler");

class BrainRuntime {
  constructor(options = {}) {
    this.scheduler = new BrainScheduler({
      barrierMode: true,
      minNewActionsBeforeStart: options.config?.brainScheduler?.minNewActionsBeforeStart ?? options.minNewActionsBeforeStart,
      simulatedLagActions: options.config?.brainScheduler?.simulatedLagActions ?? options.simulatedLagActions,
      maxActionsWithoutMemoryUpdate: options.config?.brainScheduler?.maxActionsWithoutMemoryUpdate,
      maxActionsWithoutInstructionUpdate: options.config?.brainScheduler?.maxActionsWithoutInstructionUpdate,
      instructionFallbackEnabled: options.config?.brainScheduler?.instructionFallbackEnabled,
      memoryFallbackIntervalMs: options.config?.brainScheduler?.memoryFallbackIntervalMs,
      monotonicNow: options.monotonicNow,
      runInference: options.runInference,
      onCommit: (revision) => {
        if (typeof options.commitRevision === "function") {
          options.commitRevision(revision);
        }
        if (typeof options.recordEvent === "function") {
          void options.recordEvent("brain_revision_committed", revision);
        }
      }
    });
    this.recordEvent = typeof options.recordEvent === "function" ? options.recordEvent : null;
    this.onStatus = typeof options.onStatus === "function" ? options.onStatus : null;
    this.onBrainEvent = typeof options.onBrainEvent === "function" ? options.onBrainEvent : null;
    this.scheduler.on("inference_started", (payload) => {
      this.record("brain_inference_started", payload);
      this.onBrainEvent?.({ type: "inference_started", ...payload });
      this.publishStatus();
    });
    this.scheduler.on("response_buffered", (payload) => {
      if (payload.result?.recoveryCompleted) {
        this.record('brain_recovery_buffered', payload);
        this.onBrainEvent?.({ type: 'recovery_buffered', ...payload });
        this.publishStatus();
        return;
      }
      if (typeof options.setPendingRevision === "function") {
        options.setPendingRevision(payload);
      }
      this.record("brain_response_buffered", payload);
      this.onBrainEvent?.({ type: "response_buffered", ...payload });
      this.publishStatus();
    });
    this.scheduler.on("inference_error", (payload) => {
      this.record("brain_inference_error", payload);
      this.onBrainEvent?.({ type: "inference_error", ...payload });
      this.publishStatus();
    });
    this.scheduler.on("revision_committed", (revision) => {
      this.onBrainEvent?.({ type: "revision_committed", ...revision });
      this.publishStatus();
    });
    this.scheduler.on('recovery_completed', (payload) => {
      options.completeRecovery?.(payload);
      this.record('brain_recovery_completed', payload);
      this.onBrainEvent?.({ type: 'recovery_completed', ...payload });
      this.publishStatus();
    });
    this.scheduler.on("replan_requested", (payload) => {
      this.record("brain_replan_requested", payload);
      this.onBrainEvent?.({ type: "replan_requested", ...payload });
      this.publishStatus();
    });
    this.scheduler.on("context_invalidated", (payload) => {
      this.record("brain_context_invalidated", payload);
      this.onBrainEvent?.({ type: "context_invalidated", ...payload });
      this.publishStatus();
    });
  }

  record(type, payload) {
    if (this.recordEvent) {
      void this.recordEvent(type, payload);
    }
  }

  observe(observationId) {
    this.scheduler.observe(observationId);
    this.publishStatus();
  }

  executedAction(action) {
    const sequence = this.scheduler.executedAction(action);
    this.publishStatus();
    return sequence;
  }

  requestReplan(reason = "policy_request", metadata = {}) {
    return this.scheduler.requestReplan(reason, metadata);
  }

  seedInitialRevision(result) {
    return this.scheduler.seedInitialRevision(result);
  }

  invalidate(reason = "context_invalidated") {
    this.scheduler.invalidate(reason);
  }

  advanceActiveGameTime(durationMs) {
    this.scheduler.advanceActiveGameTime(durationMs);
    this.publishStatus();
  }

  setFallbackClockRunning(running) {
    this.scheduler.setFallbackClockRunning(running);
    this.publishStatus();
  }

  getStatus() {
    return this.scheduler.getStatus();
  }

  async waitForSettled(timeoutMs = 180000) {
    const started = Date.now();
    while (this.scheduler.state !== "WAITING_FOR_ACTIONS"
      || this.scheduler.inferencePromise
      || this.scheduler.pending) {
      if (timeoutMs > 0 && Date.now() - started >= timeoutMs) {
        throw new Error("Timed out waiting for Brain barrier.");
      }
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    if (this.scheduler.lastError) throw this.scheduler.lastError;
  }

  stop() {
    this.scheduler.stop();
    this.publishStatus();
  }

  publishStatus() {
    if (this.onStatus) {
      this.onStatus(this.getStatus());
    }
  }
}

module.exports = {
  BrainRuntime
};

const { isTrivialPolicy } = require('./trivialPolicy');
const { AgentRuntime } = require('./agentRuntime');
const { FrameworkCaptureStore } = require('./frameworkCaptureStore');
const { HighFrequencyExecutor } = require('./highFrequencyExecutor');
const { FixedCadenceExecutor } = require('./fixedCadenceExecutor');
const { StreamCadenceExecutor } = require('../../cadence/src/streamCadenceExecutor');
const { FixedSubmissionClock } = require('../../cadence/src/fixedSubmissionClock');
const { sanitizeConfig } = require('./config');
const { describeHighFrequencyPackage } = require('../../cadence/src/highFrequencyActionParser');
const { classifyPausedTaskState, PAUSED_TASK_STATES } = require('./pausedTaskState');
const { sleep } = require('../../cadence/src/actionExecutor');
const { imagePathToDataUrl } = require('./captureStore');
const { extractPolicyResponse } = require('./policyResponse');
const { HotSwapPolicyCoordinator } = require('./hotSwapPolicyCoordinator');

// The Policy loop runs independently of capture and controller execution.
class HighFrequencyAgent extends AgentRuntime {
  constructor(options = {}) {
    super(options);
    if (options.chatCompletionProvider) this.chatCompletion = options.chatCompletionProvider;
    if (options.resolveModel || options.resolveModelProvider) this.resolveModel = options.resolveModel || options.resolveModelProvider;
    this.checkPolicyHandoffCapabilities = options.checkPolicyHandoffCapabilities;
    this.executorFactory = options.executorFactory || ((input, config) => (config.policyStreaming ? new StreamCadenceExecutor(input, config) : config.fixedCadence ? new FixedCadenceExecutor(input, config) : new HighFrequencyExecutor(input, config)));
    this.executor = null;
    this.pendingCadenceObservation = null;
    this.captureStore = new FrameworkCaptureStore({ captureObserve: this.captureObserve,
      captureRoot: this.sessionCaptureRoot, publicRoot: this.sessionCaptureRoot,
      onPersistence: event => { this.deferPolicyEvidence(() => this.recordHistoryEvent('capture_persistence_completed', event)); },
      onLog: (message) => this.emit('log', message), getPinnedPaths: () => {
        if (!this.running) return [];
        const start = this.currentSession?.brain.lastBrainCommitActionSeq || 0;
        return [this.messages, this.lastObservation, this.policyOcr?.activeObservation,
          this.policyOcr?.pending?.observation,
          isTrivialPolicy(this.defaultConfig) ? [] : this.currentSession?.timeline.records.filter((r) => r.type === 'observation' && r.actionSeq >= start)];
      } });
    this.captureObserve = (config) => this.captureStore.capture({
      ...config,
      directCapture: process.platform === 'win32',
      captureFormat: 'jpeg',
      captureQuality: this.defaultConfig.hfCaptureArchiveQuality || 90
    });
    const recognize = this.recognizeScreenshotText;
    this.recognizeScreenshotText = async (config, observation) => {
      const release = this.captureStore.pin(observation.imagePath);
      try { return await recognize(config, observation); } finally { release(); }
    };
    this.captureTickCount = 0;
    this.skippedInferenceTicks = 0;
    this.droppedCaptureTicks = 0;
    this.contextEpoch = 0;
    this.inFlightPrediction = null;
    this.actionWrites = Promise.resolve();
    this.pendingHighFrequencyBarrier = null;
    this.pendingAutomaticBrainFallback = null;
    this.pendingSmallBrainUpdate = null;
    this.latestCadenceObservation = null;
    this.brainCycle = null;
    this.brainProtectionActive = false;
    this.hotSwap = null;
    this.completedBrainCandidates = new Set();
    this.policyBrainSources = new WeakSet();
    this.manualPauseHandled = false;
    this.nativeWatchdogPausePromise = null;
    this.nativeWatchdogTransition = null;
    this.eventCaptureWrites = Promise.resolve();
    this.policyEvidenceWrites = Promise.resolve();
    this.preparedPolicyImages = new Map();
    this.setDefaultConfig(options.defaultConfig || {});
  }

  setDefaultConfig(config) {
    const merged = { ...this.defaultConfig, ...config };
    // An incoming legacy setting must still be able to override a previous
    // normalized default. The config reader resolves both names.
    for (const [optionsKey, legacyKey] of [
      ['hfCarryEnabled', 'pausedPackageCarryEnabled'],
      ['hfCarryMaxMs', 'pausedPackageCarryMaxMs'],
      ['resumeVerificationTailMs', 'pausedPackageResumeCarryWindowMs'],
      ['policyMaxInvalidRetries', 'pausedPackageMaxRetries']
    ]) {
      if (Object.hasOwn(config, legacyKey) && !Object.hasOwn(config, optionsKey)) delete merged[optionsKey];
    }
    if (Object.hasOwn(config, 'hfCaptureHz') && !Object.hasOwn(config, 'hfCaptureIntervalMs')) delete merged.hfCaptureIntervalMs;
    const next = sanitizeConfig(merged);
    if (this.running && (next.hfScheduling !== this.defaultConfig.hfScheduling
      || next.hfSubmissionOffsetMs !== this.defaultConfig.hfSubmissionOffsetMs
      || next.hfSubmissionMaxLatenessMs !== this.defaultConfig.hfSubmissionMaxLatenessMs
      || next.hfValidActionTimeoutMs !== this.defaultConfig.hfValidActionTimeoutMs
      || next.hfCaptureIntervalMs !== this.defaultConfig.hfCaptureIntervalMs
      || next.policyMode !== this.defaultConfig.policyMode
      || (isTrivialPolicy(next) && next.trivialOcrHistoryResetMs !== this.defaultConfig.trivialOcrHistoryResetMs)
      || (isTrivialPolicy(next) && next.policyTransport !== this.defaultConfig.policyTransport))) {
      throw new Error('Stop the active session before changing Policy mode, transport or capture scheduling.');
    }
    this.defaultConfig = next;
    // Trivial Policy reads its bounded message window, not the full evidence
    // timeline. Keep hot snapshots bounded; the existing journal + timeline
    // archive retain every record and readSessionHistoryFile restores them.
    this.historyMemoryLimit = isTrivialPolicy(next) ? 32 : 0;
  }

  supportsGamepadPauseResume() {
    // The streamed native X360 helper's Start did not pause GTA in the real run.
    // Use keyboard pause/resume and verify the real menu before accepting a pause.
    return !this.defaultConfig.policyStreaming && super.supportsGamepadPauseResume();
  }

  async pauseGame(reason) {
    this.frameworkRuntime?.setBrainFallbackClockRunning?.(false);
    const requested = await super.pauseGame(reason);
    if (!requested || !this.defaultConfig.policyStreaming) return requested;
    return this.confirmStreamPause(reason);
  }

  async resumeGame(reason, firstState = null) {
    const resumed = await super.resumeGame(reason, firstState);
    this.updateBrainFallbackClock(false);
    return resumed;
  }

  updateBrainFallbackClock(check = true) {
    const active = Boolean(this.fallbackClockReady && !this.gamePaused && !this.paused && !this.stopRequested
      && this.taskState === PAUSED_TASK_STATES.NORMAL_GAMEPLAY && !this.recoveryMode
      && !this.taskTransition && !this.recoveryToolRunning && !this.recoveryAwaitingBrain && !this.finalizing);
    this.frameworkRuntime?.setBrainFallbackClockRunning?.(active);
    if (active && check) {
      this.frameworkRuntime?.checkAutomaticBrainFallback?.();
      const trigger = this.frameworkRuntime?.consumeAutomaticBrainTrigger?.(this.pendingHighFrequencyBarrier);
      if (trigger) this.queueAutomaticBrainFallback(trigger);
    }
  }

  async confirmStreamPause(reason, { allowStopped = false, initialKey = 'p' } = {}) {
    this.gamePaused = false;
    this.phase = 'PAUSE_REQUESTED';
    // A sent key is not a pause acknowledgement, especially during animations.
    // This runs only at an authorized pause barrier; ordinary Policy never waits.
    await sleep(350);
    for (let attempt = 1; attempt <= 2; attempt++) {
      const check = await this.verifyPauseBeforeResume(`pause_ack:${reason}`, attempt);
      if (check.conclusive && check.menuOpen) {
        this.gamePaused = true; this.phase = 'PAUSED_INFERENCE';
        this.frameworkRuntime?.pauseHarness.transition(this.phase, true, { reason, confirmed: true });
        await this.recordHistoryEvent('stream_pause_confirmed', { reason, attempt,
          key: attempt === 1 ? initialKey : 'esc', imagePath: check.observation?.imagePath,
          pausedCutscene: Boolean(check.pausedCutscene) });
        return true;
      }
      const terminal = check.taskResult?.terminal || check.taskResult?.failure || check.taskResult?.success;
      if (!check.conclusive || terminal || (this.stopRequested && !allowStopped) || attempt === 2) {
        throw Error('GTA pause was not confirmed; inputs released and automatic execution remains stopped.');
      }
      await this.recordHistoryEvent('stream_pause_fallback', { reason, from: initialKey, to: 'esc',
        imagePath: check.observation?.imagePath });
      await this.tapKey('esc');
      await sleep(350);
    }
    return false;
  }

  packageCarryAllowed() { return !this.recoveryMode && this.taskState === PAUSED_TASK_STATES.NORMAL_GAMEPLAY; }

  isObservationCurrent(observation) {
    return !this.stopRequested && !this.paused && !this.taskTransition && !this.brainProtectionActive
      && !this.hasPriorityInstructionRefresh()
      && (observation?.captureRecoveryGeneration || 0) === (this.recoveryObservationGeneration || 0)
      && observation?.captureContextEpoch === this.contextEpoch;
  }

  async rebuildPolicyContext(trigger = {}) {
    this.hotSwap?.cancel(`context_rebuild:${trigger.reason || 'unknown'}`);
    this.contextEpoch += 1;
    this.discardPendingCadenceObservation();
    const resumeCadence = this.executor?.fixedCadence && this.executor.running && this.executor.accepting;
    if (this.executor?.fixedCadence && this.executor.running) {
      await this.executor.drainToBoundary('policy_context_rebuilt');
    }
    await super.rebuildPolicyContext(trigger);
    if (resumeCadence && !this.stopRequested && !this.paused && !this.taskTransition) this.executor.resumeAccepting();
    // Sampling is independent: only append a frame when predict submits it.
    this.messages = this.messages.filter((message) => message.role === 'system' || message.role === 'developer' || message.ocrAggregate);
  }

  getStatus() {
    const status = super.getStatus();
    // Data URLs belong in observation events, not every status/history snapshot.
    const { image, imageUrl, liveImageUrl, ...observation } = status.lastObservation || {};
    return { ...status, agentMode: 'high_frequency',
      lastObservation: status.lastObservation ? observation : null,
      manualPauseBoundary: this.manualPauseBoundary || null,
      manualPauseRequestId: this.manualPauseRequestId || 0,
      policyTarget: this.manualPolicy?.status() || null,
      mobilityRecovery: { running: Boolean(this.recoveryToolRunning), calls: this.recoveryCallCount || 0 },
      highFrequency: this.highFrequencyStatus()
    };
  }

  highFrequencyStatus() {
    return {
        requestedScheduling: this.defaultConfig.hfScheduling,
        effectiveScheduling: this.executor?.fixedCadence ? 'fixed_cadence' : 'legacy',
        pendingLatestObservation: Boolean(this.pendingCadenceObservation),
        captureIntervalMs: this.defaultConfig.hfCaptureIntervalMs,
        submissionOffsetMs: this.defaultConfig.hfSubmissionOffsetMs,
        submissionMaxLatenessMs: this.defaultConfig.hfSubmissionMaxLatenessMs,
        validActionTimeoutMs: this.defaultConfig.hfValidActionTimeoutMs,
        stepLimitBasis: this.defaultConfig.policyStreaming ? "generated_responses" : "completed_actions",
        actionStartDelayMs: this.defaultConfig.hfActionStartDelayMs,
        streaming: Boolean(this.defaultConfig.policyStreaming),
        hotSwap: this.hotSwap?.getStatus() || null,
        scheduledSubmissionMisses: this.scheduledSubmissionMisses || 0,
        captureHz: 1000 / this.defaultConfig.hfCaptureIntervalMs,
        chunksPerPackage: this.defaultConfig.hfChunksPerPackage,
        chunkDurationMs: this.defaultConfig.hfChunkDurationMs,
        captureTickCount: this.captureTickCount || 0,
        skippedInferenceTicks: this.skippedInferenceTicks || 0,
        droppedCaptureTicks: this.droppedCaptureTicks || 0,
        inFlightPredictions: this.inFlightPrediction ? 1 : 0,
        contextEpoch: this.contextEpoch || 0,
        executor: this.executor?.getStatus() || null,
        capturePersistence: this.captureStore?.getStatus() || null,
        captureLatencyMs: this.captureLatencyMs || 0, modelLatencyMs: this.modelLatencyMs || 0,
        lastPackageDescription: this.lastPackageDescription || '',
        policyTransport: this.frameworkRuntime?.policyActor?.transport?.effectiveMode || "stateless",
        requestedPolicyTransport: this.defaultConfig.policyTransport
    };
  }

  historySnapshot(extra = {}) {
    return { ...super.historySnapshot(extra), agentMode: 'high_frequency', highFrequency: this.highFrequencyStatus() };
  }

  hasActionCapacity() {
    // Streaming limits generated responses. Active/queued packets are already
    // included in responseCount; reserving them again starves the final beats.
    if (this.defaultConfig.policyStreaming) return !this.atActionLimit();
    const status = this.executor.getStatus();
    const reserved = this.actionCount + (status.active ? 1 : 0) + status.pendingQueueLength;
    return this.defaultConfig.maxStepsUnlimited || reserved < this.defaultConfig.maxSteps;
  }

  atActionLimit() {
    return !this.defaultConfig.maxStepsUnlimited && (this.defaultConfig.policyStreaming ? this.responseCount : this.actionCount) >= this.defaultConfig.maxSteps;
  }

  async captureInitialObservation() {
    for (let attempt = 1; attempt <= 10 && !this.stopRequested; attempt++) {
      try { return await this.capture('Initial high-frequency observation.'); }
      catch (error) {
        if (!error.nonFatalCaptureSkip) throw error;
        await this.recordHistoryEvent('initial_capture_skipped', { attempt, error: error.message });
        if (attempt === 10) throw new Error('Initial observation unavailable after 10 capture deadlines', { cause: error });
        await sleep(this.defaultConfig.hfCaptureIntervalMs);
      }
    }
    return null;
  }

  async capture(text, { deferBrainObservation = false } = {}) {
    const capturePerfStarted = performance.now();
    const captureContextEpoch = this.contextEpoch;
    const captureRecoveryGeneration = this.recoveryObservationGeneration || 0;
    const startedAt = new Date().toISOString();
    const startedActionSeq = this.currentSession?.timeline.actionSeq || 0;
    const activePackageSeq = this.executor?.getStatus().activePackageSeq || null;
    // The outer capture owner publishes only after timing, persistence and
    // context-boundary checks are final. Pending frames remain audit evidence.
    const observation = await super.capture(text, { deferBrainObservation: true });
    this.preparePolicyImage(observation);
    observation.captureReadyPerfMs = performance.now();
    // A capture may overlap an executing package. Completed-action count at
    // acquisition start is conservative; never claim a whole in-flight package
    // occurred before this frame just because disk/OCR processing took time.
    const timing = { startedAt, completedAt: new Date().toISOString(), activePackageSeq,
      completedActionSeqAtStart: startedActionSeq };
    observation.captureTick = ++this.captureTickCount;
    observation.executionAtCapture = timing;
    observation.captureContextEpoch = captureContextEpoch;
    observation.captureRecoveryGeneration = captureRecoveryGeneration;
    observation.captureInvalidated = captureContextEpoch !== this.contextEpoch || this.stopRequested
      || captureRecoveryGeneration !== (this.recoveryObservationGeneration || 0);
    this.captureLatencyMs = performance.now() - capturePerfStarted;
    observation.captureTiming = { ...(observation.captureTiming || {}), totalMs: this.captureLatencyMs };
    const record = this.currentSession?.timeline.records.find((item) => item.type === 'observation'
      && item.observationId === observation.frameworkObservationId);
    if (record) { record.captureInvalidated=observation.captureInvalidated; record.actionSeq = startedActionSeq; record.executionAtCapture = timing; this.currentSession.timeline.touch?.(record); }
    if (this.lastObservation) Object.assign(this.lastObservation, { captureTick: observation.captureTick, executionAtCapture: timing });
    await this.recordObservation(text, observation);
    // Persistence may itself overlap a context change or stop.
    if (!observation.captureInvalidated && (captureContextEpoch !== this.contextEpoch || this.stopRequested
      || captureRecoveryGeneration !== (this.recoveryObservationGeneration || 0))) {
      observation.captureInvalidated = true;
      if (record) record.captureInvalidated = true;
      if (this.lastObservation?.frameworkObservationId === observation.frameworkObservationId) this.lastObservation.captureInvalidated = true;
      await this.recordHistoryEvent('capture_invalidated_before_brain_publish', {
        observationId: observation.frameworkObservationId, captureEpoch: captureContextEpoch, currentEpoch: this.contextEpoch
      });
    }
    if (!deferBrainObservation && !observation.captureInvalidated) {
      this.frameworkRuntime?.publishBrainObservation?.(observation.frameworkObservationId);
    }
    return observation;
  }

  discardPendingCadenceObservation() {
    this.pendingCadenceObservation?.release();
    this.pendingCadenceObservation = null;
  }

  preparePolicyImage(observation) {
    if (!(this.defaultConfig.policyExecutionAcknowledgements || this.defaultConfig.policyStreaming) || !observation?.imagePath) return;
    const imagePath = observation.imagePath;
    const release = this.captureStore.pin(imagePath);
    // Only pixel materialization is speculative: OCR delivery, state/epoch and
    // execution receipts are still bound by the actual request/send boundary.
    const promise = imagePathToDataUrl(imagePath, observation.mimeType).finally(release);
    promise.catch(() => {}); // The owning prediction observes any read failure.
    this.preparedPolicyImages.set(imagePath, promise);
    while (this.preparedPolicyImages.size > 4) this.preparedPolicyImages.delete(this.preparedPolicyImages.keys().next().value);
  }

  async buildRequestMessages(sourceMessages = this.messages) {
    if (!(this.defaultConfig.policyExecutionAcknowledgements || this.defaultConfig.policyStreaming)) return super.buildRequestMessages(sourceMessages);
    const source = [sourceMessages[0], sourceMessages.at(-1)];
    return Promise.all(source.map(async message => ({ role: message.role,
      content: Array.isArray(message.content) ? await Promise.all(message.content.map(async part => part.type === 'local_image'
        ? { type: 'image_url', image_url: { url: await (this.preparedPolicyImages.get(part.imagePath)
          || imagePathToDataUrl(part.imagePath, part.mimeType)) } } : part)) : message.content })));
  }

  deferPolicyEvidence(write) {
    if (!write) return this.policyEvidenceWrites;
    this.policyEvidenceWrites = this.policyEvidenceWrites.then(() => new Promise(resolve => {
      const clock = this.submissionClock;
      const now = performance.now();
      const first = clock?.epochPerfMs + this.defaultConfig.hfSubmissionOffsetMs;
      const next = first + Math.max(0, Math.ceil((now - first) / 200)) * 200;
      // Native completion and a preceding HTTP response often land just before
      // the next submit. Keep evidence work outside this short preparation/send
      // window rather than replacing one synchronous stall with an immediate.
      if (clock && !clock.stopped && Number.isFinite(next) && next - now < 25)
        setTimeout(resolve, Math.max(0, next + 3 - now));
      else setImmediate(resolve);
    }))
      .then(write).catch(error => { this.fail(error); throw error; });
    this.policyEvidenceWrites.catch(() => {}); // Flushed/observed at finalization.
    return this.policyEvidenceWrites;
  }

  offerCadenceObservation(observation) {
    if (this.pendingCadenceObservation) this.skippedInferenceTicks++;
    this.discardPendingCadenceObservation();
    this.pendingCadenceObservation = { observation, release: this.captureStore.pin(observation.imagePath) };
    this.pumpCadencePrediction();
  }

  pumpCadencePrediction() {
    const pending = this.pendingCadenceObservation;
    if (!pending || this.inFlightPrediction || this.stopRequested || this.paused || this.taskTransition || this.brainProtectionActive
      || this.pendingHighFrequencyBarrier || this.hasPriorityInstructionRefresh() || this.finalizing || this.manualPolicy?.phase === 'switching') return;
    if (pending.observation.submissionGate?.closed) { this.discardPendingCadenceObservation(); return; }
    if (!this.isObservationCurrent(pending.observation)) { this.discardPendingCadenceObservation(); return; }
    // The active packet and its queued successor may occupy both execution
    // slots while we prepare the request for the following boundary.
    if ((!pending.observation.submissionGate && !this.executor.hasCapacity()) || !this.hasActionCapacity()) return;
    this.pendingCadenceObservation = null;
    const prediction = this.predict(pending.observation)
      .catch((error) => { if (!this.stopRequested && error.name !== 'AbortError') this.fail(error); })
      .finally(() => {
        pending.release();
        if (this.inFlightPrediction === prediction) this.inFlightPrediction = null;
        this.pumpCadencePrediction();
      });
    this.inFlightPrediction = prediction;
  }

  async runScheduledSubmissions() {
    await this.executor.readyPromise;
    this.scheduledSubmissionMisses = 0;
    const startClock = async () => {
      this.submissionClock = new FixedSubmissionClock({ offsetMs: this.defaultConfig.hfSubmissionOffsetMs,
        actionDelayMs: this.defaultConfig.hfActionStartDelayMs ?? 200, allowLateTimer: this.defaultConfig.policyStreaming,
        maxLatenessMs: this.defaultConfig.hfSubmissionMaxLatenessMs,
        clockOffsetMs: this.executor.clock.offsetMs, onError: error => this.fail(error),
        onCapture: ({ slot, scheduledPerfMs, gate }) => {
          if (this.stopRequested || this.paused || this.taskTransition || this.brainProtectionActive
            || this.hasPriorityInstructionRefresh() || this.atActionLimit() || this.scheduledCapture) return;
          if (performance.now() >= gate.expiresAt) return;
          this.scheduledCapture = (async () => {
            const o = await this.capture('High-frequency scheduled observation.');
            o.captureScheduledPerfMs = scheduledPerfMs;
            o.captureSlotReadyPerfMs = gate.submitAt;
            o.scheduledStartPerfMs = gate.executeAt;
            o.captureTiming = { ...o.captureTiming, capture_slot: slot, capture_scheduled_perf_ms: scheduledPerfMs,
              scheduled_submit_perf_ms: gate.submitAt, scheduled_action_perf_ms: gate.executeAt };
            Object.defineProperty(o, 'submissionGate', { value: gate, enumerable: false });
            if (gate.closed || this.stopRequested || this.paused || this.brainProtectionActive) return;
            this.latestCadenceObservation = o;
            const ownedObservation = this.hotSwap?.rebindObservation(o) || o;
            this.hotSwap?.offerObservation(ownedObservation);
            this.offerCadenceObservation(ownedObservation);
          })().catch(error => {
            if (error.nonFatalCaptureSkip) { this.droppedCaptureTicks++; return; }
            if (!this.stopRequested) this.fail(error);
          }).finally(() => { this.scheduledCapture = null; });
        },
        onSubmit: gate => {
          if (!gate.dispatched) this.scheduledSubmissionMisses++;
          const record = { slot: gate.slot,
            scheduledPerfMs: gate.submitAt, actualPerfMs: performance.now(),
            prepared: gate.armed, sent: gate.dispatched, reason: gate.reason || null,
            scheduledActionPerfMs: gate.executeAt };
          // gate.resolve queues fetch as a microtask. Do not snapshot history in
          // the same clock callback before that continuation can send its body.
          if (this.defaultConfig.policyExecutionAcknowledgements || this.defaultConfig.policyStreaming) this.deferPolicyEvidence(() => this.recordHistoryEvent('fixed_submission_slot', record));
          else void this.recordHistoryEvent('fixed_submission_slot', record).catch(error => this.fail(error));
          if (this.pendingCadenceObservation?.observation.submissionGate === gate) this.discardPendingCadenceObservation();
        } });
      await this.submissionClock.start();
      if (this.executor.streaming) await this.executor.arm(this.submissionClock.epochPerfMs + this.defaultConfig.hfSubmissionOffsetMs);
    };
    const stopClock = async () => {
      this.submissionClock.stop();
      this.discardPendingCadenceObservation();
      await this.scheduledCapture;
      await this.inFlightPrediction;
      this.discardPendingCadenceObservation();
    };
    await startClock();
    try {
      while (!this.stopRequested && !this.atActionLimit()) {
        this.updateBrainFallbackClock();
        if (this.taskTransition) {
          await stopClock();
          await this.taskTransition;
          if (!this.stopRequested) await startClock();
          continue;
        }
        if (this.paused) {
          if (!this.nativeWatchdogTransition?.historyRecovery)
            this.frameworkRuntime.policyActor.stop('manual pause');
          await stopClock();
          await this.handleManualPause();
          if (!this.stopRequested) await startClock();
          continue;
        }
        if (this.pendingHighFrequencyBarrier && !this.hasPriorityInstructionRefresh()) {
          await stopClock();
          const trigger=this.pendingHighFrequencyBarrier;
          this.pendingHighFrequencyBarrier=null;
          let completed=true;
          if(trigger.afterCandidate) {
            await this.executor.waitForIdle();
            completed=this.completedBrainCandidates.delete(trigger.afterCandidate);
            if (this.defaultConfig.policyExecutionAcknowledgements || this.defaultConfig.policyStreaming) {
              trigger.policyEvidence = { ...trigger.policyEvidence, actionExecutionStatus: completed ? 'completed' : 'discarded' };
              if (!completed) await this.recordHistoryEvent('policy_think_admitted_without_action', trigger.policyEvidence);
              completed = true; // The judgment is valid independently of its associated action deadline.
            } else if(!completed) await this.recordHistoryEvent('brain_trigger_discarded', {
              candidateId:trigger.afterCandidate,reason:'action_not_completed'});
            else if (trigger.policyEvidence) trigger.policyEvidence = { ...trigger.policyEvidence, actionExecutionStatus: 'completed' };
          }
          await this.actionWrites;
          if(completed && !this.stopRequested) {
            if (this.hasPriorityInstructionRefresh() && !this.isPolicyCallBrainTrigger(trigger)) await this.startPendingAutomaticBrainFallback();
            else await this.startBrainUpdate(trigger);
          }
          if(!this.stopRequested) await startClock();
          continue;
        }
        if (this.hasPriorityInstructionRefresh()) {
          await stopClock();
          await this.startPendingAutomaticBrainFallback();
          if (!this.stopRequested && !this.paused) await startClock();
          continue;
        }
        await this.startPendingSmallBrainUpdate();
        await this.startPendingAutomaticBrainFallback();
        if(this.brainCycle) {
          const scheduler=this.frameworkRuntime.brainRuntime.scheduler;
          if(scheduler.lastError) {
            if (!this.hotSwap) throw scheduler.lastError;
            await this.discardFailedSmallUpdate(scheduler.lastError);
            continue;
          }
          if(scheduler.pending && this.hotSwap) {
            this.hotSwap.stage(this.brainCycle, scheduler);
          } else if(scheduler.pending) {
            await stopClock();
            // A completing Policy response can supersede this Brain result.
            if(!this.pendingHighFrequencyBarrier && !this.stopRequested) await this.finishBrainUpdate();
            if(!this.stopRequested) await startClock();
            continue;
          }
          const protection = this.brainProtectionReason();
          if (protection) {
            this.submissionClock.stop();
            this.discardPendingCadenceObservation();
            await this.enforceBrainUpdateProtection(protection);
            await this.scheduledCapture;
            this.discardPendingCadenceObservation();
            if (!this.stopRequested && !this.paused && !this.taskTransition) await startClock();
            continue;
          }
        }
        await sleep(20);
      }
    } finally {
      this.submissionClock.stop();
      if (this.defaultConfig.policyStreaming && !this.stopRequested && this.atActionLimit()) {
        await this.inFlightPrediction;
        await this.executor.waitForIdle();
      }
    }
  }

  async run() {
    this.fallbackClockReady = false;
    this.finalizing = false;
    this.nativeWatchdogPausePromise = null;
    this.nativeWatchdogTransition = null;
    this.eventCaptureWrites = Promise.resolve();
    this.policyEvidenceWrites = Promise.resolve();
    this.preparedPolicyImages.clear();
    this.contextEpoch += 1;
    this.captureTickCount = 0;
    this.skippedInferenceTicks = 0;
    this.droppedCaptureTicks = 0;
    this.pendingHighFrequencyBarrier = null;
    this.pendingAutomaticBrainFallback = null;
    this.pendingSmallBrainUpdate = null;
    this.latestCadenceObservation = null;
    this.brainCycle = null;
    this.brainProtectionActive = false;
    this.hotSwap = null;
    this.completedBrainCandidates = new Set();
    this.policyBrainSources = new WeakSet();
    this.inFlightPrediction = null;
    this.actionWrites = Promise.resolve();
    this.manualPauseHandled = false;
    this.nativeWatchdogPausePromise = null;
    this.manualPolicy = null;
    this.recoveryCallCount = 0;
    this.lastRecoveryAt = 0;
    this.recoveryToolRunning = false;
    this.recoveryPromise = null;
    this.recoveryAwaitingBrain = false;
    this.recoveryObservationGeneration = 0;
    try {
      if (this.defaultConfig.policyManualSwitch?.enabled) {
        this.manualPolicy = new (require('./manualPolicySwitch').ManualPolicySwitch)(this);
        await this.manualPolicy.initialize();
      }
      this.captureStore.configure({ ...this.defaultConfig, captureRoot: this.sessionCaptureRoot });
      await this.captureStore.startSession(this.sessionId);
      await this.inputController.ensureStarted();
      this.currentModel = await this.resolveModelWithAbort();
      this.setDefaultConfig({ modelName: this.currentModel });
      this.frameworkRuntime.policyActor.setModel(this.currentModel);
      const fixedCadence = this.defaultConfig.hfScheduling === 'fixed_cadence'
        || (this.defaultConfig.hfScheduling === 'auto' && isTrivialPolicy(this.defaultConfig)
          && this.defaultConfig.hfCaptureIntervalMs === 200
          && this.inputController.supportsRealtimeGamepadPackage?.());
      if (fixedCadence && !this.inputController.supportsFixedCadenceGamepad?.()) {
        throw Error('Fixed cadence needs the updated native Windows gamepad helper; restart the input helper.');
      }
      this.executor = this.executorFactory(this.inputController, {
        fixedCadence, policyStreaming: this.defaultConfig.policyStreaming, watchdogPause: true,
        hfValidActionTimeoutMs: this.defaultConfig.hfValidActionTimeoutMs,
        chunkDurationMs: this.defaultConfig.hfChunkDurationMs, maxPendingPackages: 1,
        sharedProcessing: true, carryEnabled: this.carryEnabled(),
        carryMaxMs: this.defaultConfig.hfCarryMaxMs,
        ...(!fixedCadence && !this.defaultConfig.policyStreaming
          ? { packageEndHoldMs: this.defaultConfig.hfPackageEndHoldMs ?? 100,
            longHoldThresholdChunks: this.defaultConfig.hfLongHoldThresholdChunks ?? 12,
            longHoldReleaseMs: this.defaultConfig.hfLongHoldReleaseMs ?? 100 } : {}),
        actionRuntime: this.frameworkRuntime.actionRuntime,
        isPackageValid: (pkg) => !this.stopRequested && pkg.contextEpoch === this.contextEpoch
          && pkg.contextSignature === this.policyStateSignature()
      });
      this.executor.on('status', () => { if (this.executor.fixedCadence) this.pumpCadencePrediction(); });
      this.executor.on('watchdogTriggered', event => this.handleNativeWatchdogTriggered(event));
      this.executor.on('watchdog', event => this.handleNativeWatchdog(event));
      this.executor.on('nativeHandoff', event => this.deferPolicyEvidence(() => this.recordHistoryEvent('policy_handoff_native', event)));
      this.executor.on('packageComplete', (event) => this.onPackageComplete(event));
      this.executor.on('packagePreempted', (event) => this.onPackageInterrupted(event));
      this.executor.on('packageDiscarded', (event) => {
        this.acknowledgePackage(event.package, 'discarded', { reason: event.reason });
        void this.recordHistoryEvent('policy_action_discarded', event).catch((error) => this.fail(error));
      });
      this.executor.on('log', (message) => this.emit('log', message));
      this.executor.start();
      void this.executor.loopPromise.catch((error) => this.fail(error));
      await require('./screenshotCapture').prepareScreenshotCapture();
      if (!await this.captureInitialObservation()) return;
      await this.taskResultMonitor.waitForIdle();
      if (!this.stopRequested) await this.runBrainBarrier({ reason: 'session_start', initial: true });
      if (!this.stopRequested && this.defaultConfig.policyHotSwapEnabled) {
        this.hotSwap = new HotSwapPolicyCoordinator(this);
        this.hotSwap.initialize();
      }
      this.fallbackClockReady = true;
      this.updateBrainFallbackClock(false);
      if (this.defaultConfig.hfSubmissionOffsetMs) {
        if (!this.executor.fixedCadence) throw Error('Scheduled HTTP submission requires native fixed cadence');
        await this.runScheduledSubmissions();
        return;
      }
      let nextCaptureAt = performance.now();
      let captureReadyLeadMs = null;
      while (!this.stopRequested && !this.atActionLimit()) {
        this.updateBrainFallbackClock();
        if (this.recoveryToolRunning) { await sleep(25); continue; }
        if (await this.manualPolicy?.applyPending()) { nextCaptureAt = performance.now(); continue; }
        if (this.taskTransition) { await this.taskTransition; nextCaptureAt = performance.now(); captureReadyLeadMs = null; continue; }
        if (this.paused) {
          await this.handleManualPause();
          nextCaptureAt = performance.now();
          captureReadyLeadMs = null;
          continue;
        }
        if (this.pendingHighFrequencyBarrier && !this.hasPriorityInstructionRefresh()) {
          const trigger = this.pendingHighFrequencyBarrier;
          this.pendingHighFrequencyBarrier = null;
          // A trigger is attached to its action. That package must execute before
          // the barrier; no further predictions are dispatched once it is queued.
          if (trigger.afterCandidate) {
            await this.executor.waitForIdle();
            if (!this.completedBrainCandidates.delete(trigger.afterCandidate)) {
              await this.recordHistoryEvent('brain_trigger_discarded', { candidateId: trigger.afterCandidate, reason: 'action_not_completed' });
              continue;
            }
            if (trigger.policyEvidence) trigger.policyEvidence = { ...trigger.policyEvidence, actionExecutionStatus: 'completed' };
          }
          await this.actionWrites;
          if (!this.stopRequested) {
            if (this.hasPriorityInstructionRefresh() && !this.isPolicyCallBrainTrigger(trigger)) await this.startPendingAutomaticBrainFallback();
            else await this.startBrainUpdate(trigger);
          }
          nextCaptureAt = performance.now();
          captureReadyLeadMs = null;
          continue;
        }
        await this.startPendingAutomaticBrainFallback();
        if (this.brainCycle) {
          const scheduler = this.frameworkRuntime.brainRuntime.scheduler;
          if (scheduler.lastError) throw scheduler.lastError;
          if (scheduler.pending) {
            await this.finishBrainUpdate();
            nextCaptureAt = performance.now();
            continue;
          }
          const protection = this.brainProtectionReason();
          if (protection) {
            await this.enforceBrainUpdateProtection(protection);
            nextCaptureAt = performance.now();
            captureReadyLeadMs = null;
            continue;
          }
        }
        const interval = this.defaultConfig.hfCaptureIntervalMs;
        const delay = nextCaptureAt - performance.now();
        if (delay > 0) await sleep(Math.min(delay, 50));
        if (this.stopRequested || this.paused || this.pendingHighFrequencyBarrier) continue;
        if (performance.now() < nextCaptureAt) continue;
        let observation;
        try { observation = await this.capture('High-frequency sampled observation; the previous package may still be executing.'); }
        catch (error) {
          if (!error.nonFatalCaptureSkip) throw error;
          this.droppedCaptureTicks++; nextCaptureAt = performance.now() + interval; continue;
        }
        if (this.executor.fixedCadence) {
          captureReadyLeadMs ??= observation.captureReadyPerfMs - nextCaptureAt;
          observation.captureSlotReadyPerfMs = nextCaptureAt + captureReadyLeadMs;
          observation.scheduledStartPerfMs = observation.captureSlotReadyPerfMs + interval;
        }
        nextCaptureAt += interval;
        if (nextCaptureAt < performance.now()) {
          const missed = Math.floor((performance.now() - nextCaptureAt) / interval) + 1;
          this.droppedCaptureTicks += missed;
          nextCaptureAt += missed * interval;
        }
        if (this.executor.fixedCadence) {
          this.offerCadenceObservation(observation);
          continue;
        }
        if (!this.stopRequested && !this.paused && !this.taskTransition && !this.pendingHighFrequencyBarrier
          && this.isObservationCurrent(observation)
          && !this.recoveryAwaitingBrain && !this.inFlightPrediction && this.executor.hasCapacity() && this.hasActionCapacity()) {
          this.inFlightPrediction = this.predict(observation)
            .catch((error) => { if (!this.stopRequested && error.name !== 'AbortError') this.fail(error); })
            .finally(() => { this.inFlightPrediction = null; });
        } else this.skippedInferenceTicks += 1;
      }
    } catch (error) {
      if (!this.stopRequested) this.fail(error);
    } finally {
      this.finalizing = true;
      this.fallbackClockReady = false;
      this.frameworkRuntime?.setBrainFallbackClockRunning?.(false);
      this.pendingAutomaticBrainFallback = null;
      this.submissionClock?.stop();
      await this.scheduledCapture;
      this.discardPendingCadenceObservation();
      this.contextEpoch += 1;
      this.frameworkRuntime?.policyActor.stop('high-frequency finalizing');
      await this.executor?.stop({ reason: 'high-frequency finalizing' });
      // A recovery owns physical input outside the normal Policy queue. Its
      // release and interruption evidence must finish before closing the episode.
      if (this.recoveryPromise) {
        this.frameworkRuntime?.brainActor.stop('high-frequency finalizing');
        await this.recoveryPromise.catch(error => {
          if (error.name !== 'AbortError' || error.cleanupErrors?.length) this.lastError ||= error.message;
        });
      }
      await this.nativeWatchdogPausePromise?.catch(() => {});
      await this.inFlightPrediction;
      await this.hotSwap?.stop('high-frequency finalizing');
      await this.actionWrites;
    }
  }

  async finishSessionHistory(reason) {
    try {
      // Persistence callbacks append evidence; drain them before closing history.
      await this.captureStore.store.scratchQueue;
      if (this.captureStore.store.scratchError) throw this.captureStore.store.scratchError;
      await this.policyEvidenceWrites;
      await this.eventCaptureWrites;
      await super.finishSessionHistory(reason);
    }
    finally { await this.captureStore.finishSession(); }
  }

  archiveEvent(reason, observation = this.lastObservation) {
    if (!observation?.imagePath) return this.eventCaptureWrites;
    this.eventCaptureWrites = this.eventCaptureWrites.then(async () => {
      const archive = await this.captureStore.persistEvent({ imagePath: observation.imagePath,
        mimeType: observation.mimeType, capturedAt: observation.capturedAt, label: reason });
      if (archive) await this.recordHistoryEvent('event_capture', { reason, imagePath: archive.filePath });
    }).catch((error) => this.emit('log', `event capture unavailable: ${error.message}`));
    return this.eventCaptureWrites;
  }

  fail(error) {
    void this.archiveEvent('framework-error');
    this.lastError = error.message || String(error);
    this.emit('errorEvent', { message: this.lastError, stack: error.stack, failure: error.failure || null });
    this.stop('high-frequency error');
  }

  async appendObservation(text, observation) {
    if (this.defaultConfig.policyExecutionAcknowledgements || this.defaultConfig.policyStreaming) {
      // The session backend owns its committed 5/3 window. The workstation
      // retains full evidence in the timeline, not a second growing prompt.
      this.messages = this.messages.filter(m => m.role === 'system' || m.role === 'developer' || m.ocrAggregate);
    }
    return super.appendObservation(text, observation);
  }

  acknowledgePackage(pkg, status, detail = {}) {
    if (!this.defaultConfig.policyExecutionAcknowledgements || !pkg?.requestId) return;
    const receipt = { request_id: pkg.requestId, status, ...detail };
    const queued = this.frameworkRuntime.policyActor.transport.acknowledgeExecution(pkg.policyContextId, receipt);
    this.deferPolicyEvidence(() => this.recordHistoryEvent('policy_execution_receipt', { contextId: pkg.policyContextId,
      candidateId: pkg.candidateId, receipt, queued }));
  }

  policyBrainTrigger(result, observation, { afterCandidate = null, actionExecutionStatus = 'pending', actionDiscardReason = null } = {}) {
    if (!result?.brainRequested) return null;
    const response = result.response || {};
    const raw = typeof response.content === 'string' ? response.content : '';
    let callBrain = false;
    try {
      const complete = !response.truncated && !['length', 'max_tokens'].includes(response.finishReason || response.finish_reason);
      const parsed = extractPolicyResponse(raw, response.thinking, this.defaultConfig.policyRendering?.version);
      callBrain = complete && raw.includes('<|call_brain|>') && parsed.instructionUpdateRequested === true;
    } catch {} // An incomplete/invalid response cannot authorize a full update.
    const source = Object.freeze({ requestId: result.actionPackage?.requestId || response.policyRequestId || null,
      candidateId: result.candidate.candidateId, contextEpoch: this.contextEpoch, sessionId: this.sessionId,
      callBrain, renderingVersion: this.defaultConfig.policyRendering?.version || 'legacy' });
    this.policyBrainSources.add(source);
    const policyEvidence = { candidateId: result.candidate.candidateId,
      observationId: result.candidate.basedOnObservationId, observedAt: observation.capturedAt,
      actionExecutionStatus, ...(actionDiscardReason ? { actionDiscardReason } : {}), policySource: source };
    return { producedPerfMs: performance.now(), reason: callBrain ? 'policy_instruction_refresh' : 'policy_stage_update',
      thinking: result.triggerThinking || '', policyProposal: callBrain ? result.policyProposal || null : null,
      policySource: source, policyEvidence, ...(afterCandidate ? { afterCandidate } : {}) };
  }

  isPolicyCallBrainTrigger(trigger) {
    const source = trigger?.policySource;
    return Boolean(source && this.policyBrainSources.has(source) && source.callBrain
      && source.contextEpoch === this.contextEpoch && source.sessionId === this.sessionId);
  }

  stalePolicyBrainTrigger(trigger) {
    const source = trigger?.policySource;
    return Boolean(source && this.policyBrainSources.has(source)
      && (source.contextEpoch !== this.contextEpoch || source.sessionId !== this.sessionId));
  }

  queuePolicyBrainTrigger(trigger) {
    if (!trigger) return;
    if (this.hotSwap && !this.isPolicyCallBrainTrigger(trigger)) this.pendingSmallBrainUpdate = trigger;
    else this.pendingHighFrequencyBarrier = trigger;
  }

  async startPendingSmallBrainUpdate() {
    const trigger = this.pendingSmallBrainUpdate;
    if (!trigger || this.stopRequested || this.paused || this.taskTransition || this.pendingHighFrequencyBarrier
      || this.hotSwap?.pending?.phase === 'handoff') return;
    this.pendingSmallBrainUpdate = null;
    await this.startBrainUpdate(trigger);
  }

  async discardFailedSmallUpdate(error) {
    this.hotSwap?.cancel('brain_update_failed');
    this.frameworkRuntime.invalidateBrainContext('small_update_failed');
    this.frameworkRuntime.brainRuntime.scheduler.deferCommit = false;
    this.brainCycle = null;
    await this.recordHistoryEvent('asynchronous_brain_update_failed', { error: error.message, activeContinues: true });
  }

  admitThinkWithoutAction(result, observation, reason) {
    if (!(this.defaultConfig.policyExecutionAcknowledgements || this.defaultConfig.policyStreaming) || !result?.brainRequested) return false;
    const trigger = this.policyBrainTrigger(result, observation, { actionExecutionStatus: 'discarded', actionDiscardReason: reason });
    this.queuePolicyBrainTrigger(trigger);
    void this.recordHistoryEvent('policy_think_admitted_without_action', trigger.policyEvidence).catch(error => this.fail(error));
    return true;
  }

  async predict(observation) {
    const signature = this.policyStateSignature();
    if (this.policyContextSignature !== signature) await this.rebuildPolicyContext({ reason: 'state_changed' });
    const epoch = this.contextEpoch;
    const requestActor = this.frameworkRuntime.policyActor;
    const ownership = this.hotSwap?.metadata() || {};
    const savedMessages = observation.submissionGate || this.manualPolicy?.sessionBacked ? structuredClone(this.messages) : null;
    const savedOcr = observation.submissionGate || this.manualPolicy?.sessionBacked ? this.policyOcr.checkpointDelivery() : null;
    // Preparation can run early; only an on-time gate may send the HTTP body.
    await this.appendObservation('', observation);
    const started = performance.now();
    let result, streamBrainPause = null;
    const deferredControlSignals = [];
    if (this.defaultConfig.policyStreaming) Object.defineProperty(observation, 'onPolicyStream', { configurable: true, enumerable: false,
      value: event => {
        if (this.stopRequested || this.paused || epoch !== this.contextEpoch || signature !== this.policyStateSignature()) return;
        if (this.executor.handoffInProgress && ['brain_marker', 'stream_error'].includes(event.type)) {
          const deferred = this.hotSwap.waitForHandoff().then(() => observation.onPolicyStream(event));
          deferredControlSignals.push(deferred);
          deferred.catch(error => this.fail(error));
          return;
        }
        if (event.type === 'request') return this.executor.arm();
        if (event.type === 'stream_error') {
          void this.executor.clearQueue('invalid_or_broken_stream').catch(error => this.fail(error));
          return;
        }
        if (event.type === 'brain_marker') {
          this.hotSwap?.cancel('active_call_brain_marker');
          // Pause on the actual generated call marker; authorize the Brain update
          // only after the complete response passes the normal contract parser.
          this.submissionClock?.stop(); this.discardPendingCadenceObservation();
          streamBrainPause ||= (async () => {
            await this.executor.drainToBoundary('stream_call_brain_marker');
            if (!this.stopRequested && epoch === this.contextEpoch) await this.pauseGame('policy_call_brain_stream');
          })();
          streamBrainPause.catch(error => this.fail(error));
          return;
        }
        if (event.type !== 'state') return;
        this.executor.streamState(event, { requestId: event.requestId, policyContextId: event.contextId,
          ...ownership, contextEpoch: epoch, contextSignature: signature, scheduledStartPerfMs: observation.scheduledStartPerfMs,
          basedOnObservationId: observation.frameworkObservationId, observationCapturedAt: observation.capturedAt,
          captureTiming: observation.captureTiming });
      } });
    try { result = await this.frameworkRuntime.inferPolicy(observation, requestActor); }
    catch (error) {
      if (this.executor.handoffInProgress) await this.hotSwap?.waitForHandoff();
      if (error.nonFatalSlotSkip || error.policyDependencyDropped) {
        if (epoch === this.contextEpoch && savedMessages) {
          this.messages = savedMessages; this.policyOcr.restoreDelivery(savedOcr);
        }
        await this.recordHistoryEvent(error.policyDependencyDropped ? 'policy_generation_skipped' : 'policy_submission_skipped', { slot: observation.submissionGate?.slot,
          observationId: observation.frameworkObservationId, reason: error.message });
        return;
      }
      if (this.stopRequested || epoch !== this.contextEpoch || signature !== this.policyStateSignature()) return;
      throw error;
    } finally {
      const actor = requestActor;
      if (actor) {
        this.responseCount = this.hotSwap?.totalResponses() ?? actor.responseCount;
        this.invalidPackageRetries = actor.invalidPackageRetries;
        this.lastPackageParseError = actor.lastParseError;
      }
    }
    await Promise.all(deferredControlSignals);
    if (streamBrainPause) await streamBrainPause;
    const actor = requestActor;
    if (this.manualPolicy?.rollbackObservation && epoch === this.contextEpoch) {
      this.messages = savedMessages; this.policyOcr.restoreDelivery(savedOcr);
      this.manualPolicy.rollbackObservation = false;
      return;
    }
    // This is queued without performing any snapshot yet; the immediate task
    // cannot run until this continuation has admitted or rejected the packet.
    if (result?.recordEvidence) this.deferPolicyEvidence(result.recordEvidence);
    this.responseCount = this.hotSwap?.totalResponses() ?? actor.responseCount;
    this.invalidPackageRetries = actor.invalidPackageRetries;
    this.lastPackageParseError = actor.lastParseError;
    if (this.stopRequested || this.paused || this.pendingHighFrequencyBarrier || epoch !== this.contextEpoch || signature !== this.policyStateSignature()) {
      await this.recordHistoryEvent('policy_action_discarded', { candidateId: result?.candidate?.candidateId,
        reason: 'context_changed', contextEpoch: epoch });
      if (result) this.acknowledgePackage({ ...result.actionPackage, candidateId: result.candidate.candidateId }, 'discarded', { reason: 'context_changed' });
      return;
    }
    if (!result && this.defaultConfig.policyStreaming) { await this.executor.clearQueue('invalid_or_cancelled_stream'); this.pause('invalid_stream'); }
    if (!result) { if (this.invalidPackageRetries) void this.archiveEvent('invalid-package', observation); return; }
    this.modelLatencyMs = performance.now() - started;
    this.lastPackageDescription = describeHighFrequencyPackage(result.actionPackage);
    if (!this.defaultConfig.policyStreaming && observation.submissionGate && performance.now() >= observation.scheduledStartPerfMs) {
      this.acknowledgePackage({ ...result.actionPackage, candidateId: result.candidate.candidateId }, 'discarded', { reason: 'fixed_execution_deadline_missed' });
      this.admitThinkWithoutAction(result, observation, 'fixed_execution_deadline_missed');
      await this.recordHistoryEvent('policy_action_discarded', { candidateId: result.candidate.candidateId,
        slot: observation.submissionGate.slot, reason: 'fixed_execution_deadline_missed',
        scheduledStartPerfMs: observation.scheduledStartPerfMs, readyPerfMs: performance.now() });
      return;
    }
    const pkg = { ...result.actionPackage, ...ownership, strictDeadline: Boolean(observation.submissionGate), carryEnabled: this.packageCarryAllowed(), contextEpoch: epoch, contextSignature: signature,
      candidateId: result.candidate.candidateId, basedOnObservationId: result.candidate.basedOnObservationId,
      observationCapturedAt: observation.capturedAt, captureTick: observation.captureTick,
      modelLatencyMs: this.modelLatencyMs, firstExecutablePerfMs: performance.now(),
      scheduledStartPerfMs: observation.scheduledStartPerfMs,
      captureSlotReadyPerfMs: observation.captureSlotReadyPerfMs,
      captureReadyPerfMs: observation.captureReadyPerfMs,
      captureTiming: observation.captureTiming,
      clientTiming: result.response.clientTiming, vllmTiming: result.response.vllmTiming,
      requestMetrics: result.response.requestMetrics,
      firstExecutablePackageMs: result.response.firstExecutablePackageMs,
      responseCount: this.responseCount, brainTrigger: this.policyBrainTrigger(result, observation,
        { afterCandidate: result.candidate.candidateId }) };
    if (this.defaultConfig.policyStreaming) {
      // Bind the final validated candidate to any actual prefix execution receipt.
      const queued = this.executor.finalize(pkg.requestId, pkg);
      if (pkg.brainTrigger) this.queuePolicyBrainTrigger(pkg.brainTrigger);
      if (!queued) this.admitThinkWithoutAction(result, observation, 'stream_not_executed');
      this.emitStatus();
      return;
    }
    const queued = this.executor.enqueue(pkg);
    if (!queued) {
      this.acknowledgePackage(pkg, 'discarded', { reason: 'queue_rejected' });
      this.admitThinkWithoutAction(result, observation, 'queue_rejected');
      if (!this.defaultConfig.policyExecutionAcknowledgements) await this.rebuildPolicyContext({ reason: 'generated_package_not_executed' });
      await this.recordHistoryEvent('policy_action_discarded', { candidateId: pkg.candidateId, reason: 'queue_rejected' });
      return;
    }
    if (pkg.brainTrigger) this.queuePolicyBrainTrigger(pkg.brainTrigger);
    const queuedRecord = { candidateId: pkg.candidateId, packageSeq: queued.seq,
      contextEpoch: epoch, observationCapturedAt: pkg.observationCapturedAt,
      scheduledStartPerfMs: pkg.scheduledStartPerfMs, captureReadyPerfMs: pkg.captureReadyPerfMs };
    if (this.defaultConfig.policyExecutionAcknowledgements) this.deferPolicyEvidence(() => this.recordHistoryEvent('package_queued', queuedRecord));
    else await this.recordHistoryEvent('package_queued', queuedRecord);
    this.emitStatus();
  }

  onPackageComplete(event) {
    const pkg = event.package;
    if (pkg.brainTrigger && (!this.hotSwap || this.isPolicyCallBrainTrigger(pkg.brainTrigger))) this.completedBrainCandidates.add(pkg.candidateId);
    const execution = { startedAt: event.startedAt, completedAt: event.completedAt,
      durationMs: event.durationMs, requestedStates: pkg.requestedStates || pkg.states,
      appliedStates: pkg.appliedStates || pkg.states, timing: event.timing };
    const executed = this.frameworkRuntime.executedAction({ candidateId: pkg.candidateId,
      basedOnObservationId: pkg.basedOnObservationId, raw: pkg.raw, actionText: pkg.actionText,
      contextEpoch: pkg.contextEpoch, execution });
    this.actionCount = executed.actionSeq;
    this.acknowledgePackage(pkg, 'completed', { action_seq: executed.actionSeq, completed_at: event.completedAt });
    const record = { at: event.completedAt, actionCount: this.actionCount,
      responseCount: pkg.responseCount, candidateId: pkg.candidateId,
      policyOwnerId: pkg.policyOwnerId, policyOwnershipVersion: pkg.policyOwnershipVersion, policyUpdateVersion: pkg.policyUpdateVersion,
      basedOnObservationId: pkg.basedOnObservationId, executedActionId: executed.actionId,
      action: { type: 'high_frequency_package', raw: pkg.raw, actionText: pkg.actionText,
        chunks: pkg.chunks, states: pkg.states }, actions: pkg.chunks,
      operations: pkg.chunks.map((chunk, operationIndex) => ({ action: chunk, operationIndex })),
      operationCount: pkg.chunks.length, description: describeHighFrequencyPackage(pkg),
      execution, timing: { modelLatencyMs: pkg.modelLatencyMs, executor: event.timing,
        captureTiming: pkg.captureTiming, clientTiming: pkg.clientTiming, vllmTiming: pkg.vllmTiming,
        requestMetrics: pkg.requestMetrics, firstExecutablePackageMs: pkg.firstExecutablePackageMs,
        observationToExecutionStartMs: Date.parse(event.startedAt) - Date.parse(pkg.observationCapturedAt)
      }, highFrequency: true };
    this.emit('action', record);
    this.actionWrites = this.actionWrites.then(() => (this.defaultConfig.policyExecutionAcknowledgements || this.defaultConfig.policyStreaming)
      ? this.deferPolicyEvidence(() => this.recordAction(record)) : this.recordAction(record)).catch((error) => this.fail(error));
    const automatic = this.frameworkRuntime.consumeAutomaticBrainTrigger(pkg.brainTrigger);
    this.queueAutomaticBrainFallback(automatic);
    this.emitStatus();
  }

  queueAutomaticBrainFallback(automatic) {
    const instructionRefresh = this.frameworkRuntime.isAutomaticInstructionRefresh(automatic);
    if (!this.stopRequested && !this.finalizing && !this.taskTransition
      && (instructionRefresh || (!this.pendingHighFrequencyBarrier && !this.brainCycle)) && automatic && !this.atActionLimit()) {
      // Automatic memory maintenance changes no Policy input until its result
      // commits. It must not invalidate an in-flight prediction or stop the clock.
      this.pendingAutomaticBrainFallback = automatic;
    }
  }

  onPackageInterrupted(event) {
    this.acknowledgePackage(event.package, 'discarded', { reason: 'execution_interrupted' });
    const record = { type: 'interrupted_action', at: event.completedAt, actionSeq:this.currentSession?.timeline.actionSeq||0,
      policyOwnerId: event.package.policyOwnerId, policyOwnershipVersion: event.package.policyOwnershipVersion, policyUpdateVersion: event.package.policyUpdateVersion,
      candidateId: event.package.candidateId, requestId: event.package.requestId, policyContextId: event.package.policyContextId,
      basedOnObservationId: event.package.basedOnObservationId,
      requestedPackage: event.package.raw, startedAt: event.startedAt, completedAt: event.completedAt,
      completed: false, timing: event.timing, executedChunkCount: event.timing?.realtimeHelperCompletion?.events?.length ?? null,
      actualSegments: event.timing?.realtimeHelperCompletion?.events || [],
      executionStatus: this.defaultConfig.policyStreaming ? 'partial' : 'interrupted' };
    this.currentSession?.timeline.records.push(record);
    this.actionWrites = this.actionWrites.then(() => (this.defaultConfig.policyExecutionAcknowledgements || this.defaultConfig.policyStreaming)
      ? this.deferPolicyEvidence(() => this.recordHistoryEvent('action_interrupted', record)) : this.recordHistoryEvent('action_interrupted', record))
      .catch((error) => this.fail(error));
  }

  async policyBoundary(reason, { preservePolicySession = false } = {}) {
    this.contextEpoch += 1;
    if (preservePolicySession) this.frameworkRuntime.policyActor.fenceForHandoff(reason);
    else this.frameworkRuntime.policyActor.stop(reason);
    this.executor.accepting = false;
    this.executor.clearQueue(reason);
    await this.executor.drainToBoundary(reason);
    await this.actionWrites;
  }

  async requestPolicyTarget(target) {
    if (!this.running || !this.manualPolicy) throw Error('No active manually switchable traditional Policy');
    return this.manualPolicy.request(target);
  }

  async changePolicyTarget(command, commit) {
    return require('./traditionalControl').changePolicyTarget(this, command, commit);
  }

  async startPendingAutomaticBrainFallback() {
    if (!this.pendingAutomaticBrainFallback || !this.frameworkRuntime || this.stopRequested || this.paused || this.taskTransition) return false;
    const trigger = this.pendingAutomaticBrainFallback;
    const instructionRefresh = this.frameworkRuntime.isAutomaticInstructionRefresh(trigger);
    if (this.pendingHighFrequencyBarrier && (!instructionRefresh || this.isPolicyCallBrainTrigger(this.pendingHighFrequencyBarrier))) return false;
    this.pendingAutomaticBrainFallback = null;
    if (instructionRefresh) {
      if (!this.frameworkRuntime.automaticInstructionRefreshIsDue(trigger)) return false;
      this.pendingHighFrequencyBarrier = null;
      this.pendingSmallBrainUpdate = null;
      await this.startBrainUpdate(trigger);
      return true;
    }
    const scheduler = this.frameworkRuntime.brainRuntime.scheduler;
    // A newer explicit update already fulfils the memory refresh obligation.
    if (this.brainCycle || scheduler.inferencePromise || scheduler.pending) return false;
    await this.startBrainUpdate(trigger);
    return true;
  }

  hasPriorityInstructionRefresh() {
    return this.frameworkRuntime?.isAutomaticInstructionRefresh?.(this.pendingAutomaticBrainFallback)
      && !this.isPolicyCallBrainTrigger(this.pendingHighFrequencyBarrier);
  }

  async startBrainUpdate(trigger) {
    if (this.stalePolicyBrainTrigger(trigger)) {
      await this.recordHistoryEvent('brain_trigger_discarded', { candidateId: trigger.policySource.candidateId, reason: 'stale_policy_source' });
      return;
    }
    this.pendingAutomaticBrainFallback = null;
    const runtime = this.frameworkRuntime;
    let type = require('./brainProtocol').updateType(trigger);
    // A reason string, legacy <i>, or caller-provided type is not evidence that
    // Policy emitted the call marker. Only a parsed source or a framework-owned
    // instruction-age trigger can authorize a complete update.
    const frameworkFull=runtime.isFrameworkFullBrainTrigger?.(trigger)||runtime.isAutomaticInstructionRefresh?.(trigger);
    if (this.isPolicyCallBrainTrigger(trigger) || frameworkFull) type = 'full';
    else if (type === 'full') type = trigger.thinking ? 'small' : 'fallback';
    if (type === 'full' || this.defaultConfig.hfPauseAllBrainUpdates) {
      await this.hotSwap?.waitForHandoff();
      this.hotSwap?.cancel('full_brain_update');
      this.pendingSmallBrainUpdate = null;
      runtime.invalidateBrainContext('superseded by full update');
      runtime.brainRuntime.scheduler.deferCommit = false;
      this.brainCycle = null;
      return this.runBrainBarrier(frameworkFull
        ? trigger : { ...trigger, updateType: type });
    }
    const previous = this.brainCycle;
    if (previous?.type === 'fallback') type = 'fallback';
    if (this.hotSwap) {
      this.hotSwap.cancel('newer_policy_think');
      if (previous) runtime.invalidateBrainContext('newer Policy think');
      this.brainCycle = { type, thinking: trigger.thinking || '', startedAt: previous?.startedAt || Date.now(),
        producedPerfMs: trigger.producedPerfMs || performance.now(), overrides: previous ? previous.overrides + 1 : 0 };
      runtime.brainRuntime.scheduler.deferCommit = true;
      runtime.requestBrainReplan(trigger.reason, { thinking: trigger.thinking,
        policyEvidence: trigger.policyEvidence || null, updateType: type });
      await this.recordHistoryEvent('asynchronous_brain_update_started', { updateType: type,
        thinking: trigger.thinking || '', activeUnchanged: true, ...this.hotSwap.metadata() });
      return;
    }
    // A newer small event retains an unfinished fallback's responsibility.
    if (previous) runtime.invalidateBrainContext('newer Policy think');
    this.brainCycle = { type, startedAt: previous?.startedAt || Date.now(),
      overrides: previous ? previous.overrides + 1 : 0 };
    if (trigger.thinking) {
      await this.policyBoundary('new Policy think');
      if (this.stopRequested) return;
      this.currentSession.brain.think = trigger.thinking;
      await this.rebuildPolicyContext({ reason: 'policy_think_update' });
      this.executor.resumeAccepting();
    }
    runtime.brainRuntime.scheduler.deferCommit = true;
    runtime.requestBrainReplan(trigger.reason, { thinking: trigger.thinking, policyEvidence: trigger.policyEvidence || null, updateType: type });
  }

  async finishBrainUpdate() {
    const cycle = this.brainCycle;
    const scheduler = this.frameworkRuntime.brainRuntime.scheduler;
    if (!cycle || !scheduler.pending) return;
    const recoveryCompleted = scheduler.pending.result?.recoveryCompleted === true;
    if (!recoveryCompleted) await this.policyBoundary('Brain result boundary');
    if (this.stopRequested) return;
    try {
      if (this.brainCycle !== cycle) return;
      if (scheduler.lastError) throw scheduler.lastError;
      scheduler.deferCommit = false;
      if (!scheduler.tryRelease()) throw new Error('Brain result was not eligible for boundary commit.');
      this.brainCycle = null;
      if (!recoveryCompleted) await this.rebuildPolicyContext({ reason: 'brain_update_committed' });
      else await this.recordHistoryEvent('policy_context_preserved_after_recovery', {
        contextId: this.frameworkRuntime.policyActor.transport.sessionId, historyMessages: this.messages.length });
    } finally {
      scheduler.deferCommit = false;
    }
    this.executor.resumeAccepting();
  }

  brainProtectionReason(now = Date.now()) {
    const cycle = this.brainCycle;
    if (!cycle || this.stopRequested || this.paused || this.taskTransition || this.brainProtectionActive) return null;
    if (now - cycle.startedAt >= (this.defaultConfig.brainMaxWaitMs ?? 15000)) return 'wait_limit';
    if (cycle.overrides > 0 && cycle.overrides >= (this.defaultConfig.brainMaxOverrides ?? 3)) return 'override_limit';
    return null;
  }

  async consumeProtectedBrainTrigger() {
    const trigger = this.pendingHighFrequencyBarrier || this.pendingSmallBrainUpdate;
    if (!trigger) return false;
    if (trigger === this.pendingHighFrequencyBarrier) this.pendingHighFrequencyBarrier = null;
    else this.pendingSmallBrainUpdate = null;
    if (trigger.afterCandidate && !this.completedBrainCandidates.delete(trigger.afterCandidate)
      && !(this.defaultConfig.policyExecutionAcknowledgements || this.defaultConfig.policyStreaming)) {
      await this.recordHistoryEvent('brain_trigger_discarded', { candidateId: trigger.afterCandidate, reason: 'action_not_completed' });
      return false;
    }
    await this.startBrainUpdate(trigger);
    return true;
  }

  async enforceBrainUpdateProtection(reason) {
    if (!this.brainCycle || this.stopRequested || this.paused || this.brainProtectionActive) return;
    this.brainProtectionActive = true;
    const wasPaused = this.gamePaused;
    let completed = false;
    try {
      await this.hotSwap?.waitForHandoff();
      if (!this.brainCycle || this.stopRequested || this.paused) return;
      this.hotSwap?.cancel('brain_protection');
      // Stop new input, finish the active package and release. Keep a healthy
      // in-flight Policy session until the Brain result is known: it may choose
      // the recovery tool, which must preserve that original dialogue.
      this.brainProtectionBoundary=(async()=>{
        await this.settleModelRequestsForProtection?.('brain_protection');
        this.executor.accepting = false;
        this.executor.clearQueue('brain_protection');
        await this.executor.drainToBoundary('brain_protection');
        await this.actionWrites;
        if (this.stopRequested || this.paused) return;
        await this.pauseGame(`brain_protection:${reason}`);
      })();
      await this.brainProtectionBoundary;
      if(this.stopRequested||this.paused)return;
      if (!this.gamePaused && !this.stopRequested && !this.paused) throw Error('Brain protection pause was not confirmed.');
      await this.recordHistoryEvent('brain_update_protection_started', { reason,
        updateType: this.brainCycle.type, overrides: this.brainCycle.overrides,
        elapsedMs: Date.now() - this.brainCycle.startedAt });
      while (!this.stopRequested && !this.paused && !this.taskTransition && this.brainCycle) {
        if (this.hasPriorityInstructionRefresh()) await this.startPendingAutomaticBrainFallback();
        else await this.consumeProtectedBrainTrigger();
        if (!this.brainCycle || this.stopRequested || this.paused) break;
        const scheduler = this.frameworkRuntime.brainRuntime.scheduler;
        if (scheduler.lastError) throw scheduler.lastError;
        if (scheduler.pending) {
          await this.finishBrainUpdate();
          completed = !this.stopRequested && !this.paused;
          break;
        }
        await sleep(20);
      }
      if (!this.brainCycle && !this.stopRequested && !this.paused) completed = true;
      if (completed) await this.recordHistoryEvent('brain_update_protection_completed', { reason });
    } finally {
      this.brainProtectionBoundary=null;
      this.brainProtectionActive = false;
      if (completed && !this.stopRequested && !this.paused && !this.taskTransition) {
        if (!wasPaused && this.gamePaused) await this.resumeGame(`brain_protection:${reason}`);
        this.executor.resumeAccepting();
      }
    }
  }

  async runBrainBarrier(trigger) {
    // Pause-all routes small/fallback updates here too. Routing them back to
    // startBrainUpdate would cycle without ever starting the Brain request.
    if (!trigger.initial && !isTrivialPolicy(this.defaultConfig) && !this.isPolicyCallBrainTrigger(trigger)
      && !this.frameworkRuntime?.isFrameworkFullBrainTrigger?.(trigger)
      && !this.frameworkRuntime?.isAutomaticInstructionRefresh?.(trigger)
      && !this.defaultConfig.hfPauseAllBrainUpdates) {
      return this.startBrainUpdate(trigger);
    }
    if (isTrivialPolicy(this.defaultConfig)) {
      await this.policyBoundary('trivial context reset');
      if (trigger.initial) await this.initialObservationReady?.(this.lastObservation);
      if (this.stopRequested) return;
      await this.rebuildPolicyContext(trigger);
      this.executor.resumeAccepting();
      return;
    }
    await this.hotSwap?.waitForHandoff();
    this.hotSwap?.cancel('brain_barrier');
    if (!trigger.initial && this.defaultConfig.mobilityRecovery?.enabled) {
      // Preserve the original dialogue until the Brain result is known. A
      // normal state commit still rebuilds it in the shared barrier below.
      const captureBefore = this.captureBrainBeforeRelease?.(trigger);
      await require('./traditionalControl').settlePolicyForRecovery(this, 'brain_recovery_boundary_release', undefined,
        captureBefore ? async () => {
          this.pendingPauseReason = `brain:${trigger.reason}`;
          await this.capture('Completed-action Brain observation before input release.');
        } : undefined);
      if (this.stopRequested) return;
      if (!captureBefore) {
        this.pendingPauseReason = `brain:${trigger.reason}`;
        await this.capture('Observation at completed-action Brain barrier.');
      }
      await this.taskResultMonitor.waitForIdle();
      if (this.stopRequested) return;
      // A Policy small update owns its normal first rebuild, independently of
      // whether Brain subsequently returns state or chooses the recovery tool.
      if (require('./brainProtocol').updateType(trigger) === 'small' && trigger.thinking
        && this.currentSession.brain.think !== trigger.thinking) {
        this.currentSession.brain.think = trigger.thinking;
        await this.rebuildPolicyContext({ reason: 'policy_think_update' });
      }
      await super.runBrainBarrier(trigger);
      if (!this.stopRequested) this.executor.resumeAccepting();
      return;
    }
    this.contextEpoch += 1;
    this.frameworkRuntime.policyActor.stop('brain barrier');
    if (this.captureBrainBeforeRelease?.(trigger)) {
      // Finish the selected action; do not synthesize or prolong its carry.
      this.executor.accepting = false;
      this.executor.clearQueue('brain barrier');
      let captureError;
      try {
        await this.executor.waitForIdle();
        await this.actionWrites;
        this.pendingHighFrequencyBarrier = null;
      this.pendingAutomaticBrainFallback = null;
        if (!this.stopRequested) {
          // An already executing package may have changed carry while we drained.
          const stillAiming = this.captureBrainBeforeRelease?.(trigger);
          if (!stillAiming) await this.executor.drainToBoundary('brain barrier');
          if (!this.stopRequested) {
            this.pendingPauseReason = `brain:${trigger.reason}`;
            await this.capture(stillAiming ? 'Completed-action Brain observation before input release.'
              : 'Observation at completed-action Brain barrier.');
          }
        }
      } catch (error) { captureError = error; }
      // Capture, pause or stop failures must still attempt independent release.
      try { await this.executor.drainToBoundary('brain barrier'); }
      catch (error) {
        if (!captureError) throw error;
        captureError.cleanupErrors = [...(captureError.cleanupErrors || []),
          { phase: 'brain_boundary_release', message: error.message }];
      }
      if (captureError) throw captureError;
    } else {
      await this.executor.drainToBoundary('brain barrier');
      await this.actionWrites;
      this.pendingHighFrequencyBarrier = null;
      this.pendingAutomaticBrainFallback = null;
      if (this.stopRequested) return;
      if (!trigger.initial) {
        this.pendingPauseReason = `brain:${trigger.reason}`;
        await this.capture('Observation at completed-action Brain barrier.');
      }
    }
    await this.taskResultMonitor.waitForIdle();
    if (this.stopRequested) return;
    await super.runBrainBarrier(trigger);
    if (this.stopRequested) return;
    // Resume/menu verification captures never replace gameplay observations.
    await this.capture('Fresh gameplay observation after Brain barrier.');
    this.executor.resumeAccepting();
  }

  handleNativeWatchdogTriggered(event) {
    const key = event.watchdogId || (event.epoch ? `${event.epoch}:${event.releasedHelperMs}` : null);
    if (key && this.nativeWatchdogTransition?.key === key) return this.nativeWatchdogPausePromise;
    const canAutoRecover = !this.paused && !this.stopRequested;
    const recoveryEpoch = this.watchdogRecoveryEpoch || 0;
    let complete;
    const pauseComplete = new Promise(resolve => { complete = resolve; });
    const transition = { key, complete, canAutoRecover, recoveryEpoch, completed: false };
    this.nativeWatchdogTransition = transition;
    // Establish the wait barrier before setting paused: handleManualPause may
    // run before the native P-key hold completes, and must not toggle or resume.
    this.nativeWatchdogPausePromise = pauseComplete.then(async completedEvent => {
      if (completedEvent.pauseCompleted === false) throw Error(`Native watchdog pause failed: ${completedEvent.pauseError || 'not completed'}`);
      await transition.triggerEvidence;
      await this.recordHistoryEvent('policy_valid_action_watchdog', completedEvent);
      const confirmed = await this.confirmStreamPause('native_valid_action_watchdog', {
        allowStopped: true, initialKey: completedEvent.pauseSent ? 'p' : null
      });
      if (confirmed && canAutoRecover && this.paused && this.defaultConfig?.hfWatchdogAutoResume && !this.stopRequested
        && recoveryEpoch === (this.watchdogRecoveryEpoch || 0))
        this.pendingWatchdogRecovery = { retiring: transition.retiring, history: transition.historyRecovery,
          signature: transition.signature, timeoutMs: completedEvent.timeoutMs, epoch: recoveryEpoch };
      return confirmed;
    });
    void this.nativeWatchdogPausePromise.catch(error => this.fail?.(error));
    // All operations here latch synchronously. Native still holds its scheduler
    // lock while pressing P; clearQueue sends asynchronously and is not awaited.
    this.submissionClock?.stop();
    this.discardPendingCadenceObservation?.();
    const actor = this.frameworkRuntime?.policyActor;
    let retiring;
    if (canAutoRecover && this.defaultConfig?.hfWatchdogAutoResume && this.defaultConfig?.policyStreaming
      && !this.defaultConfig?.policyExecutionAcknowledgements && actor?.beginHistoryRecovery) {
      // Keep the authoritative generated dialogue, even for unexecuted actions.
      // No reset/cancel-turn rollback may race the backend recovery transaction.
      transition.signature = this.policyStateSignature();
      try {
        const state = require('./policyHistoryRecovery').stateFromSystem(this.messages[0]?.content);
        transition.historyRecovery = actor.beginHistoryRecovery(state);
        retiring = transition.historyRecovery.promise;
      } catch (error) {
        retiring = Promise.reject(error);
        transition.historyRecovery = { error };
      }
    } else retiring = actor?.cancelUnstartedGeneration?.('watchdog_retire_context', true);
    if (retiring) retiring.catch(() => {});
    transition.retiring = retiring;
    this.pause(`policy_no_valid_action_${event.timeoutMs ?? this.defaultConfig?.hfValidActionTimeoutMs ?? 1000}ms`);
    transition.triggerEvidence = Promise.resolve(this.recordHistoryEvent('policy_valid_action_watchdog_triggered', event));
    void transition.triggerEvidence.catch(error => this.fail?.(error));
    return this.nativeWatchdogPausePromise;
  }

  handleNativeWatchdog(event) {
    // Older helpers emit only the completed event; synthesize the same latch.
    const key = event.watchdogId || (event.epoch ? `${event.epoch}:${event.releasedHelperMs}` : null);
    if (!key || this.nativeWatchdogTransition?.key !== key)
      HighFrequencyAgent.prototype.handleNativeWatchdogTriggered.call(this, event);
    const transition = this.nativeWatchdogTransition;
    if (!transition.completed) { transition.completed = true; transition.complete(event); }
    return this.nativeWatchdogPausePromise;
  }

  pause(reason = 'manual') {
    if (!String(reason).startsWith('policy_no_valid_action_')) {
      this.watchdogRecoveryEpoch = (this.watchdogRecoveryEpoch || 0) + 1;
      this.pendingWatchdogRecovery = null;
    }
    if (!this.running || this.stopRequested || this.paused) return this.getStatus();
    this.frameworkRuntime?.setBrainFallbackClockRunning?.(false);
    this.hotSwap?.cancel('manual_or_watchdog_pause');
    this.paused = true;
    this.manualPauseReason = reason;
    this.manualPauseRequestId = (this.manualPauseRequestId || 0) + 1;
    this.manualPauseBoundary = null;
    this.contextEpoch += 1;
    const preserveHistory = String(reason).startsWith('policy_no_valid_action_')
      && this.nativeWatchdogTransition?.historyRecovery;
    if (!preserveHistory) {
      this.frameworkRuntime?.policyActor.stop('manual pause');
      this.policyContextSignature = null;
    }
    this.executor?.clearQueue('manual pause');
    this.emitStatus();
    return this.getStatus();
  }

  async handleManualPause() {
    await this.inFlightPrediction;
    await this.executor.drainToBoundary('manual pause');
    let reviewFrame = null;
    // Diagnostic review is opt-in. Ordinary manual pause retains its behavior.
    // Capture normal gameplay after input release, before opening the menu.
    if (!this.stopRequested && this.manualPauseReason === 'observer_review') {
      if (this.gamePaused) throw new Error('Observer review needs a fresh gameplay boundary');
      const observation = await this.capture('Released gameplay boundary for external progress review.');
      reviewFrame = { requestId: this.manualPauseRequestId, contextEpoch: this.contextEpoch, observationId: this.observationId,
        imagePath: observation.imagePath, capturedAt: observation.capturedAt };
    }
    if (this.nativeWatchdogPausePromise) await this.nativeWatchdogPausePromise;
    else if (!this.stopRequested) await this.pauseGame('manual');

    if (reviewFrame && !this.stopRequested && this.paused) {
      if (!this.gamePaused) throw new Error('Observer review pause was not confirmed');
      this.manualPauseBoundary = { ...reviewFrame, readyAt: new Date().toISOString() };
      await this.recordHistoryEvent('observer_review_boundary', this.manualPauseBoundary);
      this.emitStatus();
    }
    const recovery = this.pendingWatchdogRecovery;
    this.pendingWatchdogRecovery = null;
    let recoveredHistory = null;
    const historyStillCurrent = () => {
      const actor = this.frameworkRuntime.policyActor;
      return actor === recoveredHistory?.recovery.actor
        && actor.contextEpoch === recoveredHistory.recovery.actorEpoch
        && actor.transport.sessionId === recoveredHistory.recovery.sessionId
        && this.policyStateSignature() === recoveredHistory.signature;
    };
    if (recovery && !this.stopRequested && this.paused) {
      try {
        const receipt = recovery.retiring ? await recovery.retiring : null;
        if (!recovery.history) {
          const retired = await this.frameworkRuntime.policyActor.transport.waitForRelease();
          if (retired.pending.length) throw Error('Old Policy session release was not acknowledged');
        }
        const health = await (this.checkRecoveryHealth || require('./llmClient').checkPolicyServiceHealth)(this.hotSwap?.activeConfig() || this.defaultConfig);
        if (this.stopRequested) return;
        if ((recovery.epoch || 0) !== (this.watchdogRecoveryEpoch || 0))
          throw Error('Automatic recovery cancelled by a subsequent manual pause');
        if (recovery.history) {
          const actor = this.frameworkRuntime.policyActor;
          if (actor !== recovery.history.actor || recovery.signature !== this.policyStateSignature())
            throw Error('Authoritative Policy state changed during history recovery');
          actor.finishHistoryRecovery(recovery.history);
          if (recovery.history.remote) {
            this.policyOcr.recoverCommittedState(receipt.ocr_state);
            this.messages = this.messages.filter(m => m.role === 'system' || m.role === 'developer' || m.ocrAggregate);
            this.policyContextSignature = recovery.signature;
            recoveredHistory = { recovery: recovery.history, signature: recovery.signature, receipt };
          } else await this.rebuildPolicyContext({ reason: 'watchdog_before_first_dispatch' });
        } else await this.rebuildPolicyContext({ reason: 'watchdog_recovery' });
        await this.recordHistoryEvent('watchdog_recovery_ready', { health, timeoutMs: recovery.timeoutMs,
          ...(recoveredHistory ? { history: receipt } : {}) });
        if (this.stopRequested) return;
        if ((recovery.epoch || 0) !== (this.watchdogRecoveryEpoch || 0))
          throw Error('Automatic recovery cancelled by a subsequent manual pause');
        if (recoveredHistory && !historyStillCurrent())
          throw Error('Authoritative Policy state changed before automatic resume');
        this.resume('watchdog_recovery');
      } catch (error) {
        recoveredHistory = null;
        await this.recordHistoryEvent('watchdog_recovery_failed', { message: error.message, remainsPaused: true });
        this.emit('log', `Watchdog recovery failed; game remains paused: ${error.message}`);
      }
    }
    await this.waitWhilePaused();
    this.manualPauseBoundary = null;
    if (this.stopRequested) return;
    if (recoveredHistory && !historyStillCurrent()) {
      recoveredHistory = null;
      this.pause('history_recovery_state_changed');
      await this.recordHistoryEvent('watchdog_recovery_failed', {
        message: 'Authoritative Policy state changed before game resume', remainsPaused: true });
      await this.waitWhilePaused();
      if (this.stopRequested) return;
    }
    await this.resumeGame('manual');
    if (this.stopRequested) return;
    await this.capture('Fresh observation after manual resume.');
    if (recoveredHistory) {
      if (!historyStillCurrent())
        throw Error('Recovered Policy history changed before gameplay resumed');
      // The new image is appended by predict(). Rebuilding here would silently
      // delete the session that was just recovered and warmed on both routes.
      await this.recordHistoryEvent('policy_history_recovery_resumed', {
        contextId: recoveredHistory.recovery.sessionId, recoveryId: recoveredHistory.recovery.recoveryId,
        historyTurns: recoveredHistory.receipt.history_turns, historySha256: recoveredHistory.receipt.history_sha256 });
    } else await this.rebuildPolicyContext({ reason: 'manual_resume' });
    this.nativeWatchdogPausePromise = null;
    this.nativeWatchdogTransition = null;
    this.executor.resumeAccepting();
  }

  async handleTaskResult(result, item = {}) {
    if (this.finalizing) return;
    if (!this.running || item.metadata?.sessionId !== this.sessionId || result.ok === false || result.skipped) {
      return super.handleTaskResult(result, item);
    }
    const state = classifyPausedTaskState(result);
    const changes = state !== this.taskState && (state !== PAUSED_TASK_STATES.UNKNOWN || this.recoveryMode);
    if (!changes) return super.handleTaskResult(result, item);
    this.frameworkRuntime?.setBrainFallbackClockRunning?.(false);
    void this.archiveEvent('task-state-change');
    await this.hotSwap?.waitForHandoff();
    this.hotSwap?.cancel('task_state_changed');
    this.frameworkRuntime.invalidateBrainContext('task state changed');
    this.frameworkRuntime.brainRuntime.scheduler.deferCommit = false;
    this.brainCycle = null;
    this.contextEpoch += 1;
    this.frameworkRuntime?.policyActor.resetContext('task state changed');
    this.executor.accepting = false;
    this.executor.clearQueue('task state changed');
    this.pendingHighFrequencyBarrier = null;
    this.pendingAutomaticBrainFallback = null;
    this.taskTransition = (async () => {
      // Stop hardware execution before the base recovery code resets inputs.
      await this.executor.stop({ reason: 'task state changed' });
      await this.inFlightPrediction;
      await super.handleTaskResult(result, item);
      await this.rebuildPolicyContext({ reason: 'task_state_changed' });
      if (!this.stopRequested) {
        this.executor.start();
        void this.executor.loopPromise.catch((error) => this.fail(error));
      }
    })();
    try { await this.taskTransition; } finally { this.taskTransition = null; }
  }

  stop(reason = 'manual') {
    this.frameworkRuntime?.setBrainFallbackClockRunning?.(false);
    this.hotSwap?.cancel(reason);
    this.contextEpoch += 1;
    this.pendingAutomaticBrainFallback = null;
    void this.executor?.stop({ reason }).catch((error) => { this.lastError ||= error.message; });
    return super.stop(reason);
  }
}

module.exports = { HighFrequencyAgent };

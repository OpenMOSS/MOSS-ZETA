const { isTrivialPolicy, loadTrivialPolicyPrompt, buildTrivialSystemMessage, trivialOcrSnapshot, compactTrivialOcrHistory } = require("./trivialPolicy");
const { initialBrainStateFor } = require("./initialBrainState");
const { buildPolicySystemMessage } = require("./policyContext");
const { AsyncPolicyOcr } = require("./asyncPolicyOcr");
const { EventEmitter } = require("node:events");
const { sanitizeConfig, publicConfig, DEFAULT_MODEL_WAIT_RESUME_DELAY_MS } = require("./config");
const { resolveModel, chatCompletion } = require("./llmClient");
const { captureObserve: defaultCaptureObserve } = require("./screenshotCapture");
const { CAPTURE_ROOT, createSessionId, imagePathToDataUrl } = require("./captureStore");
const { SESSION_HISTORY_ROOT, createSessionHistory } = require("./sessionHistoryStore");
const { sessionOutputRoots } = require("./projectPaths");
const { recognizeScreenshotText } = require("./ocrClient");
const { TaskResultMonitor, shouldStopForTaskResult } = require("./taskResultMonitor");
const { sleep } = require("../../cadence/src/actionExecutor");
const { attachMinimapCrop, minimapHistoryMetadata } = require("./minimapCrop");
const { SharedDocuments } = require("./sharedDocuments");
const { SessionState, readContinuationHandoff, writeContinuationHandoff } = require("./sessionState");
const { BrainActor } = require("./brainActor");
const { PolicyActor } = require("./policyActor");
const { validatePolicyCheckpoint, policyCheckpointConfig } = require("./policyCheckpoint");
const { FrameworkRuntime } = require("./frameworkRuntime");
const { detectGtaPauseMenu } = require("./pauseMenuDetector");
const { PAUSED_TASK_STATES, classifyPausedTaskState, isRecoveryTaskState } = require("./pausedTaskState");

const PAUSE_KEY = "p";
const RESUME_KEY = "esc";
const PAUSE_GAMEPAD_BUTTON = "start";
const RESUME_GAMEPAD_BUTTON = "b";
const PAUSE_KEY_HOLD_MS = 150;
const MODEL_REQUEST_MAX_RETRIES = 2;
const MODEL_REQUEST_RETRY_BASE_DELAY_MS = 500;
const RESUME_VERIFICATION_MAX_ATTEMPTS = 3;

function cloneJson(value) {
  if (value === undefined) {
    return null;
  }
  return JSON.parse(JSON.stringify(value));
}

// Browser-native pause state is control metadata, never model evidence. An
// advertised but missing/unknown observation must not fall back to empty OCR.
function nativePauseMenuVerification(observation, expected = false) {
  const menu = observation?.nativePauseMenu;
  if (!expected && observation?.nativePauseMenuObservation !== true && menu == null) return null;
  const known = menu?.known === true && typeof menu.active === "boolean"
    && menu.source === "native_menu_flags" && typeof observation.hostEpoch === "string"
    && observation.hostEpoch.length > 0 && menu.hostEpoch === observation.hostEpoch
    && typeof menu.observedAt === "string" && Number.isFinite(Date.parse(menu.observedAt));
  return {
    menuOpen: known && menu.active,
    conclusive: known,
    matchedMarkers: known && menu.active ? ["native_pause_menu_active"] : [],
    hasMenuFooter: false,
    text: "",
    verificationSource: "native_menu_flags"
  };
}

function observationForHistory(observation) {
  const { buffer, ...metadata } = observation || {};
  if (metadata.navigation?.minimap) {
    metadata.navigation = {
      ...metadata.navigation,
      minimap: minimapHistoryMetadata(observation)
    };
  }
  return cloneJson(metadata);
}

function maxStepsLabel(config) {
  return config.maxStepsUnlimited ? "unlimited" : String(config.maxSteps);
}

function reachedMaxSteps(config, count) {
  return !config.maxStepsUnlimited && count >= config.maxSteps;
}

function nonNegativeInteger(value, fallback) {
  const number = Math.trunc(Number(value ?? fallback));
  return Number.isFinite(number) && number >= 0 ? number : fallback;
}

class AgentRuntime extends EventEmitter {
  constructor(options = {}) {
    super();
    this.agentMode = "high_frequency";
    this.chatCompletion = options.chatCompletion || chatCompletion;
    this.resolveModel = options.resolveModel || resolveModel;
    this.inputController = options.inputController;
    this.captureObserve = options.captureObserve || defaultCaptureObserve;
    this.recognizeScreenshotText = options.recognizeScreenshotText || recognizeScreenshotText;
    this.verifyPauseBeforeResume = options.verifyPauseBeforeResume || this.verifyPauseBeforeResume;
    this.systemPromptProvider = options.systemPromptProvider;
    this.defaultConfig = sanitizeConfig(options.defaultConfig || {});
    this.running = false;
    this.stopRequested = false;
    this.paused = false;
    this.pauseWaiters = [];
    this.gamePaused = false;
    this.phase = "IDLE";
    this.pendingPauseReason = null;
    this.observationId = 0;
    this.actionCount = 0;
    this.responseCount = 0;
    this.invalidPackageRetries = 0;
    this.lastPackageParseError = "";
    this.currentModel = "";
    this.lastObservation = null;
    this.lastError = null;
    this.lastStopReason = null;
    this.taskState = PAUSED_TASK_STATES.NORMAL_GAMEPLAY;
    this.recoveryMode = false;
    this.recoveryEpoch = 0;
    this.startedAt = null;
    this.stoppedAt = null;
    this.sessionId = null;
    this.sessionHistory = null;
    this.historyWriteFailed = false;
    this.sessionCaptureRoot = CAPTURE_ROOT;
    this.sessionHistoryRoot = SESSION_HISTORY_ROOT;
    this.systemPrompt = "";
    this.taskInstruction = "";
    this.recoveryCallCount=0;this.lastRecoveryAt=0;this.recoveryToolRunning=false;
    this.recoveryAwaitingBrain=false;this.recoveryPromise=null;
    this.messages = [];
    this.policyOcr = new AsyncPolicyOcr();
    this.compaction = null;
    this.pendingSharedKnowledge = [];
    this.modelRequestMaxRetries = nonNegativeInteger(options.modelRequestMaxRetries, MODEL_REQUEST_MAX_RETRIES);
    this.modelRequestRetryBaseDelayMs = nonNegativeInteger(options.modelRequestRetryBaseDelayMs, MODEL_REQUEST_RETRY_BASE_DELAY_MS);
    this.runPromise = null;
    this.startPromise = null;
    this.abortController = null;
    this.sharedDocuments = null;
    this.currentSession = null;
    this.lastSessionState = null;
    this.frameworkRuntime = null;
    this.brainChatCompletion = options.brainChatCompletion;
    this.brainResolveModel = options.brainResolveModel;
    this.taskResultMonitor = new TaskResultMonitor({
      getConfig: () => this.defaultConfig,
      isEnabled: (config) => Boolean(config.taskResultOcrEnabled),
      onResult: (result, item) => this.handleTaskResult(result, item),
      onTerminal: (result, item) => {
        if (this.running && item.metadata?.sessionId === this.sessionId) this.emit("taskTerminal", { result, item });
      },
      onLog: (message) => this.emit("log", message),
      onStatus: () => this.emitStatus()
    });
  }

  getStatus() {
    return {
      agentMode: this.agentMode,
      running: this.running,
      stopRequested: this.stopRequested,
      paused: this.paused,
      modelWaitPaused: this.gamePaused,
      gamePaused: this.gamePaused,
      phase: this.phase,
      observationId: this.observationId,
      actionCount: this.actionCount,
      responseCount: this.responseCount,
      stepCount: this.actionCount,
      maxSteps: this.defaultConfig.maxSteps,
      maxActions: this.defaultConfig.maxSteps,
      maxStepsUnlimited: this.defaultConfig.maxStepsUnlimited,
      maxActionsUnlimited: this.defaultConfig.maxStepsUnlimited,
      modelName: this.currentModel || this.defaultConfig.modelName,
      config: publicConfig(this.defaultConfig),
      hasObservation: Boolean(this.lastObservation),
      lastObservation: this.lastObservation,
      lastError: this.lastError,
      lastStopReason: this.lastStopReason,
      startedAt: this.startedAt,
      stoppedAt: this.stoppedAt,
      sessionId: this.sessionId,
      captureRoot: this.sessionCaptureRoot,
      sessionHistoryRoot: this.sessionHistoryRoot,
      sessionHistoryPath: this.sessionHistory?.filePath || null,
      policyContext: { contextId: this.frameworkRuntime?.policyActor?.transport?.sessionId || null,
        effectiveTransport: this.frameworkRuntime?.policyActor?.transport?.effectiveMode || "stateless",
        requestedTransport: this.defaultConfig.policyTransport, keepRounds: this.defaultConfig.policyWindowKeepRounds || 3 },
      contextMessages: this.messages.length,
      contextImages: this.countContextImages(),
      contextMinimapImages: this.countContextImages("minimap"),
      compaction: cloneJson(this.compaction),
      contextImagePolicy: {
        fullFrameLimit: this.contextObservationLimit(),
        minimapPolicy: "paired_with_full_frame",
        eviction: "oldest_observation_and_replies",
        prunedImageSummary: false
      },
      planCount: 0,
      lastPlan: null,
      noActionRetryCount: this.invalidPackageRetries,
      noActionMaxRetries: this.defaultConfig.policyMaxInvalidRetries,
      taskResult: this.taskResultMonitor.getStatus(),
      taskState: this.taskState,
      recoveryMode: this.recoveryMode,
      recoveryEpoch: this.recoveryEpoch,
      framework: this.frameworkRuntime?.getStatus() || this.lastSessionState || null
    };
  }

  emitStatus() {
    // Headless evaluation has no status subscriber; avoid constructing and
    // cloning the complete framework timeline for an event nobody consumes.
    if (this.listenerCount("status")) this.emit("status", this.getStatus());
  }

  async start(config = {}, options = {}) {
    if (this.startPromise) {
      return this.startPromise;
    }
    if (this.running) {
      return this.getStatus();
    }
    this.startPromise = this.startInternal(config, options);
    try {
      return await this.startPromise;
    } finally {
      this.startPromise = null;
    }
  }

  async startInternal(config = {}, options = {}) {
    if (this.running) {
      return this.getStatus();
    }

    this.initialObservationReady = options.initialObservationReady || null;
    this.policyOcr.reset();
    this.policyCheckpoint = options.policyCheckpoint ? validatePolicyCheckpoint(options.policyCheckpoint) : null;
    if (this.policyCheckpoint) config = policyCheckpointConfig(config);
    this.setDefaultConfig(config);
    require('./policyOcrConfig').validateOcrEndpoints(this.defaultConfig);
    if (isTrivialPolicy(this.defaultConfig) && this.policyCheckpoint) {
      throw new Error("Trivial Policy starts from screenshots; Brain checkpoints are unsupported.");
    }
    this.running = true;
    this.stopRequested = false;
    this.paused = false;
    this.resolvePauseWaiters();
    this.gamePaused = false;
    this.phase = "ACTIVE_CAPTURE";
    this.pendingPauseReason = null;
    this.observationId = 0;
    this.actionCount = 0;
    this.responseCount = 0;
    this.invalidPackageRetries = 0;
    this.lastPackageParseError = "";
    this.currentModel = "";
    this.lastObservation = null;
    this.recoveryCallCount=0;this.lastRecoveryAt=0;this.recoveryToolRunning=false;
    this.recoveryAwaitingBrain=false;this.recoveryPromise=null;
    this.lastError = null;
    this.lastStopReason = null;
    this.taskState = PAUSED_TASK_STATES.NORMAL_GAMEPLAY;
    this.recoveryMode = false;
    this.recoveryEpoch = 0;
    this.runDurationLimitReached = false;
    this.startedAt = new Date().toISOString();
    this.stoppedAt = null;
    this.sessionId = createSessionId();
    this.sessionHistory = null;
    this.historyWriteFailed = false;
    this.systemPrompt = "";
    this.taskInstruction = String(this.policyCheckpoint?.taskInstruction || options.taskInstruction || "").trim();
    this.messages = [];
    this.policyContextSignature = null;
    this.compaction = null;
    this.pendingSharedKnowledge = [];
    this.lastSessionState = null;
    this.frameworkRuntime = null;
    this.currentSession = null;
    this.taskResultMonitor.reset();
    const outputRoots = sessionOutputRoots(
      CAPTURE_ROOT,
      SESSION_HISTORY_ROOT,
      this.defaultConfig.sessionOutputSubpath
    );
    this.sessionCaptureRoot = outputRoots.captureRoot;
    this.sessionHistoryRoot = outputRoots.sessionHistoryRoot;

    try {
      this.sharedDocuments = isTrivialPolicy(this.defaultConfig)
        ? new SharedDocuments({ policySystemPrompt: await loadTrivialPolicyPrompt() })
        : options.sharedDocuments || await SharedDocuments.load({
          policySystemPromptProvider: this.systemPromptProvider
        });
      const baseSystemPrompt = this.sharedDocuments.get("policySystemPrompt");
      if (!baseSystemPrompt) {
        throw new Error("Policy system prompt cannot be empty.");
      }
      this.systemPrompt = String(baseSystemPrompt).trim();
      this.messages = [{ role: "system", content: this.systemPrompt }];
      const continuation = this.defaultConfig.continuationHandoffPath
        ? await readContinuationHandoff(this.defaultConfig.continuationHandoffPath)
        : null;
      this.currentSession = new SessionState({
        sessionId: this.sessionId,
        agentMode: this.agentMode,
        startedAt: this.startedAt,
        userGoal: this.taskInstruction,
        continuation
      });
      this.sessionHistory = await createSessionHistory({
        sessionId: this.sessionId,
        agentMode: this.agentMode,
        startedAt: this.startedAt,
        captureRoot: this.sessionCaptureRoot,
        sessionHistoryRoot: this.sessionHistoryRoot,
        sessionOutputSubpath: outputRoots.sessionOutputSubpath,
        config: publicConfig(this.defaultConfig),
        systemPrompt: this.systemPrompt,
        taskInstruction: this.taskInstruction,
        messages: this.messages,
        actionCount: 0,
        responseCount: 0,
        maxSteps: this.defaultConfig.maxSteps,
        maxActions: this.defaultConfig.maxSteps,
        maxStepsUnlimited: this.defaultConfig.maxStepsUnlimited,
        maxActionsUnlimited: this.defaultConfig.maxStepsUnlimited,
        sharedDocuments: this.sharedDocuments.snapshot(),
        framework: this.currentSession.snapshot(),
        memory: this.currentSession.brain.memory,
        events: [{
          type: "session_started",
          at: this.startedAt,
          agentMode: this.agentMode,
          continuationLoaded: Boolean(continuation),
          continuationPreviousSessionId: continuation?.previousSessionId || null
        }]
      }, { appendOnly: true, deltaPatches: true, asyncWrites: true,
        retainHistoryInMemory: false,
        onError: (error) => this.handleHistoryWriteError(error)
      });
      // Full Brain evidence stays in memory. Disk persistence belongs to the
      // history worker; synchronous timeline archive/trim is not a control path.
      this.currentSession.timeline.historyRecordCursor = this.currentSession.timeline.records.length;
      const brainActor = new BrainActor({
        recordEvent: (type, payload) => {
          if (type === "brain_model_retry") {
            this.emit("log", `Brain ${payload.model || "model resolution"} retry ${payload.attempt}/${payload.maxRetries} in ${payload.delayMs}ms: ${payload.error}`);
          }
          return this.recordHistoryEvent(type, payload);
        },
        config: this.defaultConfig,
        sharedDocuments: this.sharedDocuments,
        sessionState: this.currentSession,
        recoverMobility: (request, signal) => require('./traditionalControl').startRecovery(this, request, signal),
        ...(this.brainChatCompletion ? { chatCompletion: this.brainChatCompletion } : {}),
        ...(this.brainResolveModel ? { resolveModel: this.brainResolveModel } : {})
      });
      const policyActor = new PolicyActor({
        sessionState: this.currentSession,
        getConfig: () => ({ ...(this.manualPolicy?.activeConfig() || this.defaultConfig), agentMode: this.agentMode }),
        getMessages: () => this.messages,
        onSessionLost: () => this.manualPolicy?.sessionBacked ? this.manualPolicy.recoverSession()
          : this.rebuildPolicyContext({ reason: "session_lost" }),
        onDelivered: (messages) => this.policyOcr.acknowledge(messages.map(message => message.ocrToken).filter(Boolean)),
        appendMessage: (message) => this.messages.push(message),
        buildRequestMessages: (messages) => this.buildRequestMessages(messages),
        createCandidate: (candidate) => this.currentSession.addPolicyCandidate(candidate),
        recordCandidate: (candidate) => this.recordPolicyCandidate(candidate),
        recordModelResponse: (record) => this.recordModelResponse(record),
        recordEvent: (type, payload) => this.recordHistoryEvent(type, payload),
        buildRetryText: (actor) => {
          this.invalidPackageRetries = actor.invalidPackageRetries;
          this.lastPackageParseError = actor.lastParseError;
          return this.buildInvalidPackageRetryText();
        },
        requestBrainReplan: (reason) => this.frameworkRuntime?.requestBrainReplan(reason) || false,
        emit: (type, payload) => this.emit(type, payload),
        shouldStop: () => this.stopRequested,
        model: this.currentModel,
        chatCompletion: this.chatCompletion,
        maxRetries: this.modelRequestMaxRetries,
        retryBaseDelayMs: this.modelRequestRetryBaseDelayMs
      });
      this.frameworkRuntime = new FrameworkRuntime({
        config: this.defaultConfig,
        sessionState: this.currentSession,
        history: this.sessionHistory,
        brainActor,
        policyActor,
        inputController: this.inputController,
        host: {
          onHistoryError: (error) => this.handleHistoryWriteError(error),
          onBrainRevision: (revision) => {
            this.collectSharedKnowledge(revision);
            this.emit("plan", {
              frameworkBrain: true,
              revision: revision.revision,
              content: revision.result?.plan || "",
              instruction: revision.result?.instruction || "",
              plan: revision.result?.plan || ""
            });
            this.emitStatus();
          },
          onBrainRecoveryCompleted: () => { this.recoveryAwaitingBrain = false; },
          onBrainEvent: (event) => {
            this.emit("brain", {
              ...event,
              taskState: this.taskState,
              recoveryMode: this.recoveryMode
            });
          }
        }
      });
    } catch (error) {
      this.frameworkRuntime?.stop("start failed");
      if (this.sessionHistory) {
        try {
          await this.sessionHistory.appendEvent("session_start_error", { message: error.message },
            { status: "error", stoppedAt: new Date().toISOString() });
        } catch (historyError) { error.historyRecordError = historyError.message; }
        try { await this.sessionHistory.flush(); }
        catch (historyError) { error.historyFlushError = historyError.message; }
      }
      this.frameworkRuntime = null;
      this.currentSession = null;
      this.running = false;
      this.lastError = error.message;
      throw error;
    }

    this.emit("log", "high-frequency agent starting");
    this.emitStatus();
    this.cancelRunDeadline?.();
    this.cancelRunDeadline = require("./runDeadline").startRunDeadline(this, this.defaultConfig.runDurationMs);
    this.runPromise = this.run().finally(async () => {
      this.cancelRunDeadline?.();
      this.cancelRunDeadline = null;
      const wasStopRequested = this.stopRequested;
      this.policyOcr.reset();
      this.frameworkRuntime?.stop("agent finalizing");
      if(this.recoveryPromise)await this.recoveryPromise.catch(error=>{
        if(!this.lastError&&error.name!=='AbortError')this.lastError=error.message;
      });
      await this.resetController().catch((error) => this.emit("log", `gamepad reset failed: ${error.message}`));
      await this.resumeGame("agent_stopped").catch((error) => this.emit("log", `failed to resume game: ${error.message}`));
      await this.taskResultMonitor.waitForIdle();
      this.running = false;
      this.stopRequested = false;
      this.paused = false;
      this.resolvePauseWaiters();
      this.abortController = null;
      this.stoppedAt = new Date().toISOString();
      this.phase = "IDLE";
      this.lastStopReason = this.lastError
        ? "error"
        : this.runDurationLimitReached
          ? "run_duration_limit"
          : reachedMaxSteps(this.defaultConfig, this.actionCount)
            ? "max_steps"
            : wasStopRequested
              ? "stop_requested"
              : "completed";
      await this.finishSessionHistory(this.lastStopReason);
      this.lastSessionState = this.currentSession?.snapshot({ includeTimeline: false }) || null;
      this.currentSession = null;
      this.frameworkRuntime = null;
      this.emit("log", "high-frequency agent stopped");
      this.emitStatus();
    });

    return this.getStatus();
  }

  stop(reason = "manual") {
    this.cancelRunDeadline?.();
    this.policyOcr.reset();
    if (!this.running) {
      return this.getStatus();
    }
    this.stopRequested = true;
    this.policyOcr.reset();
    this.paused = false;
    this.resolvePauseWaiters();
    if (this.abortController) {
      this.abortController.abort(new Error(`stop requested: ${reason}`));
    }
    this.frameworkRuntime?.stop(reason);
    this.emit("log", `high-frequency stop requested: ${reason}`);
    void this.recordHistoryEvent("stop_requested", { reason }).catch(() => {});
    this.emitStatus();
    return this.getStatus();
  }

  resume(reason = "manual") {
    if (!this.running || !this.paused) {
      return this.getStatus();
    }
    this.paused = false;
    this.frameworkRuntime?.actionRuntime.resumePendingTimeout();
    this.resolvePauseWaiters();
    this.emit("log", `high-frequency agent resume requested: ${reason}`);
    void this.recordHistoryEvent("resume_requested", { reason }).catch(() => {});
    this.emitStatus();
    return this.getStatus();
  }

  togglePause(reason = "manual") {
    return this.paused ? this.resume(reason) : this.pause(reason);
  }

  async toggle(config = {}) {
    if (!this.running) {
      return this.start(config);
    }
    return this.togglePause("toggle");
  }

  resolvePauseWaiters() {
    for (const resolve of this.pauseWaiters.splice(0)) {
      resolve();
    }
  }

  async waitWhilePaused() {
    while (this.paused && !this.stopRequested) {
      await new Promise((resolve) => this.pauseWaiters.push(resolve));
    }
  }

  async runBrainBarrier(trigger) {
    if (this.stopRequested) return;
    const runtime = this.frameworkRuntime;
    if (trigger.initial) {
      const checkpoint = this.policyCheckpoint;
      const startupState = initialBrainStateFor(this.defaultConfig);
      runtime.brainRuntime.seedInitialRevision({
        ...(checkpoint?.brain || startupState),
        synthetic: true,
        model: checkpoint ? "synthetic-checkpoint" : "synthetic-startup"
      });
      if (checkpoint) {
        await this.recordHistoryEvent("policy_checkpoint_seeded", { checkpoint }, { policyCheckpoint: checkpoint });
      } else {
        await this.recordHistoryEvent("policy_startup_seeded", { state: startupState });
      }
      await this.initialObservationReady?.(this.lastObservation);
      if (this.stopRequested) return;
      await this.rebuildPolicyContext(trigger);
      return;
    }
    const type = require("./brainProtocol").updateType(trigger);
    if (type === "small" && trigger.thinking && this.currentSession.brain.think!==trigger.thinking) {
      this.currentSession.brain.think=trigger.thinking;
      // This is the ordinary Policy-think update, before Brain chooses state
      // or a tool. A completed tool preserves the dialogue after this update.
      await this.rebuildPolicyContext({reason:'policy_think_update'});
    }
    const recoveryBefore = Number(this.currentSession.brain.lastRecoveryActionSeq || 0);
    const pauseBrain = !this.recoveryMode
      && this.taskState === PAUSED_TASK_STATES.NORMAL_GAMEPLAY;
    if (pauseBrain) await this.pauseGame(`brain:${trigger.reason}`);
    try {
      if (this.stopRequested) return;
      runtime.requestBrainReplan(trigger.reason, {
        thinking: trigger.thinking,
        policyEvidence: trigger.policyEvidence || null,
        policyProposal: trigger.policyProposal || null,
        initial: trigger.initial === true,
        updateType: type
      });
      // BrainActor owns bounded request deadlines and retries; wait for its terminal result.
      await runtime.waitForBrainBarrier(0);
      if (this.stopRequested) return;
      const recoveryCompleted = Number(this.currentSession.brain.lastRecoveryActionSeq || 0) > recoveryBefore;
      if (!recoveryCompleted) await this.rebuildPolicyContext(trigger);
      else await this.recordHistoryEvent('policy_context_preserved_after_recovery', {
        contextId: runtime.policyActor.transport.sessionId, historyMessages: this.messages.length });
    } finally {
      if (pauseBrain) await this.resumeGame(`brain:${trigger.reason}`);
    }
  }

  policyStateSignature() {
    if (isTrivialPolicy(this.defaultConfig)) return JSON.stringify([this.taskInstruction, this.taskState, this.recoveryMode]);
    return JSON.stringify([
      this.currentSession?.brain?.revision || 0,
      this.currentSession?.brain?.think || "",
      this.currentSession?.currentPolicyInstruction?.text || this.taskInstruction,
      this.taskState,
      this.recoveryMode
    ]);
  }

  buildPolicySystemMessage(trigger = {}) {
    return isTrivialPolicy(this.defaultConfig)
      ? buildTrivialSystemMessage(this) : buildPolicySystemMessage(this, trigger);
  }

  async rebuildPolicyContext(trigger = {}) {
    this.frameworkRuntime?.policyActor?.resetContext?.(trigger.reason);
    this.policyOcr.beginSegment({ preserveScreen: isTrivialPolicy(this.defaultConfig) });
    this.messages = [this.buildPolicySystemMessage(trigger), { role: "user", content: "<screen_ocr></screen_ocr>", ocrAggregate: true, ocrRecords: [] }];
    this.policyContextSignature = this.policyStateSignature();
    this.compaction = null;
    await this.recordHistoryEvent('policy_context_rebuilt', { reason: trigger.reason || 'state_changed',
      contextId: this.frameworkRuntime?.policyActor?.transport?.sessionId,
      contextEpoch: this.frameworkRuntime?.policyActor?.contextEpoch, aggregateEmpty: true });
    this.emit("log", `high-frequency Policy context rebuilt (${trigger.reason || "state_changed"})`);
    this.emitStatus();
  }

  async resolveModelWithAbort() {
    this.abortController = new AbortController();
    try {
      return await this.resolveModel(this.defaultConfig, { signal: this.abortController.signal });
    } finally {
      this.abortController = null;
    }
  }

  buildInvalidPackageRetryText() {
    const chunks = this.defaultConfig.hfChunksPerPackage;
    return [
      `上一次输出无效。请重试 ${this.invalidPackageRetries}/${this.defaultConfig.policyMaxInvalidRetries}。`,
      `解析器错误：${this.lastPackageParseError || "未知动作包格式错误"}`,
      `修正紧凑格式：返回一个小写 <a>...</a> 动作块或相同的裸动作包；动作前输出 <think>...</think>；无新判断时留空，普通非空判断请求 memory 小更新，阶段更新在 think 内加入 <|call_brain|>。动作必须以四个带符号的一位摇杆坐标开头，严格包含 ${chunks} 个 action chunk，并用 ${Math.max(0, chunks - 1)} 个分号分隔。`,
      "坐标前缀固定为八个字符，顺序是左摇杆 x、左摇杆 y、右摇杆 x、右摇杆 y，例如 -9+0+2+9。居中必须使用 +0，不能使用 -0。",
      "坐标前缀之后只能拼接允许的按钮或扳机 token。不要使用空格、换行、L(...)、R(...)、逗号、数字 chunk 标签或 neutral。",
      "没有按钮或扳机输入的 chunk，直接在分号之间留空。",
      "使用对话中已有的最新 observation。只返回修正后的动作块。"
    ].join("\n");
  }

  async buildRequestMessages(sourceMessages = this.messages) {
    if (this.manualPolicy?.sessionBacked) return require('./policyHandoffContext').materialize(sourceMessages);
    if (this.defaultConfig.policyTransport === "lumine_session") {
      sourceMessages = [sourceMessages[0], sourceMessages.at(-1)];
    }
    const messages = [];
    for (const message of sourceMessages) {
      if (!Array.isArray(message.content)) {
        messages.push({ role: message.role, content: message.content });
        continue;
      }
      const content = [];
      for (const part of message.content) {
        content.push(part.type === "local_image"
          ? {
              type: "image_url",
              image_url: { url: await imagePathToDataUrl(part.imagePath, part.mimeType) }
            }
          : part);
      }
      messages.push({ role: message.role, content });
    }
    return messages;
  }

  async appendObservation(text, observation) {
    if (this.policyContextSignature !== this.policyStateSignature() || this.messages.length === 0) {
      this.frameworkRuntime?.policyActor?.resetContext?.("state_changed");
      this.policyOcr.beginSegment({ preserveScreen: isTrivialPolicy(this.defaultConfig) });
      this.messages = [this.buildPolicySystemMessage(), { role: "user", content: "<screen_ocr></screen_ocr>", ocrAggregate: true, ocrRecords: [] }];
      this.policyContextSignature = this.policyStateSignature();
    }
    const observationId = observation.frameworkObservationId || observation.observationId || null;
    const content = [{
      type: "local_image",
      imageKind: "full_frame",
      observationId,
      imagePath: observation.imagePath || "",
      mimeType: observation.mimeType || "image/png",
      width: observation.width,
      height: observation.height,
      capturedAt: observation.capturedAt
    }];
    if (this.defaultConfig.policyIncludeMinimap && (this.defaultConfig.policyTransport !== 'lumine_session' || this.manualPolicy?.sessionBacked)) {
      const minimap = this.manualPolicy?.sessionBacked
        ? await require('./policyHandoffContext').minimap(observation, this.manualPolicy.settings.imagePython)
        : await require('./policyMinimap').policyMinimap(observation);
      content.push({ type: 'local_image', imageKind: 'minimap', observationId, ...minimap });
    }
    const ocr = this.policyOcr.reserve(observation);
    const ocrRecords = isTrivialPolicy(this.defaultConfig)
      ? compactTrivialOcrHistory(this.messages, ocr, this.defaultConfig.trivialOcrHistoryResetMs) : ocr.records;
    const ocrText = isTrivialPolicy(this.defaultConfig) && this.defaultConfig.policyTransport !== "lumine_session"
      ? trivialOcrSnapshot(ocr.snapshot) : ocr.text;
    content.unshift({ type: "text", text: ocrText + "\n" });
    this.messages.push({ role: "user", content, ocrToken: ocr.token, ocrRecords });
    const removed = await this.pruneContextImages();
    this.emitStatus();
    return removed;
  }

  contextObservationLimit() {
    return this.defaultConfig.policyWindowMaxRounds || 16;
  }

  countContextImages(kind = "full_frame") {
    return this.messages.reduce((count, message) => count
      + (Array.isArray(message.content) ? message.content.filter((part) =>
        part.type === "local_image" && (part.imageKind || "full_frame") === kind).length : 0), 0);
  }

  async pruneContextImages() {
    if (!this.manualPolicy?.sessionBacked && !isTrivialPolicy(this.defaultConfig) && this.defaultConfig.policyTransport === "lumine_session") return [];
    const isObservation = (message) => message.role === "user" && Array.isArray(message.content)
      && message.content.some((part) => part.type === "local_image" && part.imageKind !== "minimap");
    const removed = [];
    const trim = this.countContextImages() > this.contextObservationLimit();
    const keep = (this.defaultConfig.policyWindowKeepRounds || 8) + 1;
    while (trim && this.countContextImages() > keep) {
      const start = this.messages.findIndex(isObservation);
      const next = this.messages.findIndex((message, index) => index > start && isObservation(message));
      if (start < 0 || next < 0) throw new Error("Invalid Policy observation window.");
      const frame = this.messages[start].content.find((part) => part.type === "local_image" && part.imageKind !== "minimap");
      const minimap = this.messages[start].content.find((part) => part.imageKind === "minimap");
      // Remove the observation and all its replies/retry messages, never the next observation.
      removed.push({
        imagePath: frame.imagePath,
        observationId: frame.observationId,
        minimapImagePath: minimap?.imagePath || null,
        reason: "rolling_window",
        removedMessages: next - start
      });
      const aggregate = this.messages.find(message => message.ocrAggregate);
      {
        if (!aggregate) throw new Error("Policy requires a fixed first OCR aggregate user.");
        const evicted = this.messages.slice(start, next);
        aggregate.ocrRecords.push(...evicted.flatMap(message => message.ocrRecords || []));
        aggregate.content = (isTrivialPolicy(this.defaultConfig) ? "" : "此前已移除图片和动作的 OCR 历史（按出现顺序）：\n") + require("./asyncPolicyOcr").ocrBlock(aggregate.ocrRecords.map(record => record.text).filter(Boolean).join("\n"));
      }
      this.messages.splice(start, next - start);
    }
    if (removed.length) {
      await this.recordHistoryEvent('policy_window_trimmed', {
        contextId: this.frameworkRuntime?.policyActor?.transport?.sessionId,
        keptRounds: this.countContextImages(), removedObservations: removed.map(record => record.observationId),
        aggregateOcrIds: this.messages.find(message => message.ocrAggregate)?.ocrRecords.map(record => record.id) || []
      });
      this.compaction = {
        active: false, mode: "drop_observation_turns",
        total: removed.length, completed: removed.length,
        images: removed.map((record) => ({ ...record, status: "removed" }))
      };
      this.emit("compact", cloneJson(this.compaction));
      this.emit("log", `high-frequency context window: removed ${removed.length} observation turns; no summary requests`);
    }
    return removed;
  }

  async capture(text, { deferBrainObservation = false } = {}) {
    this.phase = "ACTIVE_CAPTURE";
    this.frameworkRuntime?.pauseHarness.transition(this.phase, this.gamePaused === true, { reason: "capture" });
    let observation = await this.captureObserve({
      displayId: this.defaultConfig.displayId,
      resolution: this.defaultConfig.resolution,
      captureFormat: "jpeg",
      captureQuality: 90,
      inputController: this.inputController,
      sessionId: this.sessionId || "manual",
      label: `high-frequency-observe-${String(this.actionCount).padStart(4, "0")}`,
      captureRoot: this.sessionCaptureRoot
    });
    if (this.pendingPauseReason && !this.stopRequested) {
      const reason = this.pendingPauseReason;
      this.pendingPauseReason = null;
      await this.pauseGame(reason);
    }
    observation = await attachMinimapCrop(observation, this.defaultConfig, {
      sessionId: this.sessionId || "manual",
      label: `high-frequency-minimap-${String(this.actionCount).padStart(4, "0")}`,
      captureRoot: this.sessionCaptureRoot
    });

    const frameworkObservation = this.frameworkRuntime?.observation(observationForHistory(observation), { deferBrainObservation });
    this.observationId = frameworkObservation?.observationSeq || (this.observationId + 1);
    observation.observationId = this.observationId;
    observation.frameworkObservationId = frameworkObservation?.observationId || `O${this.observationId}`;
    this.lastObservation = {
      observationId: observation.observationId,
      frameworkObservationId: observation.frameworkObservationId,
      text,
      image: observation.imageUrl,
      imageUrl: observation.imageUrl,
      imagePath: observation.imagePath,
      mimeType: observation.mimeType,
      width: observation.width,
      height: observation.height,
      display: observation.display,
      cursor: observation.cursor,
      cursorVisible: observation.cursorVisible,
      contentBounds: observation.contentBounds,
      letterboxed: observation.letterboxed,
      navigation: observationForHistory(observation).navigation || null,
      ocr: observation.ocr,
      capturedAt: observation.capturedAt
    };
    if (this.defaultConfig.ocrEnabled) this.policyOcr.schedule(observation, (frame) => this.recognizeScreenshotText({
      ...this.defaultConfig, ocrRois: require('./policyOcrConfig').isTrainingPolicyOcr(this.defaultConfig)
        ? ['full_screen'] : ["tutorial_prompt", "mission_objective"]
    }, frame), async (result) => {
      observation.ocr = result;
      if (frameworkObservation) {
        frameworkObservation.ocr = result;
        this.currentSession?.timeline.touch(frameworkObservation);
      }
      if (this.lastObservation?.observationId === observation.observationId) this.lastObservation.ocr = result;
      await this.recordHistoryEvent("asynchronous_ocr_completed", {
        observationId: observation.frameworkObservationId, capturedAt: observation.capturedAt, result
      });
    });
    this.taskResultMonitor.observe(observation, {
      sessionId: this.sessionId,
      actionCount: this.actionCount,
      responseCount: this.responseCount
    });
    this.emit("observe", this.lastObservation);
    this.emitStatus();
    return observation;
  }

  async tapKey(key) {
    await this.inputController.keyDown(key);
    try {
      await sleep(PAUSE_KEY_HOLD_MS);
      await this.inputController.keyUp(key);
    } catch (error) {
      await this.inputController.keyUp(key).catch(() => {});
      throw error;
    }
  }

  async tapPauseKey() {
    return this.tapKey(PAUSE_KEY);
  }

  async tapResumeKey() {
    return this.tapKey(RESUME_KEY);
  }

  supportsGamepadPauseResume() {
    const capabilities = typeof this.inputController?.capabilities === "function"
      ? this.inputController.capabilities()
      : null;
    return capabilities?.gamepad === true
      && typeof this.inputController.gamepadButtonDown === "function"
      && typeof this.inputController.gamepadButtonUp === "function";
  }

  supportsNativePauseMenuObservation() {
    return this.inputController?.capabilities?.()?.nativePauseMenuObservation === true;
  }

  async tapGamepadButton(button) {
    await this.inputController.gamepadButtonDown(button);
    try {
      await sleep(PAUSE_KEY_HOLD_MS);
      await this.inputController.gamepadButtonUp(button);
    } catch (error) {
      await this.inputController.gamepadButtonUp(button).catch(() => {});
      throw error;
    }
  }

  async pauseGame(reason) {
    if (this.gamePaused || this.recoveryMode || this.taskState !== PAUSED_TASK_STATES.NORMAL_GAMEPLAY) {
      return false;
    }
    await this.frameworkRuntime?.actionRuntime.suspendCarry("pause_harness");
    this.phase = "PAUSE_REQUESTED";
    this.frameworkRuntime?.pauseHarness.transition(this.phase, false, { reason });
    const input = this.supportsGamepadPauseResume() ? "gamepad" : "keyboard";
    const buttonOrKey = input === "gamepad" ? PAUSE_GAMEPAD_BUTTON : PAUSE_KEY;
    if (this.supportsNativePauseMenuObservation()) {
      await this.enterVerifiedNativePause(reason, input);
    } else {
      await this.enterVerifiedLocalPause(reason, input);
    }
    this.gamePaused = true;
    this.phase = "PAUSED_INFERENCE";
    this.frameworkRuntime?.pauseHarness.transition(this.phase, true, { reason });
    await this.recordHistoryEvent("game_paused_for_model_wait", {
      reason,
      input,
      ...(input === "gamepad" ? { button: buttonOrKey } : { key: buttonOrKey })
    });
    this.emit("log", `game paused while waiting for high-frequency ${reason}`);
    this.emitStatus();
    return true;
  }

  async enterVerifiedLocalPause(reason, input) {
    if (!this.defaultConfig.ocrEnabled) throw new Error('Local pause confirmation requires OCR.');
    const stopped = () => { if (this.stopRequested) throw new Error('Stopped while confirming local pause.'); };
    try {
      let check = await this.verifyGameResumed(reason, 0, { beforeResume: true });
      stopped();
      if (check.conclusive && check.menuOpen) return;
      if (!check.conclusive) throw new Error('Pause state unavailable before START; Brain inference not started.');
      for (let attempt = 1; attempt <= RESUME_VERIFICATION_MAX_ATTEMPTS; attempt++) {
        stopped();
        if (input === 'gamepad') await this.tapGamepadButton(PAUSE_GAMEPAD_BUTTON);
        else await this.tapPauseKey();
        await sleep(this.resumeDelayMs());
        stopped();
        check = await this.verifyPauseBeforeResume(reason, attempt);
        stopped();
        if (check.conclusive && check.menuOpen) return;
        if (!check.conclusive) throw new Error('Pause state unavailable after START; Brain inference not started.');
      }
      throw new Error('GTA pause menu did not open after 3 START attempts; Brain inference not started.');
    } catch (error) {
      await this.resetController('pause_verification_failed').catch(() => {});
      throw error;
    }
  }

  // START may be ignored during a character transition. Do not freeze the
  // framework clock or start a blocking inference until the Host confirms it.
  async enterVerifiedNativePause(reason, input) {
    const read = async (attempt, sample) => {
      const observation = await this.captureObserve({
        displayId: this.defaultConfig.displayId, resolution: this.defaultConfig.resolution,
        captureFormat: "jpeg", captureQuality: 90, inputController: this.inputController,
        sessionId: this.sessionId || "manual", captureRoot: this.sessionCaptureRoot,
        label: `pause-entry-${this.actionCount}-${attempt}-${sample}`
      });
      const check = nativePauseMenuVerification(observation, true);
      await this.recordHistoryEvent("game_pause_verification", { reason, attempt, sample,
        menuOpen: check.menuOpen, conclusive: check.conclusive,
        nativePauseMenu: observation.nativePauseMenu, observationHostEpoch: observation.hostEpoch,
        imagePath: observation.imagePath || null });
      return check;
    };
    const stopped = () => {
      if (this.stopRequested) throw new Error("Stopped while confirming native pause.");
    };
    try {
      let check = await read(0, 0);
      stopped();
      if (check.conclusive && check.menuOpen) return;
      if (!check.conclusive) throw new Error("Native pause state unavailable before START; inference not started.");
      for (let attempt = 1; attempt <= RESUME_VERIFICATION_MAX_ATTEMPTS; attempt += 1) {
        stopped();
        if (input === "gamepad") await this.tapGamepadButton(PAUSE_GAMEPAD_BUTTON);
        else await this.tapPauseKey();
        for (let sample = 1; sample <= RESUME_VERIFICATION_MAX_ATTEMPTS; sample += 1) {
          await sleep(this.resumeDelayMs());
          stopped();
          check = await read(attempt, sample);
          stopped();
          if (check.conclusive && check.menuOpen) return;
        }
        if (!check.conclusive) throw new Error("Native pause state unavailable after START; inference not started.");
      }
      throw new Error("GTA pause menu did not open after 3 START attempts; inference not started.");
    } catch (error) {
      await this.resetController("pause_verification_failed").catch(() => {});
      throw error;
    }
  }

  resumeDelayMs() {
    const value = Number(this.defaultConfig.modelWaitResumeDelayMs);
    return Number.isFinite(value) ? Math.max(0, Math.min(5000, Math.trunc(value))) : DEFAULT_MODEL_WAIT_RESUME_DELAY_MS;
  }

  carryEnabled() {
    return this.defaultConfig.hfCarryEnabled === true;
  }

  async resumeGame(reason) {
    if (!this.gamePaused) {
      return false;
    }
    if (this.recoveryMode || this.taskState !== PAUSED_TASK_STATES.NORMAL_GAMEPLAY) {
      return this.rejectPauseAssumption({}, reason);
    }
    this.phase = "RESUME";
    this.frameworkRuntime?.pauseHarness.transition(this.phase, true, { reason });
    const input = this.supportsGamepadPauseResume() ? "gamepad" : "keyboard";
    const buttonOrKey = input === "gamepad" ? RESUME_GAMEPAD_BUTTON : RESUME_KEY;
    const delayMs = this.resumeDelayMs();
    const controlWindowMs = Math.min(
      delayMs,
      Math.max(0, Math.trunc(Number(this.defaultConfig.resumeVerificationTailMs) || 0))
    );
    const verifyResume = input === "gamepad"
      && (this.defaultConfig.ocrEnabled || this.supportsNativePauseMenuObservation());
    let verification = null;
    let attempts = 0;

    try {
      while (attempts < (verifyResume ? RESUME_VERIFICATION_MAX_ATTEMPTS : 1)) {
        attempts += 1;
        const before = await this.verifyPauseBeforeResume(reason, attempts);
        if (!before.conclusive) {
          throw new Error("Pause menu could not be verified before resume; no B/Esc sent. Check OCR or resume manually.");
        }
        if (!before.menuOpen) return await this.rejectPauseAssumption(before, reason);
        if (input === "gamepad") {
          await this.tapGamepadButton(RESUME_GAMEPAD_BUTTON);
        } else {
          await this.tapResumeKey();
        }
        this.phase = "RESUME_SETTLE";
        this.frameworkRuntime?.pauseHarness.transition(this.phase, true, { reason, delayMs, attempt: attempts });
        await sleep(Math.max(0, delayMs - controlWindowMs));

        if (verifyResume) {
          this.phase = "RESUME_VERIFY";
          this.frameworkRuntime?.pauseHarness.transition(this.phase, true, { reason, attempt: attempts });
          verification = await this.verifyGameResumed(reason, attempts);
          if (!verification.conclusive) {
            throw new Error("Game resume could not be verified: inconclusive; no action will be executed; inputs remain released.");
          }
          if (verification.menuOpen) {
            await this.frameworkRuntime?.actionRuntime.suspendCarry("resume_menu_retry");
            if (attempts < RESUME_VERIFICATION_MAX_ATTEMPTS) {
              continue;
            }
            throw new Error(`GTA pause menu remained open after ${attempts} B resume attempts.`);
          }
        }

        if (controlWindowMs > 0) await sleep(controlWindowMs);
        if (this.stopRequested) await this.resetController("stop_during_resume");
        break;
      }
    } catch (error) {
      await this.resetController("resume_failed");
      throw error;
    }
    this.frameworkRuntime?.resumeSettle(delayMs);
    this.gamePaused = false;
    this.phase = "ACTION_ACTIVE";
    this.frameworkRuntime?.actionRuntime.resumePendingTimeout();
    this.frameworkRuntime?.pauseHarness.transition(this.phase, false, {
      reason,
      delayMs,
      attempts,
      verified: verification?.conclusive ? !verification.menuOpen : null
    });
    await this.recordHistoryEvent("game_resumed_after_model_wait", {
      reason,
      input,
      delayMs,
      controlWindowMs,
      attempts,
      verified: verification?.conclusive ? !verification.menuOpen : null,
      ...(input === "gamepad" ? { button: buttonOrKey } : { key: buttonOrKey })
    });
    this.emit("log", `game resumed for high-frequency ${reason}; waited ${delayMs} ms`);
    this.emitStatus();
    return true;
  }

  async rejectPauseAssumption(check, reason) {
    await this.resetController("pause_not_confirmed");
    this.gamePaused = false;
    this.pendingPauseReason = null;
    this.phase = "ACTIVE_INFERENCE";
    this.frameworkRuntime?.pauseHarness.transition(this.phase, false, { reason: "pause_not_confirmed" });
    if (check.taskResult?.ok) {
      await this.handleTaskResult(check.taskResult, {
        observation: check.observation,
        metadata: { sessionId: this.sessionId, source: "before_resume" }
      });
    }
    await this.recordHistoryEvent("game_resume_skipped", {
      reason, menuOpen: false, taskState: this.taskState,
      imagePath: check.observation?.imagePath || null
    });
    if (this.running && !this.stopRequested) {
      const text = "Pause was not confirmed; discard the pending action and observe the current screen.";
      const observation = await this.capture(text);
      const pruned = await this.appendObservation(text, observation);
      await this.recordObservation(text, observation, pruned);
      await this.taskResultMonitor.waitForIdle();
      await this.rebuildPolicyContext({ reason: "pause_not_confirmed" });
    }
    return false;
  }

  async verifyPauseBeforeResume(reason, attempt) {
    if (!this.defaultConfig.ocrEnabled && !this.supportsNativePauseMenuObservation()) {
      return { conclusive: false, menuOpen: false };
    }
    let check;
    for (let sample = 1; sample <= RESUME_VERIFICATION_MAX_ATTEMPTS; sample += 1) {
      check = await this.verifyGameResumed(reason, attempt, { beforeResume: true });
      if (check.menuOpen) return check;
      if (check.taskResult?.ok && isRecoveryTaskState(classifyPausedTaskState(check.taskResult))) return check;
      // START can return before the menu opens, including its native flags.
      // Recheck closed/unknown states within the existing settle/retry bound
      // before deciding the pause did not take; only positive evidence sends B.
      if (this.stopRequested) return check;
      if (sample < RESUME_VERIFICATION_MAX_ATTEMPTS) {
        const settleMs = this.resumeDelayMs();
        await this.recordHistoryEvent("pause_resume_preflight_recheck", { reason, attempt, sample, settleMs,
          menuOpen: check.menuOpen, conclusive: check.conclusive });
        await sleep(settleMs);
        if (this.stopRequested) return { ...check, conclusive: false };
      }
    }
    return check;
  }

  async verifyGameResumed(reason, attempt, { beforeResume = false } = {}) {
    const startedPerfMs = performance.now();
    const observation = await this.captureObserve({
      displayId: this.defaultConfig.displayId,
      resolution: this.defaultConfig.resolution,
      captureFormat: "jpeg",
      captureQuality: 90,
      inputController: this.inputController,
      sessionId: this.sessionId || "manual",
      label: `high-frequency-${beforeResume ? "pre-resume" : "resume"}-check-${String(this.actionCount).padStart(4, "0")}-${attempt}`,
      captureRoot: this.sessionCaptureRoot
    });
    const captureFinishedPerfMs = performance.now();
    const nativeState = nativePauseMenuVerification(observation, this.supportsNativePauseMenuObservation());
    if (nativeState) {
      await this.recordHistoryEvent(beforeResume ? 'game_resume_preflight' : 'game_resume_native_observation', {
        reason, attempt, ...nativeState, imagePath: observation.imagePath || null,
        nativePauseMenu: observation.nativePauseMenu, observationHostEpoch: observation.hostEpoch
      });
      return nativeState;
    }
    const ocr = await this.recognizeScreenshotText({
      ...this.defaultConfig,
      // Include tabs and the cutscene pause/back footer in both checks.
      // Verification consumes raw menu text, independently of Policy PBCK.
      policyOcrProfile: 'legacy',
      effectivePolicyOcrProfile: 'legacy',
      ocrRois: ["full_screen"]
    }, observation);
    const gpuFinishedPerfMs = performance.now();
    const gpuState = detectGtaPauseMenu(ocr, { requirePositiveHeader: beforeResume });
    const gpuText = (ocr?.results || []).map(result => String(result.text || '')).join(' ');
    const gpuTaskState = classifyPausedTaskState({ text: gpuText,
      ...(/(?:任务(?:完成|通过|成功)|mission\s+(?:passed|complete(?:d)?|success))/i.test(gpuText) ? { signal: 'success' } : {}) });
    const gpuConflict = ![PAUSED_TASK_STATES.NORMAL_GAMEPLAY, PAUSED_TASK_STATES.UNKNOWN].includes(gpuTaskState)
      && !(gpuTaskState === PAUSED_TASK_STATES.CUTSCENE_OR_TRANSITION && gpuState.pausedCutscene);
    const gpuPositive = beforeResume && gpuState.conclusive && gpuState.menuOpen
      && !gpuConflict && !this.recoveryMode && this.taskState === PAUSED_TASK_STATES.NORMAL_GAMEPLAY;
    let state = { ...gpuState };
    let taskResult = null;
    let cpuState = null;
    let usedCpuFallback = false;
    let cpuStartedPerfMs = null, cpuFinishedPerfMs = null;
    const needsCpu = this.defaultConfig.taskResultOcrEnabled
      && (beforeResume ? !gpuPositive : !gpuState.conclusive || gpuConflict);
    if (needsCpu) {
      usedCpuFallback = true;
      cpuStartedPerfMs = performance.now();
      taskResult = await this.taskResultMonitor.detectTaskResult({
        ...this.defaultConfig, taskResultOcrRois: ["full_screen"]
      }, observation);
      cpuFinishedPerfMs = performance.now();
      if (taskResult.ok) {
        cpuState = detectGtaPauseMenu({ ok: true, results: [{ text: taskResult.text || "" }] }, { requirePositiveHeader: beforeResume });
        state = { ...cpuState };
      }
      const taskState = taskResult.ok ? classifyPausedTaskState(taskResult) : PAUSED_TASK_STATES.UNKNOWN;
      if (taskState !== PAUSED_TASK_STATES.UNKNOWN && taskState !== PAUSED_TASK_STATES.NORMAL_GAMEPLAY
        && !(taskState === PAUSED_TASK_STATES.CUTSCENE_OR_TRANSITION && cpuState?.pausedCutscene)) {
        state.menuOpen = false;
        state.conclusive = true;
      } else if (!taskResult.ok) {
        state.conclusive = false;
      }
    } else if (gpuConflict) {
      // Positive menu words mixed with terminal/confirmation evidence are not
      // sufficient to press B or release a new action without clarification.
      state.conclusive = false;
    }
    const verificationFinishedPerfMs = performance.now();
    const evidence = (value, menu) => value ? { ok: Boolean(value.ok),
      requestLatencyMs: value.requestLatencyMs ?? null, serviceLatencyMs: value.latencyMs ?? null,
      error: String(value.error || ''), menu: menu ? { ...menu } : null,
      text: String(value.text || (value.results || []).map(r => String(r.text || '')).join(' ')).slice(0, 2000) } : null;
    const eventType = beforeResume ? "game_resume_preflight" : state.menuOpen
      ? "game_resume_not_acknowledged"
      : state.conclusive
        ? "game_resume_verified"
        : "game_resume_verification_inconclusive";
    await this.recordHistoryEvent(eventType, {
      reason, attempt, menuOpen: state.menuOpen, pausedCutscene: Boolean(state.pausedCutscene),
      matchedMarkers: state.matchedMarkers, hasMenuFooter: state.hasMenuFooter,
      ocrOk: Boolean(ocr?.ok), ocrError: String(ocr?.error || ""), conclusive: state.conclusive,
      ...(beforeResume ? { taskResultOk: taskResult?.ok ?? null, taskState: taskResult?.ok ? classifyPausedTaskState(taskResult) : null } : {}),
      verificationSource: usedCpuFallback ? 'cpu_full_screen_fallback' : gpuPositive ? 'gpu_positive_menu' : 'gpu_full_screen',
      usedCpuFallback, gpuConflict, gpuTaskState,
      verificationTiming: { startedPerfMs, captureFinishedPerfMs, gpuFinishedPerfMs,
        cpuStartedPerfMs, cpuFinishedPerfMs, verificationFinishedPerfMs, captureMs: captureFinishedPerfMs - startedPerfMs,
        gpuOcrRequestMs: gpuFinishedPerfMs - captureFinishedPerfMs,
        cpuFallbackRequestMs: usedCpuFallback ? cpuFinishedPerfMs - cpuStartedPerfMs : 0,
        totalMs: verificationFinishedPerfMs - startedPerfMs },
      verificationEvidence: { gpu: evidence(ocr, gpuState), cpu: evidence(taskResult, cpuState) },
      capturedAt: observation.capturedAt || null,
      imagePath: observation.imagePath || null
    });
    return { ...state, taskResult, observation: observationForHistory(observation) };
  }

  async resetController(reason = "reset") {
    if (this.frameworkRuntime?.actionRuntime) {
      await this.frameworkRuntime.actionRuntime.reset(reason);
      return;
    }
    if (typeof this.inputController.gamepadReset === "function") {
      await this.inputController.gamepadReset();
    }
  }

  async handleTaskResult(result, item = {}) {
    if (!this.running || item.metadata?.sessionId !== this.sessionId) {
      return;
    }
    await this.recordTaskResult(result, item).catch(() => {});
    if (result.ok === false || result.skipped) return;
    const nextState = classifyPausedTaskState(result);
    if (nextState === PAUSED_TASK_STATES.UNKNOWN && result.signal === "none") {
      if (this.recoveryMode) {
        const previousState = this.taskState;
        this.taskState = PAUSED_TASK_STATES.NORMAL_GAMEPLAY;
        this.recoveryMode = false;
        this.currentSession?.setTaskState(this.taskState, false);
        await this.recordHistoryEvent("paused_task_state_changed", {
          previousState,
          state: this.taskState,
          recoveryMode: false,
          recoveryEpoch: this.recoveryEpoch,
          signal: "none",
          reason: "recovery_screen_no_terminal_marker"
        });
        this.emit("log", "high-frequency recovery cleared; normal gameplay pause restored");
        this.emitStatus();
      }
      return;
    }
    if (nextState !== this.taskState) {
      const previousState = this.taskState;
      this.taskState = nextState;
      if (isRecoveryTaskState(nextState) && nextState !== PAUSED_TASK_STATES.NORMAL_GAMEPLAY) {
        this.recoveryMode = true;
        this.recoveryEpoch += 1;
        this.pendingPauseReason = null;
        this.gamePaused = false;
        this.currentSession?.invalidatePolicyInstruction(
          `Recovery state ${nextState}: follow the current HUD and session goal; do not assume confirm means retry.`
        );
        this.frameworkRuntime?.invalidateBrainContext(`task_state_${nextState.toLowerCase()}`);
        await this.resetController(`task_state_${nextState.toLowerCase()}`).catch(() => {});
        this.emit("log", `high-frequency entering recovery state: ${nextState}`);
      }
      this.currentSession?.setTaskState(this.taskState, this.recoveryMode);
      await this.recordHistoryEvent("paused_task_state_changed", {
        previousState,
        state: nextState,
        recoveryMode: this.recoveryMode,
        recoveryEpoch: this.recoveryEpoch,
        signal: result.signal || "none"
      });
      this.emitStatus();
    }
    if (shouldStopForTaskResult(this.defaultConfig, result) && nextState === PAUSED_TASK_STATES.MISSION_SUCCESS) {
      this.stop(`task_result_${result.signal || "terminal"}`);
    }
  }

  collectSharedKnowledge(revision = {}) {
    if (revision.synthetic || revision.result?.synthetic) return;
    const lessons = Array.isArray(revision.result?.memory?.persistent?.lessons)
      ? revision.result.memory.persistent.lessons.map((item) => String(item || "").trim()).filter(Boolean)
      : [];
    if (lessons.length === 0 || !this.currentSession) return;
    const tutorialObservations = this.currentSession.timeline.records.filter((record) => {
      if (record.type !== "observation") return false;
      return (record.ocr?.results || []).some((item) => (
        item.roi === "tutorial_prompt" && String(item.text || "").trim() && Number(item.score ?? 1) >= 0.8
      ));
    });
    const source = tutorialObservations.at(-1);
    if (!source) return;
    for (const text of lessons) {
      if (!/(教程|提示|使用|按|瞄准|锁定|切换|摇杆|扳机|手机|视角|目标)/i.test(text)) continue;
      if (this.pendingSharedKnowledge.some((entry) => entry.text === text)) continue;
      this.pendingSharedKnowledge.push({
        text,
        sourceObservationId: source.observationId,
        sourceSessionId: this.sessionId,
        confidence: null,
        source: "brain_lesson_with_tutorial_context"
      });
    }
  }

  historySnapshot(extra = {}) {
    return {
      agentMode: this.agentMode,
      status: this.running ? "running" : this.lastError ? "error" : "stopped",
      config: publicConfig(this.defaultConfig),
      modelName: this.currentModel || this.defaultConfig.modelName,
      actionCount: this.actionCount,
      responseCount: this.responseCount,
      stepCount: this.actionCount,
      maxSteps: this.defaultConfig.maxSteps,
      maxActions: this.defaultConfig.maxSteps,
      maxStepsUnlimited: this.defaultConfig.maxStepsUnlimited,
      maxActionsUnlimited: this.defaultConfig.maxStepsUnlimited,
      contextMessages: this.messages.length,
      contextImages: this.countContextImages(),
      contextMinimapImages: this.countContextImages("minimap"),
      compaction: cloneJson(this.compaction),
      contextImagePolicy: {
        fullFrameLimit: this.contextObservationLimit(),
        minimapPolicy: "paired_with_full_frame",
        eviction: "oldest_observation_and_replies",
        prunedImageSummary: false
      },
      taskResult: this.taskResultMonitor.getStatus(),
      messages: this.messages,
      lastObservation: this.lastObservation,
      lastError: this.lastError,
      lastStopReason: this.lastStopReason,
      startedAt: this.startedAt,
      stoppedAt: this.stoppedAt,
      gamePaused: this.gamePaused,
      ...(this.currentSession?.historySnapshot() || { framework: this.lastSessionState || null }),
      ...extra
    };
  }

  handleHistoryWriteError(error) {
    if (this.historyWriteFailed) return;
    this.historyWriteFailed = true;
    this.lastError = error.message;
    this.emit("errorEvent", { message: error.message, source: "session_history" });
    this.stop("history_write_failed");
  }

  recordHistoryEvent(type, payload = {}, extra = {}) {
    return this.sessionHistory
      ? this.sessionHistory.appendEvent(type, payload, this.historySnapshot(extra))
      : Promise.resolve();
  }

  recordObservation(text, observation, prunedImages = []) {
    const record = {
      ...observationForHistory(observation),
      at: observation.capturedAt || new Date().toISOString(),
      actionCount: this.actionCount,
      responseCount: this.responseCount,
      text,
      contextPrunedImages: prunedImages
    };
    return this.sessionHistory.appendRecord("observations", record, "observation", {
      actionCount: this.actionCount,
      responseCount: this.responseCount,
      text,
      imagePath: observation.imagePath,
      contextPrunedImageCount: prunedImages.length
    }, this.historySnapshot());
  }

  recordModelResponse(record) {
    return this.sessionHistory.appendRecord("modelResponses", record, "model_response", {
      model: record.model,
      responseCount: record.responseCount,
      thinking: record.thinking,
      content: record.content,
      actionPackage: record.actionPackage,
      parseError: record.parseError
    }, this.historySnapshot({ modelName: record.model }));
  }

  recordPolicyCandidate(record) {
    if (!this.sessionHistory || !record) {
      return Promise.resolve();
    }
    return this.sessionHistory.appendRecord("policyActionCandidates", record, "policy_action_candidate", {
      candidateId: record.candidateId,
      basedOnObservationId: record.basedOnObservationId,
      valid: record.valid,
      parseError: record.parseError || null
    }, this.historySnapshot());
  }

  async recordAction(record) {
    await this.sessionHistory.appendRecord("actions", record, "action", {
      actionCount: record.actionCount,
      responseCount: record.responseCount,
      description: record.description,
      action: record.action,
      operations: record.operations,
      operationCount: record.operationCount
    }, this.historySnapshot());
    return this.sessionHistory.appendRecord("executedActions", {
      actionId: record.executedActionId,
      candidateId: record.candidateId,
      basedOnObservationId: record.basedOnObservationId,
      actionSeq: record.actionCount,
      action: record.action,
      execution: record.execution,
      description: record.description
    }, "executed_action", {
      actionId: record.executedActionId,
      candidateId: record.candidateId,
      actionSeq: record.actionCount,
      basedOnObservationId: record.basedOnObservationId
    }, this.historySnapshot());
  }

  recordTaskResult(result, item) {
    if (!this.sessionHistory) {
      return Promise.resolve();
    }
    const record = {
      at: new Date().toISOString(),
      sequence: item.sequence ?? null,
      observation: item.observation ? observationForHistory(item.observation) : null,
      metadata: item.metadata || {},
      result: cloneJson(result)
    };
    return this.sessionHistory.appendRecord("taskResults", record, "task_result", {
      sequence: record.sequence,
      signal: result.signal || "none",
      terminal: Boolean(result.terminal),
      imagePath: result.imagePath || item.observation?.imagePath || ""
    }, this.historySnapshot());
  }

  recordHistoryError(error) {
    const record = { at: new Date().toISOString(), message: error.message, stack: error.stack, failure: error.failure || null, cleanupErrors: error.cleanupErrors || [] };
    return this.sessionHistory
      ? this.sessionHistory.appendRecord("errors", record, "error", { message: error.message }, this.historySnapshot({ status: "error" }))
      : Promise.resolve();
  }

  async finishSessionHistory(reason) {
    if (!this.sessionHistory) {
      return;
    }
    try {
      this.frameworkRuntime?.stop(reason);
      const handoff = isTrivialPolicy(this.defaultConfig) ? null : this.currentSession?.createContinuationHandoff() || null;
      const handoffPath = handoff
        ? await writeContinuationHandoff(this.sessionHistory.filePath, handoff)
        : null;
      if (this.pendingSharedKnowledge.length > 0 && this.sharedDocuments?.persistBehaviorGuideEntries) {
        const committed = await this.sharedDocuments.persistBehaviorGuideEntries(this.pendingSharedKnowledge).catch(() => []);
        if (committed.length > 0) {
          await this.sessionHistory.appendEvent("shared_knowledge_auto_committed", {
            entries: committed.map((entry) => ({
              text: entry.text,
              sourceObservationId: entry.sourceObservationId,
              confidence: entry.confidence
            }))
          }, this.historySnapshot());
        }
      }
      await this.sessionHistory.appendEvent("session_stopped", { reason }, this.historySnapshot({
        status: this.lastError ? "error" : "stopped",
        stoppedAt: this.stoppedAt,
        memory: this.currentSession?.brain?.memory || null,
        sessionHandoff: handoff ? { ...handoff, filePath: handoffPath } : null
      }));
    } finally {
      // A failed final event/handoff must not skip the writer drain/close.
      // flush rethrows any poisoned writer instead of marking evidence complete.
      await this.sessionHistory.flush();
    }
  }
}

module.exports = { AgentRuntime };

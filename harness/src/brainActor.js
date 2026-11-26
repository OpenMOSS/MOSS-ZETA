const { chatCompletion, resolveModel } = require("./llmClient");
const { imagePathToDataUrl } = require("./captureStore");
const { validateBrainOutput, brainView } = require("./brainSchema");
const { retryableModelError, retryDelay } = require("./modelRetry");
const { PAUSED_STICK_MIN_RADIUS } = require("./actionRuntime");
const { createHash } = require("node:crypto");

function providerConfig(policyConfig) {
  if (policyConfig.brainProvider?.inheritPolicy !== false) {
    return {
      ...policyConfig,
      allowThinking: policyConfig.brainProvider?.allowThinking ?? policyConfig.allowThinking,
      reasoningEffort: policyConfig.brainProvider?.reasoningEffort ?? policyConfig.reasoningEffort,
      maxTokens: policyConfig.brainProvider?.maxTokens === undefined ? policyConfig.maxTokens : policyConfig.brainProvider.maxTokens,
      temperature: policyConfig.brainProvider?.temperature ?? policyConfig.temperature
    };
  }
  return {
    ...policyConfig,
    inferenceBackend: "openai_http",
    baseUrl: policyConfig.brainProvider.baseUrl,
    apiFormat: policyConfig.brainProvider.apiFormat || "chat_completions",
    apiKey: policyConfig.brainProvider.apiKey,
    modelName: policyConfig.brainProvider.modelName,
    allowThinking: policyConfig.brainProvider.allowThinking,
    reasoningEffort: policyConfig.brainProvider.reasoningEffort,
    maxTokens: policyConfig.brainProvider.maxTokens === undefined ? policyConfig.maxTokens : policyConfig.brainProvider.maxTokens,
    temperature: policyConfig.brainProvider.temperature ?? policyConfig.temperature
  };
}

function parseBrainResponse(content, agentMode, type = "full", scope = 'memory_think') {
  return validateBrainOutput(JSON.parse(String(content || "").trim()), agentMode, type, scope);
}

class BrainActor {
  constructor(options = {}) {
    this.config = options.config;
    this.sharedDocuments = options.sharedDocuments;
    this.sessionState = options.sessionState;
    this.chatCompletion = options.chatCompletion || chatCompletion;
    this.resolveModel = options.resolveModel || resolveModel;
    this.recordEvent = options.recordEvent || (() => {});
    this.recoverMobility = options.recoverMobility;
    if (this.config?.mobilityRecovery?.enabled && this.config.brainPrefillEnabled) throw Error('Recovery tool currently requires ordinary Brain inference, without prefill');
    if (this.config?.mobilityRecovery?.enabled && this.config.policyHotSwapEnabled) throw Error('Recovery tool cannot run with the automatic dual-service Policy coordinator');
    this.abortController = null;
    this.currentModel = "";
    if (this.config?.brainPrefillEnabled === true) {
      const { BrainPrefillCoordinator } = require('./brainPrefillCoordinator');
      const { BrainPrefillTransport } = require('./brainPrefillTransport');
      const { stableSystem, buildPrefix } = require('./brainPrefillMessages');
      this.prefillTransport = options.brainPrefillTransport || new BrainPrefillTransport();
      this.prefillSystem = stableSystem(this.config, this.sharedDocuments);
      this.prefill = new BrainPrefillCoordinator({ enabled: true, intervalMs: this.config.brainPrefillIntervalMs ?? 1000,
        isCurrent: snapshot => snapshot.sessionId === this.sessionState.sessionId
          && snapshot.brainRevision === this.sessionState.brain.revision,
        prepare: (snapshot, signal) => buildPrefix(snapshot, this.prefillSystem, signal),
        run: async (prefix, options) => {
          const config = providerConfig(this.config);
          const model = await this.resolveModel(config, { signal: options.signal });
          options.signal.throwIfAborted();
          return this.prefillTransport.prefill(config, prefix.messages, { ...options, model });
        }, recordEvent: (type, payload) => this.recordEvent(type, payload) });
    }
  }

  canOfferPrefill() { return Boolean(this.prefill?.canOffer() && !this.abortController); }
  offerPrefill(snapshotOrFactory) { return this.canOfferPrefill() ? this.prefill.offer(snapshotOrFactory) : false; }
  requestPrefill(snapshotOrFactory) { return this.offerPrefill(snapshotOrFactory); }
  invalidatePrefill(reason) { this.prefill?.invalidate(reason); }
  getPrefillStatus() { return this.prefill?.getStatus() || { enabled: false }; }
  getPrefillPinnedPaths() { return this.prefill?.getPinnedPaths() || []; }

  async buildPrefilledMessages(request, reusablePrefix, signal) {
    const preparationStarted = performance.now();
    const { freezeSnapshot, buildPrefix, buildFinal, prefixEligible } = require('./brainPrefillMessages');
    const state = this.sessionState;
    let prefix = prefixEligible(reusablePrefix, request, state, this.prefillSystem) ? reusablePrefix : null;
    const reused = Boolean(prefix);
    if (!prefix) {
      const trajectory = request.trajectory || state.brainObservationWindow(request.cutoffActionSeq,
        Math.min(8, this.config.brainTrajectoryObservationLimit || 8));
      prefix = await buildPrefix(freezeSnapshot({ sessionId: state.sessionId, brainRevision: state.brain.revision,
        cutoffActionSeq: request.cutoffActionSeq, previousActionSeq: request.previousActionSeq ?? state.brain.lastBrainCommitActionSeq ?? 0,
        basedOnObservationId: request.basedOnObservationId, trajectory,
        evidenceRecords: request.evidenceRecords || state.timeline.readRecords() }), this.prefillSystem, signal);
    }
    const result = await buildFinal(prefix, { ...request, brainFallbackScope: this.config.brainFallbackScope }, state, signal);
    signal?.throwIfAborted();
    request.inputEvidence = result.inputEvidence;
    this.inputEvidence = result.inputEvidence;
    request.prefillAudit = { enabled: true, reusedCompletedPrefix: reused,
      jobId: reused ? prefix.jobId : null, prefixHash: prefix.prefixHash,
      stableMessagePrefixExact: true, prefillCompletedAt: reused ? prefix.completedAt : null,
      prefillCutoffActionSeq: prefix.snapshot.cutoffActionSeq,
      prefillBackendRequestId: reused ? prefix.prefillResponse.audit?.backendRequestId : null,
      prefillCache: reused ? prefix.prefillResponse.audit : null,
      formalMessagePreparationMs: performance.now() - preparationStarted,
      actualTokenPrefixMatch: null, actualTokenPrefixMatchReason: 'Awaiting formal backend token audit; message identity alone is not token verification.' };
    const provider = providerConfig(this.config);
    await this.recordEvent('brain_model_input', { schemaVersion: 2, model: this.currentModel,
      basedOnObservationId: request.basedOnObservationId, cutoffActionSeq: request.cutoffActionSeq,
      brainPromptProfile: this.config.brainPromptProfile || 'default', systemSha256: prefix.systemSha256,
      generation: { maxTokens: provider.maxTokens, temperature: provider.temperature,
        allowThinking: provider.allowThinking, timeoutMs: this.config.brainRequestTimeoutMs ?? 0 },
      inputEvidence: result.inputEvidence, images: result.images, prefill: request.prefillAudit,
      messages: result.referenceMessages });
    return result.messages;
  }

  async buildMessages(request) {
    const state = this.sessionState;
    let trajectory = request.trajectory || state.brainObservationWindow(request.cutoffActionSeq,
      this.config.brainTrajectoryObservationLimit || 8);
    if (!trajectory.some((record) => record.observationId === request.basedOnObservationId && record.imagePath)) {
      throw new Error(`Brain input window is missing trigger observation ${request.basedOnObservationId} or its screenshot.`);
    }
    const view = {
      goal: state.userGoal,
      previous: request.previous || brainView(state), updateType: request.updateType || "full"
    };
    if (request.policyTriggerThinking) view.trigger = request.policyTriggerThinking;
    if (request.policyEvidence) view.policyEvidence = { ...request.policyEvidence,
      note: 'This is an observation-based Policy judgment. Its associated action may be unexecuted; only executedSincePreviousObservation is completed action evidence.' };
    if (request.policyProposal) view.policyProposal = {
      ...request.policyProposal,
      note: 'Policy proposes this next instruction and initial judgment. Validate against the attached observations; this is not yet committed state.'
    };
    if (state.recoveryMode) view.recovery = "Previous instruction may be invalid. Inspect latest HUD and task outcome before replanning.";
    const content = [{ type: "text", text: JSON.stringify(view) }];
    const evidence = { observations: [], actions: [], imageCount: 0 };
    let previousActionSeq = Number(request.previousActionSeq ?? state.brain.lastBrainCommitActionSeq ?? 0);
    const evidenceRecords = request.evidenceRecords || state.timeline.readRecords();
    const sourceObservationId = request.policyEvidence?.observationId;
    if (sourceObservationId && !trajectory.some(record => record.observationId === sourceObservationId)) {
      const source = evidenceRecords.find(record => record.observationId === sourceObservationId && record.imagePath);
      if (!source) throw new Error(`Brain is missing Policy think source screenshot ${sourceObservationId}`);
      trajectory = [...trajectory, source].sort((a, b) => Date.parse(a.capturedAt || a.at) - Date.parse(b.capturedAt || b.at));
    }
    let previousFrameAt = -Infinity;
    for (const [index, record] of trajectory.entries()) {
      const actions = evidenceRecords.filter((item) => item.type === "executed_action"
        && item.actionSeq > previousActionSeq && item.actionSeq <= record.actionSeq);
      if (actions.length) {
        content.push({ type: "text", text: JSON.stringify({
          executedSincePreviousObservation: actions.map((item) => item.raw || item.actionText)
        }) });
        evidence.actions.push(...actions.map((item) => ({ actionId: item.actionId, actionSeq: item.actionSeq,
          basedOnObservationId: item.basedOnObservationId, beforeFrame: index + 1 })));
      }
      previousActionSeq = Math.max(previousActionSeq, record.actionSeq);
      const frameAt = Date.parse(record.capturedAt || record.at);
      const interruptions = evidenceRecords.filter((item) => item.type === "interrupted_action"
        && Date.parse(item.completedAt || item.at) > previousFrameAt
        && Date.parse(item.completedAt || item.at) <= frameAt);
      if (interruptions.length) content.push({ type: "text", text: JSON.stringify({
        interruptedActions: interruptions.map((item) => ({ requestedPackage: item.requestedPackage,
          completed: false, executedChunkCount: item.executedChunkCount, actualSegments: item.actualSegments || [],
          note: "Only a portion may have executed. The complete requested package is NOT executed evidence." }))
      }) });
      if (record.executionAtCapture?.activePackageSeq) content.push({ type: "text", text:
        "An action package was still executing during this capture; its full result is not yet visible in this frame." });
      previousFrameAt = frameAt;
      const ocr = (record.ocr?.results || []).filter((item) => String(item.text || "").trim())
        .map((item) => ({ roi: item.roi, text: item.text,
          ...(Number.isFinite(item.confidence ?? item.score) ? { confidence: item.confidence ?? item.score } : {}) }));
      content.push({ type: "text", text: JSON.stringify({ frame: index + 1, ...(record.observationId === sourceObservationId ? { isPolicyThinkSource: true, observationId: record.observationId, capturedAt: record.capturedAt || record.at } : {}), ...(ocr.length ? { ocr } : {}) }) });
      evidence.observations.push({ observationId: record.observationId, actionSeq: record.actionSeq,
        frame: index + 1, isPolicyThinkSource: record.observationId === sourceObservationId, imagePath: record.imagePath, minimapPath: record.navigation?.minimap?.imagePath || null,
        capturedAt: record.capturedAt || record.at, hasMinimap: Boolean(record.navigation?.minimap?.imagePath) });
      if (record.imagePath) content.push({ type: "image_ref", path: record.imagePath, mimeType: record.mimeType || "image/jpeg" });
      if (record.navigation?.minimap?.imagePath) {
        content.push({ type: "text", text: `Minimap crop from the same frame ${index + 1}, not a new observation.` });
        content.push({ type: "image_ref", path: record.navigation.minimap.imagePath,
          mimeType: record.navigation.minimap.mimeType || "image/jpeg" });
      }
    }
    const promptType = request.updateType === 'fallback' && this.config.brainFallbackScope === 'full' ? 'full' : request.updateType;
    const system = [
      promptType === "fallback" ? require("./brainProtocol").FALLBACK_PROMPT
        : request.updateType === "small" ? (this.sharedDocuments.get("brainMemorySystemPrompt") || require("./brainProtocol").MEMORY_PROMPT) : this.sharedDocuments.get("brainSystemPrompt"),
      this.sharedDocuments.get("gtaKnowledge"),
      this.config.mobilityRecovery?.enabled ? require('./mobilityRecovery').RECOVERY_PROMPT + '\n固定恢复参数：' + JSON.stringify(this.config.mobilityRecovery) : '',
      `Executed packages use signed Lx Ly Rx Ry levels -9..+9 and button chunks. Each package has ${this.config.hfChunksPerPackage || 6} chunks of ${this.config.hfChunkDurationMs || 33}ms. Nonzero stick radius is floored at ${PAUSED_STICK_MIN_RADIUS} and capped at 1. Images are chronological; intervening executed packages are evidence, not proposed actions.`
    ].filter(Boolean).join("\n\n");
    evidence.imageCount = content.filter((part) => part.type === "image_ref").length;
    request.inputEvidence = evidence;
    this.inputEvidence = evidence; // Last-started audit view; replies use their own request evidence.
    const provider = providerConfig(this.config);
    await this.recordEvent("brain_model_input", { schemaVersion: 2, model: this.currentModel,
      basedOnObservationId: request.basedOnObservationId, cutoffActionSeq: request.cutoffActionSeq,
      brainPromptProfile: this.config.brainPromptProfile || "default",
      systemSha256: createHash("sha256").update(system).digest("hex"),
      generation: { maxTokens: provider.maxTokens, temperature: provider.temperature,
        allowThinking: provider.allowThinking, reasoningEffort: provider.reasoningEffort,
        timeoutMs: this.config.brainRequestTimeoutMs ?? 0 },
      inputEvidence: evidence,
      messages: [{ role: "system", content: system }, { role: "user", content }] });
    const materialized = [];
    for (const part of content) {
      materialized.push(part.type === "image_ref"
        ? { type: "image_url", image_url: { url: await imagePathToDataUrl(part.path, part.mimeType) } }
        : part);
    }
    return [{ role: "system", content: system }, { role: "user", content: materialized }];
  }

  async infer(request) {
    request = { ...request };
    const config = providerConfig(this.config);
    const controller = new AbortController();
    this.abortController = controller;
    const reusablePrefix = this.prefill?.beginInference();
    const maxRetries = Math.min(5, Math.max(0, Math.trunc(this.config.brainMaxRetries ?? 1)));
    let messages, toolCalls = 0;
    const messagePromise = this.prefill
      ? this.buildPrefilledMessages(request, reusablePrefix, controller.signal) : this.buildMessages(request);
    messagePromise.catch(() => {});
    try {
      for (let attempt = 0; ; attempt += 1) {
        controller.signal.throwIfAborted();
        try {
          this.currentModel = await this.resolveModel(config, { signal: controller.signal });
          controller.signal.throwIfAborted();
          messages ||= await messagePromise;
          controller.signal.throwIfAborted();
          const complete = this.prefill ? this.prefillTransport.generate.bind(this.prefillTransport) : this.chatCompletion;
          const response = await complete(config, messages, {
            model: this.currentModel, timeoutMs: this.config.brainRequestTimeoutMs ?? 0,
            signal: controller.signal, disableCompactRetry: true
          });
          if(controller.signal.aborted){
            // A provider may complete after cancellation. Retain the response,
            // but never invoke an old tool (even its cleanup could release the
            // current controller). Logging cannot override cancellation.
            await Promise.resolve().then(()=>this.recordEvent('brain_response_discarded',{
              reason:'request_cancelled',model:response.model||this.currentModel,content:response.content,
              raw:response.raw||null,finishReason:response.finishReason||null,inputEvidence:request.inputEvidence
            })).catch(()=>{});
            controller.signal.throwIfAborted();
          }
          let parsed;
          try {
            if (["length", "max_tokens"].includes(response.finishReason)) throw new Error("Brain output budget exhausted");
            const control = JSON.parse(String(response.content || '').trim());
            if (Object.hasOwn(control, 'tool_request')) {
              if (!this.config.mobilityRecovery?.enabled || !this.recoverMobility
                || Object.keys(control).length !== 1 || control.tool_request?.name !== 'recover_mobility'
                || Object.keys(control.tool_request).sort().join(',') !== 'arguments,name') throw Error('Invalid or disabled Brain recovery tool');
              if (++toolCalls > this.config.mobilityRecovery.maxCallsPerEpisode) throw Error('Brain tool round limit reached');
              require('./mobilityRecovery').validateRecovery(control.tool_request.arguments, this.config.mobilityRecovery);
              await this.recordEvent('brain_tool_request', { protocol: 'structured-v1', model: this.currentModel,
                reasoningEffort: config.reasoningEffort, content: response.content, inputEvidence: request.inputEvidence });
              const receipt = await this.recoverMobility(control.tool_request.arguments, controller.signal);
              controller.signal.throwIfAborted();
              if (receipt?.status !== 'inputs_completed' || !receipt.completed?.length
                || receipt.completed.some(item => item.completed !== true) || !receipt.recoveryContext) {
                throw Object.assign(Error('Recovery execution was not confirmed complete'), { recoveryExecutionError: true });
              }
              // This completes a tool operation, not a new semantic Brain
              // revision. The owner resumes the original Policy dialogue.
              return { recoveryCompleted: true, recoveryContext: receipt.recoveryContext, recoveryReceipt: receipt,
                inputEvidence: request.inputEvidence, model: response.model || this.currentModel,
                content: response.content, raw: response.raw || null, finishReason: response.finishReason || null };
            }
            parsed = parseBrainResponse(response.content, this.config.agentMode, request.updateType || "full", this.config.brainFallbackScope);
          } catch (error) {
            error.modelProtocolError = true;
            error.brainResponse = { model: response.model || this.currentModel,
              content: String(response.content || ""), thinking: String(response.thinking || ""),
              raw: response.raw || null,
              finishReason: response.finishReason || null, usage: response.usage || null,
              inputEvidence: request.inputEvidence, prefill: request.prefillAudit,
              wallLatencyMs: response.clientTiming?.client_total_ms ?? null };
            throw error;
          }
          controller.signal.throwIfAborted();
          const prefill = this.prefill ? { ...request.prefillAudit, ...response.prefillAudit } : undefined;
          if (prefill) await this.recordEvent('brain_prefill_formal_result', { ...prefill, inputEvidence: request.inputEvidence,
            clientTiming: response.clientTiming, rawResponse: response.raw });
          return { ...parsed, brainFallbackScope: this.config.brainFallbackScope || 'memory_think', updateType: request.updateType || "full", inputEvidence: request.inputEvidence,
            ...(prefill ? { prefill, clientTiming: response.clientTiming, raw: response.raw } : {}), model: response.model || this.currentModel,
            content: response.content, thinking: response.thinking || "",
            raw: response.raw || null,
            finishReason: response.finishReason || null, usage: response.usage || null,
            wallLatencyMs: response.clientTiming?.client_total_ms ?? null };
        } catch (error) {
          if (error.recoveryExecutionError) throw error;
          if (controller.signal.aborted || attempt >= maxRetries || !retryableModelError(error)) throw error;
          const delayMs = Math.min(5000, (this.config.brainRetryDelayMs ?? 500) * 2 ** attempt);
          await this.recordEvent("brain_model_retry", { attempt: attempt + 1, maxRetries, delayMs,
            model: this.currentModel, error: error.message, status: error.status || null });
          await retryDelay(delayMs, controller.signal);
        }
      }
    } finally {
      if (this.abortController === controller) this.abortController = null;
      this.prefill?.endInference();
    }
  }

  stop(reason = "session stopped") {
    this.abortController?.abort(new Error(reason));
    // Superseding an unfinished Brain request does not change committed state.
    // Runtime explicitly invalidates completed prefixes on commit/task change/stop.
    this.prefill?.cancelInFlight(reason);
  }
}

module.exports = { BrainActor, parseBrainResponse, providerConfig };

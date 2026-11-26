const { isTrivialPolicy, extractTrivialPolicyResponse } = require("./trivialPolicy");
const { randomUUID } = require("node:crypto");
const { PolicyTransport } = require("./policyTransport");
const { chatCompletion, cancelChatTurn, recoverPolicySession } = require("./llmClient");
const { retryableModelError, retryDelay } = require("./modelRetry");
const { parseHighFrequencyPackage } = require("../../cadence/src/highFrequencyActionParser");
const { extractPolicyResponse } = require("./policyResponse");

const DEFAULT_RETRY_BASE_DELAY_MS = 500;

function clone(value) {
  return value === undefined ? null : JSON.parse(JSON.stringify(value));
}

const transientRequestError = retryableModelError;

class PolicyActor {
  constructor(options = {}) {
    this.sessionState = options.sessionState;
    this.getConfig = options.getConfig || (() => options.config || {});
    this.onSessionLost = options.onSessionLost || (async () => {});
    this.onDelivered = options.onDelivered || (() => {});
    this.getMessages = options.getMessages || (() => []);
    this.appendMessage = options.appendMessage || (() => {});
    this.buildRequestMessages = options.buildRequestMessages || (async (messages) => messages);
    this.createCandidate = options.createCandidate || ((candidate) => this.sessionState.addPolicyCandidate(candidate));
    this.recordCandidate = options.recordCandidate || (async () => {});
    this.recordModelResponse = options.recordModelResponse || (async () => {});
    this.recordEvent = options.recordEvent || (async () => {});
    this.buildRetryText = options.buildRetryText || (() => "Return a valid action package.");
    this.emit = options.emit || (() => {});
    this.requestBrainReplan = options.requestBrainReplan || (() => false);
    this.shouldStop = options.shouldStop || (() => false);
    this.model = options.model || "";
    this.resolveModel = options.resolveModel || ((config) => config.modelName);
    this.chatCompletion = options.chatCompletion || chatCompletion;
    this.requestModelOverride = options.requestModelWithRetry || null;
    this.maxRetries = Number.isFinite(Number(options.maxRetries)) ? Math.max(0, Math.trunc(Number(options.maxRetries))) : 2;
    this.retryBaseDelayMs = Number.isFinite(Number(options.retryBaseDelayMs))
      ? Math.max(0, Math.trunc(Number(options.retryBaseDelayMs)))
      : DEFAULT_RETRY_BASE_DELAY_MS;
    this.responseCount = 0;
    this.invalidPackageRetries = 0;
    this.lastParseError = "";
    this.abortController = null;
    this.transport = new PolicyTransport();
    this.contextEpoch = 0;
    this.cancelChatTurn = options.cancelChatTurn || cancelChatTurn;
    this.recoverPolicySession = options.recoverPolicySession || recoverPolicySession;
    this.historyRecovery = null;
    this.activeGeneration = null;
  }

  resetContext(reason = "Policy context invalidated") {
    this.contextEpoch += 1;
    this.abortController?.abort(new Error(reason));
    this.transport.reset();
    // Do not delete the session before its in-flight withdrawal is acknowledged.
    // The watchdog awaits the same receipt before allowing automatic recovery.
    const cancellation = this.historyRecovery?.promise || this.activeGeneration?.cancelPromise;
    this.historyRecovery = null;
    if (cancellation) void cancellation.catch(() => {}).then(() => this.transport.releaseRetired());
    else void this.transport.releaseRetired();
  }

  fenceForHandoff(reason) {
    this.contextEpoch += 1;
    this.abortController?.abort(Error(reason));
  }

  currentInstruction() {
    return this.sessionState?.currentPolicyInstruction || null;
  }

  requestReplan(reason = "policy_request") {
    return this.requestBrainReplan(reason);
  }

  setModel(model) {
    this.model = model || this.model;
  }

  getStatus() {
    return {
      responseCount: this.responseCount,
      invalidPackageRetries: this.invalidPackageRetries,
      lastParseError: this.lastParseError
    };
  }

  cancelUnstartedGeneration(reason = 'history_dependency_deadline', retireContext = false) {
    const active = this.activeGeneration;
    if (!active || (!retireContext && (active.hasExecutableState || active.brainCall))) return false;
    if (active.cancelPromise) return active.cancelPromise;
    active.cancelled = true;
    active.cancelReason = reason;
    active.cancelledBeforeDispatch = active.dispatchStarted === false;
    if (active.cancelledBeforeDispatch) {
      // The gate or native-arm await can still be pending. No fetch has been
      // invoked for this request, so the deployment has no turn to withdraw.
      const local = { request_id: active.requestId, context_id: active.prepared.contextId,
        cancelled_locally: true, not_dispatched: true, reason };
      active.cancelPromise = Promise.resolve().then(() => this.recordEvent('policy_generation_cancelled_before_dispatch', local))
        .then(() => local);
    } else {
      // Once fetch is invoked delivery may be uncertain. Require the genuine
      // remote acknowledgement; an unknown-turn error is never a local success.
      active.cancelPromise = this.cancelChatTurn(active.config, active.prepared.contextId, active.requestId);
    }
    active.cancelPromise.catch(() => {}); // The request continuation must observe the receipt/error.
    active.controller.abort(Object.assign(new Error(reason), { policyDependencyDropped: true,
      ...(active.cancelledBeforeDispatch ? { policyNotDispatched: true } : {}) }));
    return active.cancelPromise;
  }

  beginHistoryRecovery(state) {
    if (this.historyRecovery) return this.historyRecovery;
    const active = this.activeGeneration;
    const config = { ...this.getConfig(), modelName: this.model || this.getConfig().modelName };
    const sessionId = this.transport.sessionId;
    const remote = Boolean(this.transport.initialized || this.transport.dispatched || active?.dispatchStarted);
    // Invalidates every continuation, including one between HTTP completion and
    // candidate admission. The backend alone decides which full turn committed.
    this.contextEpoch += 1;
    const recovery = { actor: this, actorEpoch: this.contextEpoch, sessionId, state,
      recoveryId: randomUUID(), remote, requestId: active?.dispatchStarted ? active.requestId : null,
      window: { max: config.policyWindowMaxRounds ?? 5, keep: config.policyWindowKeepRounds ?? 3 } };
    this.historyRecovery = recovery;
    recovery.promise = remote
      ? Promise.resolve().then(() => this.recoverPolicySession(config, recovery))
      : Promise.resolve({ ready: true, local_only: true, history_turns: 0 });
    recovery.promise.catch(() => {});
    if (active) {
      active.cancelled = true;
      active.cancelReason = 'watchdog_history_recovery';
      active.cancelledBeforeDispatch = !active.dispatchStarted;
      active.historyRecovery = true;
      active.cancelPromise = recovery.promise;
      active.controller.abort(Object.assign(Error(active.cancelReason), { policyDependencyDropped: true }));
    } else this.abortController?.abort(Error('watchdog_history_recovery'));
    return recovery;
  }

  finishHistoryRecovery(recovery) {
    if (this.historyRecovery !== recovery || this.contextEpoch !== recovery.actorEpoch
      || this.transport.sessionId !== recovery.sessionId)
      throw Error('Policy state changed during history recovery');
    if (recovery.remote) this.transport.initialized = true;
    else this.transport.sent = false;
    this.historyRecovery = null;
  }

  async requestModelWithRetry(messages, submissionGate = null, onPolicyStream = null) {
    if (this.requestModelOverride) {
      if (submissionGate) await submissionGate.wait();
      return this.requestModelOverride(messages);
    }
    const requestConfig = { ...this.getConfig() };
    const requestModel = this.model || requestConfig.modelName;
    let retries = 0;
    let lengthRetries = 0;
    let requestMessages = messages;
    const requestId = randomUUID();
    let prepared = null;
    while (true) {
      if (this.shouldStop()) throw new Error("stop requested");
      this.abortController = new AbortController();
      let requestDispatched = false;
      let requestStartedAt;
      let response;
      let active;
      try {
        const config = requestConfig;
        prepared ||= this.transport.prepare({ ...config, modelName: requestModel }, requestMessages, requestId);
        if (prepared.fallback && !this.transportFallbackReported) {
          this.transportFallbackReported = true;
          this.emit('log', prepared.fallback);
          await this.recordEvent('policy_transport_fallback', { requested: config.policyTransport, effective: 'stateless', reason: prepared.fallback });
        }
        requestStartedAt = new Date().toISOString();
        active = { requestId, prepared, config, controller: this.abortController,
          hasExecutableState: false, brainCall: false, cancelled: false, dispatchStarted: false };
        this.activeGeneration = active;
        response = await this.chatCompletion(config, prepared.messages, {
          ...prepared.options,
          onRequestDispatch: () => {
            active.controller.signal.throwIfAborted();
            active.dispatchStarted = true;
            this.transport.dispatched = true;
            requestDispatched = true;
          },
          ...(onPolicyStream ? { onPolicyStream: event => {
            if (active.cancelled) return;
            if (event.type === 'state') active.hasExecutableState = true;
            if (event.type === 'brain_marker') active.brainCall = true;
            return onPolicyStream({ ...event, requestId, contextId: prepared.contextId });
          } } : {}),
          // Submission lateness bounds unsent observations, not model generation.
          // Let an admitted stream finish think or emit a late action; the native
          // executor checks its action window/ownership and the valid-action
          // watchdog remains responsible for retiring a stalled active context.
          ...(submissionGate ? { beforeSend: signal => submissionGate.wait(signal) } : {}),
          model: requestModel,
          timeoutMs: config.policyRequestTimeoutMs ?? 120000,
          signal: this.abortController.signal,
          disableCompactRetry: true,
          preserveThinkingTags: true
        });
        if (active.cancelled) throw Object.assign(Error(active.cancelReason || 'Policy generation cancelled'), { policyDependencyDropped: true });
        this.transport.accept(prepared, response);
        void this.transport.releaseRetired();
        response.policyRequestId = requestId;
        response.policyContextId = prepared.contextId;
        return { response, attempts: retries + lengthRetries + 1 };
      } catch (error) {
        if (error.nonFatalSlotSkip) throw error;
        if (active?.cancelPromise) {
          const receipt = await active.cancelPromise;
          if (active.cancelledBeforeDispatch) throw Object.assign(Error('Unsent generation cancelled locally'), {
            policyDependencyDropped: true, nonFatalSlotSkip: true, policyNotDispatched: true
          });
          await this.recordEvent(active.historyRecovery ? 'policy_generation_recovery_fenced' : 'policy_generation_withdrawn', { requestId, contextId: prepared.contextId,
            reason: active.cancelReason, receipt, streamDiagnostic: error.streamDiagnostic || null });
          throw Object.assign(Error('Unstarted generation withdrawn by deployment'), { policyDependencyDropped: true, receipt });
        }
        // Successful requests already have their full input/response history.
        // Retain failed attempts too, before retry/cancellation can discard them.
        // Only transport inputs are recorded; config/API credentials are excluded.
        if (requestDispatched) {
          try {
            await this.recordEvent("policy_request_failed", {
              requestId, contextId: prepared.contextId, requestStartedAt,
              attempt: retries + lengthRetries + 1,
              model: requestModel,
              requestMessages: clone(prepared.messages),
              requestOptions: clone(prepared.options),
              modelParameters: {
                temperature: requestConfig.temperature,
                maxTokens: requestConfig.maxTokens,
                allowThinking: requestConfig.allowThinking,
                reasoningEffort: requestConfig.reasoningEffort,
                timeoutMs: requestConfig.policyRequestTimeoutMs ?? 120000
              },
              rawResponse: response?.raw || null,
              streamDiagnostic: error.streamDiagnostic || null,
              error: { message: error.message, name: error.name, status: error.status ?? null },
              cancelled: this.shouldStop() || this.abortController.signal.aborted
            });
          } catch (evidenceError) {
            // Do not retry after losing evidence or hide the original model error.
            throw new Error(`${error.message}; failed request evidence write: ${evidenceError.message}`, { cause: error });
          }
        }
        if (submissionGate || this.shouldStop() || this.abortController.signal.aborted || error.nonRetryablePolicyProtocol) throw error;
        // Retry the identical logical request. A missing/expired session is
        // surfaced explicitly; never reinterpret it as a cold continuation.
        const sessionMode = this.transport.effectiveMode === "lumine_session";
        if (sessionMode && error.status === 409 && !/in progress/i.test(error.message)) {
          error.sessionLost = /lost|expired|unavailable|Unknown session/i.test(error.message)
            || (requestConfig.policyHandoffTarget === 'small' && /retired|recovery failed/i.test(error.message));
          throw error;
        }
        const finish = error.choice?.finish_reason ?? error.body?.stop_reason;
        if (!requestConfig.policyHandoffTarget && !isTrivialPolicy(requestConfig) && !sessionMode && !lengthRetries && ["length", "max_tokens"].includes(finish)
          && /^LLM response did not contain message content\./.test(error.message || "")) {
          lengthRetries = 1;
          prepared = null;
          requestMessages = clone(messages);
          const note = "The previous response exhausted its output budget without an action. Use minimal reasoning: decide only the next package, do not restate the scene or compare long plans. Return one action package, either bare or wrapped in <a>...</a>; precede it with <think>...</think> (empty when no new judgment; <|call_brain|> inside think only for a stage update). No action has been executed since this observation.";
          if (requestMessages[0]?.role === "system") {
            if (typeof requestMessages[0].content === "string") requestMessages[0].content += `\n\n${note}`;
            else requestMessages[0].content.push({ type: "text", text: note });
          } else requestMessages.unshift({ role: "system", content: note });
          await this.recordEvent("policy_output_length_retry", {
            retryCount: 1, maxRetries: 1, nextAttempt: retries + lengthRetries + 1,
            maxTokens: this.getConfig().maxTokens, finishReason: finish
          });
          this.emit("log", "high-frequency Policy output budget exhausted without content; retrying once with concise guidance.");
          continue;
        }
        if (["length", "max_tokens"].includes(finish)) throw error;
        if (retries >= this.maxRetries || !(transientRequestError(error) || (sessionMode && error.status === 409 && /in progress/i.test(error.message)))) throw error;
        retries += 1;
        const delayMs = this.retryBaseDelayMs * (2 ** (retries - 1));
        await this.recordEvent("model_request_retry", {
          retryCount: retries,
          maxRetries: this.maxRetries,
          nextAttempt: retries + lengthRetries + 1,
          delayMs,
          error: error.message,
          status: error.status ?? null
        });
        this.emit("log", `high-frequency model request failed transiently; retrying ${retries}/${this.maxRetries} in ${delayMs} ms: ${error.message}`);
        if (delayMs > 0) await retryDelay(delayMs, this.abortController.signal);
      } finally {
        if (this.activeGeneration === active) this.activeGeneration = null;
        this.abortController = null;
      }
    }
  }

  async infer(observation) {
    const epoch = this.contextEpoch;
    const config = this.getConfig();
    const requestStartedAt = new Date().toISOString();
    const requestMessages = clone(this.getMessages());
    const requestPayloadMessages = await this.buildRequestMessages(requestMessages);
    if (epoch !== this.contextEpoch || this.shouldStop()) return null;
    let modelResult;
    try {
      modelResult = await this.requestModelWithRetry(requestPayloadMessages, observation.submissionGate, observation.onPolicyStream);
    } catch (error) {
      if (epoch !== this.contextEpoch) return null;
      if (error.sessionLost && !this.sessionRecoveryAttempted) {
        this.sessionRecoveryAttempted = true;
        await this.onSessionLost();
        return null;
      }
      throw error;
    }
    if (epoch !== this.contextEpoch) return null;
    this.sessionRecoveryAttempted = false;
    const { response, attempts } = modelResult;
    if (this.shouldStop()) {
      throw new Error("stop requested");
    }
    this.model = response.model || this.model || config.modelName;
    if (response.deploymentConnected) await this.recordEvent("policy_deployment_connected", {
      model: response.model, rendering: response.rendering, contextId: this.transport.sessionId
    });
    this.responseCount += 1;
    const generation = response.generation || response.raw?.lumine_outputs?.[0];
    const truncated = response.truncated === true || generation?.truncated === true ||
      [response.finishReason, response.finish_reason, generation?.finish_reason,
        response.raw?.choices?.[0]?.finish_reason].some(value => ["length", "max_tokens"].includes(value));
    if (truncated) {
      // Diagnostic evidence is separate from committed history and executable candidates.
      this.invalidPackageRetries += 1;
      this.lastParseError = "Policy output truncated; discarded before history or action admission";
      await this.recordModelResponse({ at: new Date().toISOString(), requestStartedAt,
        requestMessages, requestMessageCount: requestMessages.length, responseCount: this.responseCount,
        requestAttempts: attempts, model: this.model, content: response.content,
        responseMessage: response.responseMessage, finishReason: response.finishReason,
        generation, raw: response.raw, truncated: true, actionPackage: null,
        parseError: this.lastParseError, observationImagePath: observation?.imagePath || "" });
      await this.recordEvent("policy_output_truncated", { responseCount: this.responseCount,
        retryCount: this.invalidPackageRetries, maxRetries: config.policyMaxInvalidRetries });
      // Online sessions accept only a fresh observation when cold. Rebuild local OCR
      // and history through the same recovery path, not just the transport session ID.
      if (this.transport.effectiveMode === "lumine_session" || isTrivialPolicy(config)) {
        this.resetContext("Policy output truncated");
        await this.onSessionLost();
      }
      if (this.invalidPackageRetries > config.policyMaxInvalidRetries) {
        throw new Error(`Policy output remained truncated after ${config.policyMaxInvalidRetries} retries`);
      }
      return null;
    }
    this.onDelivered(requestMessages);
    let policyResponse;
    let protocolError = null;
    try {
      policyResponse = isTrivialPolicy(config)
        ? extractTrivialPolicyResponse(response.content, response.thinking)
        : extractPolicyResponse(response.content, response.thinking, config.policyRendering?.version);
    } catch (error) {
      protocolError = error;
      policyResponse = {
        content: response.content,
        thinking: response.thinking || ""
      };
      response.pausedPolicyResponseError = error.message;
    }

    const modelRecord = {
      at: new Date().toISOString(),
      requestStartedAt,
      requestMessageCount: requestMessages.length,
      requestMessages,
      responseCount: this.responseCount,
      requestAttempts: attempts,
      model: this.model,
      content: policyResponse.content,
      actionCode: policyResponse.content,
      thinking: policyResponse.thinking,
      triggerThinking: policyResponse.triggerThinking,
      brainRequested: policyResponse.brainRequested,
      policyProposal: policyResponse.policyProposal || null,
      instructionUpdateRequested: policyResponse.instructionUpdateRequested,
      responseMessage: response.responseMessage,
      finishReason: response.finishReason,
      raw: response.raw,
      clientTiming: response.clientTiming,
      ...(response.policyHandoff ? { policyHandoff: response.policyHandoff } : {}),
      observationImagePath: observation?.imagePath || ""
    };
    const emitModel = () => this.emit("model", {
      content: policyResponse.content,
      actionCode: policyResponse.content,
      thinking: policyResponse.thinking,
      highFrequency: true,
      responseCount: this.responseCount,
      model: this.model
    });
    if (!config.policyExecutionAcknowledgements && !config.policyStreaming) emitModel();

    let actionPackage;
    try {
      if (protocolError) {
        throw protocolError;
      }
      actionPackage = parseHighFrequencyPackage(policyResponse.content, {
        chunksPerPackage: config.hfChunksPerPackage,
        chunkDurationMs: config.hfChunkDurationMs,
        allowBarePackage: true,
        requireOnlyActionBlock: true
      });
    } catch (error) {
      if (config.policyExecutionAcknowledgements || config.policyStreaming) emitModel();
      const candidate = this.createCandidate({
        basedOnObservationId: observation.frameworkObservationId || `O${observation.observationId || ""}`,
        responseCount: this.responseCount,
        raw: policyResponse.content,
        valid: false,
        parseError: error.message
      });
      await this.recordCandidate(candidate);
      if (this.transport.effectiveMode === "lumine_session") this.transport.reset();
      this.invalidPackageRetries += 1;
      this.lastParseError = error.message;
      await this.recordModelResponse({ ...modelRecord, actionPackage: null, parseError: error.message });
      await this.recordEvent("invalid_package_retry", {
        responseCount: this.responseCount,
        retryCount: this.invalidPackageRetries,
        maxRetries: config.policyMaxInvalidRetries,
        error: error.message
      });
      if (this.invalidPackageRetries > config.policyMaxInvalidRetries) {
        throw new Error(`Policy response remained invalid after ${config.policyMaxInvalidRetries} retries: ${error.message}`);
      }
      if (this.transport.effectiveMode === "lumine_session") {
        this.resetContext();
        if (isTrivialPolicy(config)) await this.onSessionLost();
      } else if (isTrivialPolicy(config)) {
        await this.onSessionLost();
      } else {
        this.appendMessage({ role: "user", content: this.buildRetryText(this) });
      }
      return null;
    }

    // Preserve generated history byte-for-byte (including trained separators).
    // Deployment retains these same tokens; reconstructed whitespace diverges.
    if (!isTrivialPolicy(config) && !config.policyExecutionAcknowledgements) this.appendMessage({ role: "assistant", content: response.content });

    if (isTrivialPolicy(config)) this.appendMessage({ role: "assistant", content: response.content });
    this.invalidPackageRetries = 0;
    this.lastParseError = "";
    actionPackage.requestId = response.policyRequestId;
    actionPackage.policyContextId = response.policyContextId;
    actionPackage.responseCount = this.responseCount;
    actionPackage.model = this.model;
    actionPackage.predictedAt = new Date().toISOString();
    const candidate = this.createCandidate({
      basedOnObservationId: observation.frameworkObservationId || `O${observation.observationId || ""}`,
      responseCount: this.responseCount,
      raw: policyResponse.content,
      valid: true,
      actionText: actionPackage.actionText,
      package: {
        chunks: actionPackage.chunks,
        states: actionPackage.states,
        chunkDurationMs: actionPackage.chunkDurationMs
      }
    });
    // A scheduled packet has only the remainder of its 200 ms budget after
    // HTTP. Evidence is immutable here, but writing/snapshotting it must not
    // stand between parsing and native admission. The HF owner queues this
    // callback after enqueue and flushes it before session finalization.
    let evidencePromise;
    const recordEvidence = () => evidencePromise ||= (async () => {
      if (config.policyExecutionAcknowledgements || config.policyStreaming) emitModel();
      await this.recordCandidate(candidate);
      await this.recordModelResponse({ ...modelRecord, actionPackage });
    })();
    if (!config.policyExecutionAcknowledgements && !config.policyStreaming) await recordEvidence();
    return {
      candidate,
      actionPackage,
      response,
      responseCount: this.responseCount,
      ...((config.policyExecutionAcknowledgements || config.policyStreaming) ? { recordEvidence } : {}),
      thinking: policyResponse.thinking,
      triggerThinking: policyResponse.triggerThinking,
      brainRequested: policyResponse.brainRequested,
      policyProposal: policyResponse.policyProposal || null,
      instructionUpdateRequested: policyResponse.instructionUpdateRequested
    };
  }

  stop(reason = "session stopped") {
    this.resetContext(reason);
  }
}

module.exports = {
  PolicyActor
};

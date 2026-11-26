const { stripThinkingBlocks } = require("./actionParser");
const { usesGlmAnthropic, glmAnthropicEndpoint, glmAnthropicRequest, glmAnthropicResponse } = require("./glmAnthropic");
const { usesResponses, responsesEndpoint, responsesRequest, responsesResult } = require("./openaiResponses");

function normalizeBaseUrl(baseUrl) {
  const normalized = String(baseUrl || "").trim().replace(/\/+$/, "");
  if (!normalized) {
    throw new Error("LLM base URL is required.");
  }

  return normalized;
}

function endpointFor(baseUrl, endpoint) {
  const base = normalizeBaseUrl(baseUrl);
  if (/^https:\/\/open\.bigmodel\.cn\/api\/anthropic(?:\/|$)/i.test(base)) {
    throw new Error("Bigmodel protocol mismatch: /api/anthropic uses Messages, not Chat Completions. Set Base URL to https://open.bigmodel.cn/api/paas/v4 for the standard API; do not silently switch billing endpoints.");
  }

  if (endpoint === "chat" && /\/chat\/completions$/.test(base)) {
    return base;
  }

  if (endpoint === "models" && /\/chat\/completions$/.test(base)) {
    return base.replace(/\/chat\/completions$/, "/models");
  }

  return `${base}/${endpoint === "chat" ? "chat/completions" : "models"}`;
}

function endpointCandidatesFor(baseUrl, endpoint) {
  const base = normalizeBaseUrl(baseUrl);
  const primary = endpointFor(base, endpoint);
  const hasVersionSegment = /\/v\d+(?:\/|$)/.test(base);
  const exactEndpoint = /\/(?:models|chat\/completions)$/.test(base);

  if (hasVersionSegment || exactEndpoint) {
    return [primary];
  }

  const versioned = endpointFor(`${base}/v1`, endpoint);
  return versioned === primary ? [primary] : [primary, versioned];
}

function headersFor(apiKey) {
  const headers = {
    "content-type": "application/json"
  };

  if (apiKey) {
    headers.authorization = `Bearer ${apiKey}`;
  }

  return headers;
}

async function fetchJson(url, options = {}, timeoutMs = 120000, externalSignal = null) {
  const requestStarted = performance.now();
  const controller = new AbortController();
  const timeout = timeoutMs === 0 ? null
    : setTimeout(() => controller.abort(new Error(`Request timed out after ${timeoutMs}ms.`)), timeoutMs);
  const onAbort = () => controller.abort(externalSignal.reason);

  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort(externalSignal.reason);
    } else {
      externalSignal.addEventListener("abort", onAbort, { once: true });
    }
  }

  try {
    const headersStarted = performance.now();
    const transport = timeoutMs === 0 || timeoutMs > 300000
      ? require("./httpWithoutDeadline").httpWithoutDeadline : fetch;
    const { onRequestDispatch, ...fetchOptions } = options;
    controller.signal.throwIfAborted();
    // Synchronous boundary: no await may separate this notification and fetch.
    onRequestDispatch?.();
    const response = await transport(url, {
      ...fetchOptions,
      signal: controller.signal
    });
    const responseHeadersMs = performance.now() - headersStarted;

    const bodyStarted = performance.now();
    const text = await response.text();
    const responseBodyReadMs = performance.now() - bodyStarted;
    const parseStarted = performance.now();
    let body = null;
    if (text) {
      try {
        body = JSON.parse(text);
      } catch (error) {
        body = { raw: text };
      }
    }
    const responseJsonParseMs = performance.now() - parseStarted;

    if (!response.ok) {
      const message = body?.error?.message || body?.message || body?.raw || response.statusText;
      const error = new Error(`LLM request failed (${response.status}): ${message}`);
      error.status = response.status;
      error.body = body;
      error.url = url;
      throw error;
    }

    if (body && typeof body === "object") {
      body.__client_timing = {
        request_started_perf_ms: requestStarted,
        response_finished_perf_ms: performance.now(),
        request_total_ms: Number((performance.now() - requestStarted).toFixed(3)),
        response_headers_ms: Number(responseHeadersMs.toFixed(3)),
        response_body_read_ms: Number(responseBodyReadMs.toFixed(3)),
        response_json_parse_ms: Number(responseJsonParseMs.toFixed(3))
      };
    }
    return body;
  } catch (error) {
    if (!error?.status) {
      const cause = error?.cause;
      const details = [
        `url=${url}`,
        `message=${error?.message || "unknown error"}`,
        cause?.code ? `causeCode=${cause.code}` : "",
        cause?.message ? `causeMessage=${cause.message}` : "",
        error?.name ? `name=${error.name}` : ""
      ].filter(Boolean).join("; ");
      const enriched = new Error(`LLM network request failed: ${details}`);
      enriched.name = error?.name || "LLMNetworkError";
      enriched.cause = error;
      enriched.url = url;
      throw enriched;
    }
    throw error;
  } finally {
    clearTimeout(timeout);
    externalSignal?.removeEventListener("abort", onAbort);
  }
}

async function fetchFirstJson(urls, options = {}, timeoutMs = 120000, externalSignal = null) {
  let lastError = null;

  for (let index = 0; index < urls.length; index += 1) {
    try {
      return await fetchJson(urls[index], options, timeoutMs, externalSignal);
    } catch (error) {
      lastError = error;
      const shouldRetry = index < urls.length - 1 && [401, 403, 404, 405].includes(Number(error.status));
      if (!shouldRetry) {
        throw error;
      }
    }
  }

  throw lastError || new Error("LLM request failed.");
}

async function listModels(config, options = {}) {
  if (usesGlmAnthropic(config.baseUrl, config.modelName)) return [config.modelName];
  const urls = endpointCandidatesFor(config.baseUrl, "models");
  const body = await fetchFirstJson(
    urls,
    {
      method: "GET",
      headers: headersFor(config.apiKey)
    },
    options.timeoutMs || 30000,
    options.signal
  );

  const models = Array.isArray(body?.data) ? body.data : [];
  return models
    .map((model) => (typeof model === "string" ? model : model.id || model.name))
    .filter(Boolean);
}

async function resolveModel(config, options = {}) {

  if (config.modelName && String(config.modelName).trim()) {
    return String(config.modelName).trim();
  }

  const models = await listModels(config, options);
  if (models.length === 0) {
    throw new Error("No models returned by /models and no modelName was provided.");
  }

  return models[0];
}

function textFromValue(value) {
  if (value === undefined || value === null) {
    return "";
  }

  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  if (Array.isArray(value)) {
    return value
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }

        return textFromValue(part?.text ?? part?.content ?? part?.summary ?? part?.output_text ?? part);
      })
      .join("");
  }

  if (typeof value === "object") {
    const nested = value.text ?? value.content ?? value.summary ?? value.output_text ?? value.reasoning_content ?? value.reasoning ?? value.thinking;
    if (nested !== undefined) {
      return textFromValue(nested);
    }

    return JSON.stringify(value);
  }

  return String(value);
}

function isReasoningPart(part) {
  return ["reasoning", "thinking", "reasoning_text", "thought"].includes(String(part?.type || "").toLowerCase());
}

function normalizeContent(content) {
  if (Array.isArray(content)) {
    return content
      .filter((part) => !isReasoningPart(part))
      .map((part) => textFromValue(part?.text ?? part?.content ?? part?.summary ?? part?.output_text ?? part))
      .join("");
  }

  return textFromValue(content);
}

function extractThinking(choice) {
  const message = choice?.message || {};
  const candidates = [
    message.reasoning_content,
    message.reasoningContent,
    message.reasoning,
    message.thinking,
    message.thoughts,
    choice?.reasoning_content,
    choice?.reasoningContent,
    choice?.reasoning,
    choice?.thinking
  ];

  for (const candidate of candidates) {
    const text = textFromValue(candidate).trim();
    if (text) {
      return text;
    }
  }

  if (Array.isArray(message.content)) {
    return message.content
      .filter((part) => isReasoningPart(part))
      .map((part) => textFromValue(part?.text ?? part?.content ?? part))
      .join("")
      .trim();
  }

  return "";
}

function isQwenModel(model) {
  return /qwen/i.test(String(model || ""));
}

function isKimiModel(model) {
  return /kimi/i.test(String(model || ""));
}

function isGlmReasoningModel(model) {
  return /^glm-5(?:[.-]|$)/i.test(String(model || "").trim());
}

function chatTemplateKwargsForModel(model, allowThinking = false) {
  if (isKimiModel(model)) {
    return {
      thinking: Boolean(allowThinking)
    };
  }

  if (isQwenModel(model)) {
    return {
      enable_thinking: Boolean(allowThinking)
    };
  }

  return null;
}

function chatCompletionRequestBody(config, model, messages) {
  const body = {
    model,
    messages,
    temperature: config.temperature ?? 0.2,
    max_tokens: config.maxTokens === undefined ? 800 : config.maxTokens
  };

  const chatTemplateKwargs = chatTemplateKwargsForModel(model, config.allowThinking);
  if (chatTemplateKwargs) {
    body.chat_template_kwargs = chatTemplateKwargs;
  }
  if (/kimi[-_]?k3/i.test(String(model || "")) && config.allowThinking) {
    // K3's native tokenizer reads thinking_effort, and defaults to max when
    // omitted. The generic reasoning_effort field alone does not select it.
    body.chat_template_kwargs.thinking_effort = ["low", "high", "max"].includes(config.reasoningEffort)
      ? config.reasoningEffort
      : "low";
  }
  if (isGlmReasoningModel(model)) {
    body.thinking = { type: config.policyMode === "trivial" && config.agentMode === "high_frequency" ? "disabled" : "enabled" };
    if (body.thinking.type === "enabled") body.reasoning_effort = ["low", "high", "max"].includes(config.reasoningEffort)
      ? config.reasoningEffort
      : "low";
  }
  // GPT-6 uses the OpenAI field too; Brain's configured effort must reach HTTP.
  // Do not send GLM's separate `thinking` extension to this provider.
  if (/^gpt-6(?:[.-]|$)/i.test(String(model || "").trim())
      && typeof config.reasoningEffort === "string" && config.reasoningEffort.trim()) {
    body.reasoning_effort = config.reasoningEffort.trim();
  }

  return body;
}

function qwen35CompactMessages(messages) {
  const compactSystem = [
    "You are a multimodal GUI agent controlling GTA with a PlayStation-style gamepad.",
    "The lower-left minimap shows navigation; yellow markers indicate objectives.",
    "Return exactly one fenced python code block and no prose.",
    "Use only these calls: gamepad_stick(stick=\"left\", x=0, y=1, duration_ms=200), gamepad_trigger(trigger=\"r2\", value=1, duration_ms=200), gamepad_down(button=\"cross\"), gamepad_up(button=\"cross\"), gamepad_press(button=\"cross\", duration_ms=80), gamepad_reset(), wait(ms=100).",
    "Stick x/y and trigger values are normalized floats. If uncertain, output wait(ms=100)."
  ].join(" ");
  const rest = Array.isArray(messages) ? messages.slice() : [];

  if (rest[0]?.role === "system") {
    return [
      {
        role: "system",
        content: compactSystem
      },
      ...rest.slice(1)
    ];
  }

  return [
    {
      role: "system",
      content: compactSystem
    },
    ...rest
  ];
}

function emptyContentError(model, choice, body, retried) {
  if (body?.success === false || body?.error || (body?.code && !body?.choices)) {
    const code = body?.error?.code ?? body?.code ?? "unknown";
    const message = String(body?.error?.message ?? body?.msg ?? body?.message ?? "Provider returned an error envelope").slice(0,500);
    const error = new Error(`LLM provider error: model=${model}; code=${code}; message=${message}`);
    error.providerCode = code;
    error.body = body;
    return error;
  }
  const error = new Error(`LLM response did not contain message content. model=${model}; finishReason=${choice?.finish_reason ?? choice?.finishReason ?? "unknown"}; matchedStop=${choice?.matched_stop ?? "none"}; completionTokens=${body?.usage?.completion_tokens ?? "unknown"}; qwen35CompactRetry=${retried ? "used" : "not_used"}.`);
  error.body = body;
  error.choice = choice || null;
  return error;
}

async function chatCompletion(config, messages, options = {}) {
  const clientStarted = performance.now();
  const model = options.model || config.modelName || (await resolveModel(config, options));
  const responses = usesResponses(config);
  if (responses && (options.sessionId || Object.keys(options.extraBody || {}).length))
    throw new Error("Native Responses uses full stateless messages without Lumine/template extensions.");
  const anthropic = usesGlmAnthropic(config.baseUrl, model);
  if (anthropic && config.policyMode === "trivial" && config.agentMode === "high_frequency") {
    throw new Error("Trivial Policy requires an OpenAI-compatible chat endpoint.");
  }
  const urls = responses ? [responsesEndpoint(config.baseUrl)] : anthropic ? [glmAnthropicEndpoint(config.baseUrl)] : endpointCandidatesFor(config.baseUrl, "chat");
  const requestBuildStarted = performance.now();
  const requestBody = responses ? responsesRequest(config, model, messages) : anthropic ? glmAnthropicRequest(config, model, messages) : chatCompletionRequestBody(config, model, messages);
  Object.assign(requestBody, options.extraBody || {});
  const requestBuildMs = Number((performance.now() - requestBuildStarted).toFixed(3));
  const requestSerializeStarted = performance.now();
  let requestJson = JSON.stringify(requestBody);
  const requestJsonSerializeMs = Number((performance.now() - requestSerializeStarted).toFixed(3));
  let requestBodyBytes = Buffer.byteLength(requestJson, "utf8");
  const requestHeaders = {
    ...headersFor(config.apiKey),
    ...(anthropic ? { "anthropic-version": "2023-06-01", "x-api-key": config.apiKey || "" } : {}),
    ...(!responses && options.sessionMode ? { "x-lumine-session-mode": options.sessionMode } : {}),
    ...(options.sessionId ? {
      "x-lumine-session-mode": "session",
      "x-lumine-session-id": String(options.sessionId),
      "x-lumine-request-id": String(options.requestId || "")
    } : {})
  };
  const submissionTiming = options.beforeSend ? await options.beforeSend(options.signal) : {};
  const sendMutationStarted = performance.now();
  if (options.extraBodyAtSend) {
    const extra = options.extraBodyAtSend();
    const keys = Object.keys(extra || {});
    if (keys.length && keys.every(key => !Object.hasOwn(requestBody, key))) {
      // The current completed-action receipts are small. Append new fields to
      // the pre-serialized body instead of serializing its image base64 again
      // after the gate. For general overriding options retain exact Object.assign
      // semantics via the fallback below.
      const suffix = JSON.stringify(extra);
      const separator = requestJson === '{}' ? '' : ',';
      requestJson = requestJson.slice(0, -1) + separator + suffix.slice(1);
      requestBodyBytes += Buffer.byteLength(suffix, 'utf8') - 2 + separator.length;
      Object.assign(requestBody, extra);
    } else if (keys.length) {
      Object.assign(requestBody, extra);
      requestJson = JSON.stringify(requestBody);
      requestBodyBytes = Buffer.byteLength(requestJson, 'utf8');
    }
  }
  const sendMutationMs = performance.now() - sendMutationStarted;
  const fetchResponse = requestBody.stream
    ? (urls, init, timeout, signal) => require('../../cadence/src/policyActionStream').fetchPolicyStream(urls[0], init, timeout, signal, {
        onEvent: options.onPolicyStream, sessionId: options.sessionId, requestId: options.requestId, rendering: config.policyRendering })
    : fetchFirstJson;
  let body = await fetchResponse(
    options.beforeSend ? urls.slice(0, 1) : urls,
    {
      method: "POST",
      headers: requestHeaders,
      body: requestJson,
      onRequestDispatch: options.onRequestDispatch
    },
    options.timeoutMs ?? 120000,
    options.signal
  );

  const anthropicRaw = anthropic ? body : null;
  if (anthropic) body = glmAnthropicResponse(body);
  const responsesTiming = body?.__client_timing;
  const nativeResponse = responses ? responsesResult(body, model) : null;
  if (nativeResponse) body = {
    choices: [{ message: { role: "assistant", content: nativeResponse.content }, finish_reason: nativeResponse.finishReason }],
    usage: nativeResponse.usage, __client_timing: responsesTiming
  };
  let choice = body?.choices?.[0];
  const normalizeResponseContent = (value) => {
    const normalized = normalizeContent(value);
    return options.preserveThinkingTags ? normalized : stripThinkingBlocks(normalized);
  };
  let content = normalizeResponseContent(choice?.message?.content ?? choice?.text);
  let retriedWithCompactPrompt = false;
  let retryRequestBuildMs = 0;
  let retryRequestJsonSerializeMs = 0;
  let retryRequestBodyBytes = 0;
  if (!content && isQwenModel(model) && !options.disableCompactRetry) {
    retriedWithCompactPrompt = true;
    const retryBuildStarted = performance.now();
    const retryBody = chatCompletionRequestBody(config, model, qwen35CompactMessages(messages));
    retryRequestBuildMs = Number((performance.now() - retryBuildStarted).toFixed(3));
    const retrySerializeStarted = performance.now();
    const retryJson = JSON.stringify(retryBody);
    retryRequestJsonSerializeMs = Number((performance.now() - retrySerializeStarted).toFixed(3));
    retryRequestBodyBytes = Buffer.byteLength(retryJson, "utf8");
    body = await fetchFirstJson(
      urls,
      {
        method: "POST",
        headers: requestHeaders,
        body: retryJson
      },
      options.timeoutMs ?? 120000,
      options.signal
    );
    choice = body?.choices?.[0];
    content = normalizeResponseContent(choice?.message?.content ?? choice?.text);
  }

  if (!content) {
    throw emptyContentError(model, choice, body, retriedWithCompactPrompt);
  }

  const responseClientTiming = body?.__client_timing || {};
  if (body && typeof body === "object") {
    // This is an internal transport detail. Keep it in the structured timing
    // field below without leaking it into the raw OpenAI-compatible payload.
    delete body.__client_timing;
  }

  return {
    model,
    content,
    thinking: extractThinking(choice),
    responseMessage: choice?.message || null,
    finishReason: choice?.finish_reason ?? choice?.finishReason ?? null,
    usage: body?.usage || null,
    vllmTiming: body?.vllm_timing || null,
    requestMetrics: body?.request_metrics || null,
    requestPreprocess: body?.request_preprocess || null,
    serverOutputTiming: body?.server_output_timing || null,
    timingEvents: Array.isArray(body?.timing_events) ? body.timing_events : [],
    firstActionDelimiterMs: body?.first_action_delimiter_ms ?? null,
    firstExecutablePackageMs: body?.first_executable_package_ms ?? null,
    firstStateApplyMs: body?.first_state_apply_ms ?? null,
    clientTiming: {
      ...(options.clientTiming || {}),
      ...submissionTiming,
      request_build_ms: requestBuildMs,
      request_json_serialize_ms: requestJsonSerializeMs,
      request_body_bytes: requestBodyBytes,
      execution_receipt_body_update_ms: sendMutationMs,
      retry_request_build_ms: retryRequestBuildMs,
      retry_request_json_serialize_ms: retryRequestJsonSerializeMs,
      retry_request_body_bytes: retryRequestBodyBytes,
      retry_used: retriedWithCompactPrompt,
      client_total_ms: Number((performance.now() - clientStarted).toFixed(3)),
      ...responseClientTiming
    },
    raw: nativeResponse ? { protocol: "responses-v1", request: requestBody, response: nativeResponse.raw }
      : anthropicRaw ? Object.fromEntries(Object.entries(anthropicRaw).filter(([key]) => key !== "__client_timing")) : body
  };
}

async function cancelChatTurn(config, sessionId, requestId) {
  const url = new URL(endpointCandidatesFor(config.baseUrl, 'chat')[0]);
  url.pathname = url.pathname.replace(/\/chat\/completions$/,
    `/lumine/sessions/${encodeURIComponent(sessionId)}/requests/${encodeURIComponent(requestId)}/cancel`);
  const receipt = await fetchJson(url, { method: 'POST', headers: headersFor(config.apiKey) }, 1000);
  if (receipt?.cancelled !== true || receipt.request_id !== requestId || receipt.lumine_session_id !== sessionId
    || receipt.history_cancel_version !== 'latest-unstarted-turn-v1' || typeof receipt.history_sha256 !== 'string')
    throw Error('Deployment did not acknowledge withdrawal of the unstarted turn');
  return receipt;
}

async function checkPolicyServiceHealth(config) {
  const url = new URL(endpointCandidatesFor(config.baseUrl, 'chat')[0]);
  url.pathname = url.pathname.replace(/\/v1\/chat\/completions$/, '/health');
  if (/\/chat\/completions$/.test(url.pathname)) throw Error('Policy health endpoint is not defined');
  const health = await fetchJson(url, { method: 'GET', headers: headersFor(config.apiKey) }, 15000);
  if (health.routes !== 2) throw Error('Dual Policy health verification failed');
  return { routes: health.routes, pipelineRevision: health.pipeline_revision, active: health.active };
}

async function checkPolicyHandoffCapabilities(config) {
  const url = new URL(endpointCandidatesFor(config.baseUrl, 'chat')[0]);
  url.pathname = url.pathname.replace(/\/v1\/chat\/completions$/, '/health');
  if (/\/chat\/completions$/.test(url.pathname)) throw Error('Policy handoff health endpoint is not defined');
  return fetchJson(url, { method: 'GET', headers: headersFor(config.apiKey) }, 15000);
}

async function recoverPolicySession(config, { sessionId, recoveryId, requestId = null, state, window,
  timeoutMs = 120000 }) {
  if (!/^[a-zA-Z0-9_.:-]{1,128}$/.test(sessionId || '') || !/^[a-zA-Z0-9_.:-]{1,128}$/.test(recoveryId || ''))
    throw Error('Invalid Policy history recovery identity');
  const url = new URL(endpointCandidatesFor(config.baseUrl, 'chat')[0]);
  url.pathname = url.pathname.replace(/\/chat\/completions$/, `/lumine/sessions/${encodeURIComponent(sessionId)}/recover`);
  const receipt = await fetchJson(url, { method: 'POST', headers: headersFor(config.apiKey),
    body: JSON.stringify({ recovery_id: recoveryId, request_id: requestId, lumine_state: state, lumine_window: window }) }, timeoutMs);
  return require('./policyHistoryRecovery').validateRecovery(receipt, { sessionId, recoveryId, state, window });
}

async function preparePolicySession(config, { sessionId, state, updateVersion, signal, timeoutMs = 15000 }) {
  if (!/^[a-zA-Z0-9_.:-]{1,128}$/.test(sessionId || '') || !Number.isSafeInteger(updateVersion) || updateVersion < 1)
    throw Error('Invalid standby session identity or update version');
  const body = { model: config.modelName, lumine_state: state, update_version: updateVersion,
    lumine_format_version: 'think_memory_minimap_compact_v1', lumine_history_mode: 'generated_dialogue_v1',
    chat_template_kwargs: { enable_thinking: true },
    lumine_window: { max: config.policyWindowMaxRounds, keep: config.policyWindowKeepRounds } };
  const stateHash = require('node:crypto').createHash('sha256')
    .update(JSON.stringify(Object.fromEntries(Object.entries(state).sort(([a], [b]) => a.localeCompare(b)))))
    .digest('hex');
  const endpoints = endpointCandidatesFor(config.baseUrl, 'chat');
  for (let index = 0; index < endpoints.length; index++) {
    const url = new URL(endpoints[index]);
    url.pathname = url.pathname.replace(/\/chat\/completions$/, `/lumine/sessions/${encodeURIComponent(sessionId)}/prepare`);
    try {
      const receipt = await fetchJson(url, { method: 'POST', headers: headersFor(config.apiKey), body: JSON.stringify(body) }, timeoutMs, signal);
      if (receipt?.ready !== true || receipt.prepare_version !== 'state-prefix-v1' || receipt.session_id !== sessionId
        || receipt.update_version !== updateVersion || receipt.history_turns !== 0 || receipt.image_count !== 0
        || receipt.state_sha256 !== stateHash || !Array.isArray(receipt.routes)
        || receipt.routes.length !== 2 || ![0,1].every(route => receipt.routes.some(r => r.route === route
          && r.prepared === true && /^[a-f0-9]{64}$/.test(r.prefix_sha256 || ''))))
        throw Error('Deployment did not acknowledge both empty-history standby prefixes');
      return receipt;
    } catch (error) {
      if (error.status === 404 && index < endpoints.length - 1) continue;
      throw error;
    }
  }
}

async function resetChatSession(config, sessionId, options = {}) {
  const normalizedSessionId = String(sessionId || "").trim();
  if (!normalizedSessionId) {
    return false;
  }
  const resetUrls = endpointCandidatesFor(config.baseUrl, "chat").map((url) => {
    const resetUrl = new URL(url);
    resetUrl.pathname = resetUrl.pathname.replace(
      /\/chat\/completions$/,
      `/lumine/sessions/${encodeURIComponent(normalizedSessionId)}/reset`
    );
    return resetUrl;
  });
  for (let index = 0; index < resetUrls.length; index += 1) {
    try {
      const body = await fetchJson(
        resetUrls[index],
        { method: "POST", headers: headersFor(config.apiKey) },
        options.timeoutMs || 10000,
        options.signal
      );
      return Boolean(body?.reset);
    } catch (error) {
      if (error.status === 404 && index < resetUrls.length - 1) {
        continue;
      }
      if (error.status === 404 || error.status === 501) {
        return false;
      }
      throw error;
    }
  }
  return false;
}

module.exports = {
  normalizeBaseUrl,
  endpointFor,
  endpointCandidatesFor,
  listModels,
  resolveModel,
  isQwenModel,
  isKimiModel,
  isGlmReasoningModel,
  chatTemplateKwargsForModel,
  chatCompletionRequestBody,
  qwen35CompactMessages,
  chatCompletion,
  resetChatSession,
  cancelChatTurn,
  checkPolicyServiceHealth,
  checkPolicyHandoffCapabilities,
  preparePolicySession,
  recoverPolicySession
};

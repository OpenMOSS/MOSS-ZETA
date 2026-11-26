// Explicit SGLang Brain transport. /v1/chat/completions max_tokens=0 with
// image messages was verified on the actual Kimi service; no local tokenizer
// or guessed native multimodal serialization is used here.
const { randomUUID } = require('node:crypto');
const { endpointFor, chatCompletionRequestBody } = require('./llmClient');
const { hash } = require('./brainPrefillMessages');

function cacheAudit(body) {
  const details = body.sglext?.cached_tokens_details || body.meta_info?.cached_tokens_details || null;
  return { cachedTokens: details && Number.isFinite(details.device)
    ? Number(details.device) + Number(details.host || 0)
    : body.meta_info?.cached_tokens ?? body.usage?.prompt_tokens_details?.cached_tokens ?? null,
  cachedTokensDetails: details, usageCachedTokens: body.usage?.prompt_tokens_details?.cached_tokens ?? null };
}

async function request(config, body, { signal, timeoutMs, requestId = `brain-${randomUUID()}`, fetchImpl = fetch, stream = false } = {}) {
  const started = performance.now(), controller = new AbortController();
  const onAbort = () => controller.abort(signal.reason);
  if (signal?.aborted) controller.abort(signal.reason);
  else signal?.addEventListener('abort', onAbort, { once: true });
  const timer = timeoutMs ? setTimeout(() => controller.abort(Error(`Brain request timed out after ${timeoutMs}ms`)), timeoutMs) : null;
  const serialized = JSON.stringify(body);
  const serializationMs = performance.now() - started;
  const sentAt = performance.now();
  const timing = { request_id: requestId, request_started_perf_ms: sentAt, request_body_bytes: Buffer.byteLength(serialized),
    request_json_serialize_ms: serializationMs, request_started_at: new Date().toISOString() };
  try {
    controller.signal.throwIfAborted();
    const response = await fetchImpl(endpointFor(config.baseUrl, 'chat'), { method: 'POST',
      headers: { 'content-type': 'application/json', 'x-request-id': requestId,
        ...(config.apiKey ? { authorization: `Bearer ${config.apiKey}` } : {}) },
      body: serialized, signal: controller.signal });
    timing.response_headers_ms = performance.now() - sentAt;
    if (!response.ok) {
      const detail = await response.text();
      throw Object.assign(Error(`Brain HTTP ${response.status}: ${detail.slice(0, 1200)}`), { status: response.status });
    }
    if (!stream) {
      const raw = await response.json();
      timing.client_total_ms = performance.now() - started;
      timing.request_total_ms = performance.now() - sentAt;
      return { raw, timing };
    }
    if (!response.body) throw Error('Brain streaming response has no body');
    let buffer = '', content = '', thinking = '', done = false, finishReason = null;
    const decoder = new TextDecoder(), events = [], raw = { choices: [] };
    const accept = block => {
      const data = block.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
      if (!data) return;
      if (data.trim() === '[DONE]') { done = true; return; }
      const event = JSON.parse(data);
      if (event.error) throw Error(`Brain stream error: ${JSON.stringify(event.error)}`);
      const elapsed = performance.now() - sentAt;
      events.push({ elapsedMs: elapsed, event });
      for (const key of ['id', 'model', 'usage', 'sglext', 'metadata']) if (event[key] != null) raw[key] = event[key];
      const choice = event.choices?.[0];
      const delta = choice?.delta || {};
      const thought = delta.reasoning_content || delta.reasoning || '';
      const text = delta.content || '';
      if (text || thought) timing.first_token_ms ??= elapsed;
      if (text) timing.first_content_token_ms ??= elapsed;
      if (thought) timing.first_reasoning_token_ms ??= elapsed;
      content += text;
      thinking += thought;
      if (choice?.finish_reason) finishReason = choice.finish_reason;
    };
    for await (const chunk of response.body) {
      controller.signal.throwIfAborted();
      buffer = (buffer + decoder.decode(chunk, { stream: true })).replace(/\r\n/g, '\n');
      let end;
      while ((end = buffer.indexOf('\n\n')) >= 0) { const block = buffer.slice(0, end); buffer = buffer.slice(end + 2); accept(block); }
    }
    buffer += decoder.decode();
    if (buffer.trim()) accept(buffer);
    controller.signal.throwIfAborted();
    if (!done || !finishReason) throw Error('Brain stream ended before its final completion marker');
    if (!content) throw Error('Brain stream completed without JSON content');
    raw.choices = [{ index: 0, message: { role: 'assistant', content, reasoning_content: thinking }, finish_reason: finishReason }];
    raw.stream_events = events;
    timing.client_total_ms = performance.now() - started;
    timing.request_total_ms = performance.now() - sentAt;
    return { raw, timing };
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
}

class BrainPrefillTransport {
  constructor({ fetchImpl } = {}) { this.fetchImpl = fetchImpl; }

  async prefill(config, messages, { signal, jobId, model } = {}) {
    const body = { ...chatCompletionRequestBody(config, model || config.modelName, messages), max_tokens: 0,
      stream: false, return_cached_tokens_details: true, return_prompt_token_ids: true };
    const { raw, timing } = await request(config, body, { signal, requestId: jobId,
      timeoutMs: config.brainPrefillTimeoutMs ?? 10000, fetchImpl: this.fetchImpl });
    if (raw.usage?.completion_tokens !== 0 || String(raw.choices?.[0]?.message?.content || '') !== '') {
      throw Error('Brain prefill endpoint did not honour max_tokens=0');
    }
    const promptTokenIds = raw.choices?.[0]?.prompt_token_ids || raw.prompt_token_ids || null;
    if (!Array.isArray(promptTokenIds) || !promptTokenIds.length || !promptTokenIds.every(Number.isInteger)) {
      throw Error('Brain prefill did not return actual prompt token IDs');
    }
    return { promptTokenIds, raw, audit: { transport: 'sglang_chat_zero_v1', requestId: timing.request_id,
      backendRequestId: raw.id || null, promptTokens: raw.usage?.prompt_tokens ?? null,
      actualPromptTokenCount: promptTokenIds.length, promptTokenIdsSha256: hash(promptTokenIds),
      completionTokens: 0, discardedOutput: '', ttftMs: null, ...cacheAudit(raw), clientTiming: timing,
      note: 'Zero output tokens. Server chat rendering may include an assistant stub; only the actual common prefix is reusable.' } };
  }

  async generate(config, messages, { signal, model, timeoutMs, requestId } = {}) {
    const body = { ...chatCompletionRequestBody(config, model || config.modelName, messages), stream: true,
      stream_options: { include_usage: true }, return_cached_tokens_details: true };
    const { raw, timing } = await request(config, body, { signal, requestId, timeoutMs: timeoutMs ?? 120000,
      fetchImpl: this.fetchImpl, stream: true });
    const choice = raw.choices[0];
    return { model: raw.model || model || config.modelName, content: choice.message.content,
      thinking: choice.message.reasoning_content || '', finishReason: choice.finish_reason,
      responseMessage: choice.message, usage: raw.usage || null, clientTiming: timing, raw,
      prefillAudit: { requestId: timing.request_id, backendRequestId: raw.id || null,
        ttftMs: timing.first_token_ms ?? null, firstContentTokenMs: timing.first_content_token_ms ?? null,
        firstReasoningTokenMs: timing.first_reasoning_token_ms ?? null, ...cacheAudit(raw),
        actualTokenPrefixMatch: null,
        actualTokenPrefixMatchReason: 'This SGLang chat endpoint rejects streaming with return_prompt_token_ids; exact formal-token prefix requires backend audit.' } };
  }
}

module.exports = { BrainPrefillTransport, cacheAudit };

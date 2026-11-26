const { parseCompactState } = require('./highFrequencyActionParser');

// A token or opening <a> is not an executable state. A state is complete only
// after all four coordinates and the whole button field's terminator arrive.
class PolicyActionStream {
  constructor(prefix = '<think>') { this.text = prefix; this.emitted = 0; this.states = []; }
  push(delta) {
    this.text += delta;
    const match = /^<think>[\s\S]*?<\/think>\s*<a>/.exec(this.text);
    if (!match) return [];
    const body = this.text.slice(match[0].length);
    if (body.length < 8) return [];
    const coordinates = body.slice(0, 8);
    const fields = body.slice(8).split(';');
    const closed = fields.at(-1).includes('</a>');
    const count = closed ? fields.length : fields.length - 1;
    if (count > 6) throw Error('Streaming action has too many states');
    const next = [];
    while (this.emitted < count) {
      const index = this.emitted;
      const raw = fields[index].replace(/<\/a>[\s\S]*$/, '');
      const state = parseCompactState(coordinates, raw, index);
      if (index % 3 && JSON.stringify(state) !== JSON.stringify(this.states[index - 1]))
        throw Error('Streaming compact action violates two ordered 100ms button states');
      this.states.push(state); this.emitted++;
      next.push({ index, state, raw, coordinates, brainCall: match[0].includes('<|call_brain|>') });
    }
    return next;
  }
}

async function fetchPolicyStream(url, options, timeoutMs, externalSignal, { onEvent, sessionId, requestId, rendering }) {
  await onEvent?.({ type: "request", receivedPerfMs: performance.now() });
  const started = performance.now();
  const controller = new AbortController();
  const abort = () => controller.abort(externalSignal.reason);
  const timer = timeoutMs ? setTimeout(() => controller.abort(Error('Policy stream timeout')), timeoutMs) : null;
  if (externalSignal?.aborted) abort(); else externalSignal?.addEventListener('abort', abort, { once: true });
  const timing = { request_started_perf_ms: started, stream_events: [], first_token_ms: null, first_executable_state_ms: null };
  let parser, result, identity, brainMarkerSeen = false;
  try {
    const { onRequestDispatch, ...fetchOptions } = options;
    controller.signal.throwIfAborted();
    // The request event above may await native arming. Only this final boundary
    // makes remote delivery possible; a pre-arm cancellation stays local.
    onRequestDispatch?.();
    const response = await fetch(url, { ...fetchOptions, signal: controller.signal });
    timing.response_headers_ms = performance.now() - started;
    if (!response.ok) throw Object.assign(Error(`Policy stream HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`), { status: response.status });
    if (!response.headers.get('content-type')?.includes('text/event-stream'))
      throw Error('Policy stream requires an SSE deployment; refusing buffered/replayed input');
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let pending = '';
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      let end;
      while ((end = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, end).replace(/\r$/, ''); pending = pending.slice(end + 1);
        if (!line.startsWith('data: ')) continue;
        const event = JSON.parse(line.slice(6)), now = performance.now();
        if (event.lumine_session_id !== sessionId || event.lumine_request_id !== requestId)
          throw Error('Policy stream correlation mismatch');
        if (event.type === 'error') throw Error(`Policy stream failed: ${event.detail || event.error}`);
        if (event.type === 'start') {
          if (parser) throw Error('Repeated stream start');
          const r = event.lumine_rendering;
          if (r?.version !== 'think-memory-minimap-compact-v1' || r.response_format !== 'full'
            || (rendering?.template_sha256 && r.template_sha256 !== rendering.template_sha256))
            throw Error('Policy stream rendering identity mismatch');
          identity = JSON.stringify(r);
          parser = new PolicyActionStream(r.generation_prefix);
          timing.route = event.route; timing.worker_id = event.worker_id;
          onEvent?.({ ...event, receivedPerfMs: now });
        } else if (event.type === 'delta') {
          if (!parser || result) throw Error('Out-of-order Policy stream');
          if (timing.first_token_ms == null && event.token_ids?.length) timing.first_token_ms = now - started;
          const states = parser.push(event.text || '');
          if (!brainMarkerSeen && parser.text.includes('<|call_brain|>')) {
            brainMarkerSeen = true; timing.brain_call_marker_ms = now-started;
            onEvent?.({ type: 'brain_marker', receivedPerfMs: now, elapsedMs: now-started });
          }
          timing.stream_events.push({ received_ms: now - started, worker_elapsed_ms: event.worker_elapsed_ms,
            text: event.text, token_ids: event.token_ids, state_indices: states.map(s => s.index) });
          for (const state of states) {
            if (!state.brainCall && timing.first_executable_state_ms == null) {
              timing.first_executable_state_ms = now - started;
              timing.first_executable_perf_ms = now;
              timing.first_executable_state = state;
            }
            onEvent?.({ type: 'state', ...state, receivedPerfMs: now, elapsedMs: now - started });
          }
        } else if (event.type === 'result') {
          if (!parser || result || event.status !== 200 || JSON.stringify(event.result?.lumine_rendering) !== identity)
            throw Error('Invalid Policy stream result');
          result = event.result;
        } else throw Error(`Unknown Policy stream event: ${event.type}`);
      }
      if (done) break;
    }
    if (!result) throw Error('Policy stream ended before validated result');
    timing.response_finished_perf_ms = performance.now();
    timing.request_total_ms = timing.response_finished_perf_ms - started;
    result.__client_timing = timing;
    return result;
  } catch (error) {
    error.streamDiagnostic = { rawGeneratedContent: parser?.text || '', timing, result: result || null };
    onEvent?.({ type: 'stream_error', receivedPerfMs: performance.now(), message: error.message });
    throw error;
  } finally {
    clearTimeout(timer); externalSignal?.removeEventListener('abort', abort);
    controller.abort();
  }
}
module.exports = { PolicyActionStream, fetchPolicyStream };

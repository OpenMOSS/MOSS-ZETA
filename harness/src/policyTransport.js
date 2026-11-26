const { isTrivialPolicy } = require('./trivialPolicy');
const { randomUUID } = require('node:crypto');
const { resetChatSession } = require('./llmClient');

class PolicyTransport {
  constructor({resetSession=resetChatSession}={}) { this.resetSession=resetSession;this.releaseJobs=new Set();this.retired = []; this.effectiveMode = 'stateless'; this.reset(); }
  reset() {
    if (this.sent && this.config) this.retired.push({ id: this.sessionId, config: this.config });
    this.executionReceipts = new Map();
    this.sessionId = `policy-${randomUUID()}`; this.initialized = false; this.sent = false; this.dispatched = false; this.renderIdentity = null;
  }
  async releaseRetired() {
    const job=this.releaseRetiredBatch();this.releaseJobs.add(job);
    try {return await job;} finally {this.releaseJobs.delete(job);}
  }
  async waitForRelease() {
    await Promise.allSettled([...this.releaseJobs]);
    return {pending:this.retired.map(item=>item.id),inFlight:this.releaseJobs.size};
  }
  async releaseRetiredBatch() {
    const pending = this.retired.splice(0);
    for (const item of pending) {
      let released = false;
      for (let attempt = 0; attempt < 4; attempt++) {
        try {
          if(!await this.resetSession(item.config,item.id))throw Error('Policy session release was not acknowledged');
          released = true; break;
        }
        catch (error) {
          if (attempt < 3) await new Promise(resolve => setTimeout(resolve, 250 * 2 ** attempt));
        }
      }
      if (!released) this.retired.push(item);
    }
  }
  acknowledgeExecution(contextId, receipt) {
    if (!this.config?.policyExecutionAcknowledgements || contextId !== this.sessionId) return false;
    const previous = this.executionReceipts.get(receipt.request_id);
    if (previous && JSON.stringify(previous) !== JSON.stringify(receipt)) throw Error('Conflicting execution receipts');
    this.executionReceipts.set(receipt.request_id, Object.freeze({ ...receipt }));
    return true;
  }
  prepare(config, messages, requestId = randomUUID()) {
    this.config = config;
    if (isTrivialPolicy(config)) {
      this.effectiveMode = config.policyTransport === 'lumine_session' ? 'lumine_session' : 'stateless';
      const extraBody = { stream: false, chat_template_kwargs: { enable_thinking: false }, lumine_format_version: 'action_only_v1' };
      if (this.effectiveMode === 'stateless') return { messages, requestId, options: { extraBody }, contextId: this.sessionId };
      const user = messages.at(-1);
      if (user?.role !== 'user' || !Array.isArray(user.content) || user.content.filter(p => p.type === 'image_url').length !== 1)
        throw new Error('Trivial session requires one current screenshot and OCR snapshot.');
      this.sent = true;
      return { messages: [user], requestId, contextId: this.sessionId,
        options: { sessionId: this.sessionId, requestId,
          extraBody: { ...extraBody, lumine_ocr_history_reset_ms: config.trivialOcrHistoryResetMs ?? 10000 } } };
    }
    this.effectiveMode = config.policyTransport === 'lumine_session'
      ? 'lumine_session' : 'stateless';
    const extraBody = config.apiFormat !== 'responses' && (this.effectiveMode === 'lumine_session' || config.remoteServiceType === 'lumine' || config.policyRendering?.version === 'compact-think-v1')
      ? { chat_template_kwargs: { enable_thinking: Boolean(config.allowThinking) } } : {};
    if (['think-memory-minimap-v1', 'think-memory-minimap-compact-v1'].includes(config.policyRendering?.version)) {
      if (!config.allowThinking || this.effectiveMode !== 'lumine_session')
        throw Error('think-memory-minimap-v1 requires thinking and lumine_session');
      extraBody.lumine_format_version = config.policyRendering.version === 'think-memory-minimap-compact-v1'
        ? 'think_memory_minimap_compact_v1' : 'think_memory_minimap_v1';
    }
    if (this.effectiveMode === 'stateless') return { messages, requestId, options: { sessionMode: 'stateless', extraBody }, contextId: this.sessionId };
    const user = messages.at(-1);
    if (user?.role !== 'user' || !Array.isArray(user.content) || user.content.filter(p => p.type === 'image_url').length !== 1)
      throw new Error('Lumine v2 requires exactly one image in the current user.');
    const system = messages[0]?.content;
    const match = typeof system === 'string' && system.match(/\n\n当前有效目标：([\s\S]*?)\n\nMemory：([\s\S]*?)\n\n模型思考：([\s\S]*)$/);
    if (!match) throw new Error('Missing canonical Policy state fields.');
    const state = { instruction: ['think-memory-minimap-v1', 'think-memory-minimap-compact-v1'].includes(config.policyRendering?.version) ? match[1] : (match[1] === '暂无明确的阶段目标，需要根据后续观测确认。' ? null : match[1]),
      // New checkpoint's backend inserts literal state values into the training template.
      // Preserve the rendered empty-Memory marker; compact-think keeps its old wire format.
      memory: ['think-memory-minimap-v1', 'think-memory-minimap-compact-v1'].includes(config.policyRendering?.version) ? match[2] : (match[2] === '暂无。' ? '' : match[2]), model_thought: match[3] };
    this.sent = true;
    const prepared = { messages: [user], requestId, contextId: this.sessionId,
      options: { sessionId: this.sessionId, requestId,
        extraBody: { ...extraBody, ...(this.initialized ? {} : { lumine_state: state }) } } };
    if (config.policyStreaming) Object.assign(prepared.options.extraBody, {
      stream: true, lumine_history_mode: 'generated_dialogue_v1', lumine_stream_version: 'policy-state-stream-v1' });
    if (config.policyStreaming && config.policyRendering?.version === 'think-memory-minimap-compact-v1'
      && Number.isInteger(config.policyWindowMaxRounds) && Number.isInteger(config.policyWindowKeepRounds)) {
      prepared.options.extraBody.lumine_window = { max: config.policyWindowMaxRounds, keep: config.policyWindowKeepRounds };
    }
    if (config.policyAwaitNextPrepare === true) prepared.options.extraBody.await_next_prepare = true;
    if (config.policyUpdateVersion != null) prepared.options.extraBody.update_version = config.policyUpdateVersion;
    if (config.policyExecutionAcknowledgements) {
      prepared.options.extraBody.lumine_execution_mode = 'completed_ack_v1';
      const receiptQueue = this.executionReceipts;
      prepared.options.extraBodyAtSend = () => {
        if (prepared.contextId !== this.sessionId) throw new Error('Policy context changed before submission');
        // Freeze once at actual send, after the submission gate. Retry identical input.
        prepared.executionReceipts ||= [...receiptQueue.values()].map(r => ({ ...r }));
        return { lumine_execution_receipts: prepared.executionReceipts };
      };
    }
    return prepared;
  }
  accept(prepared, response) {
    const wire = response.content;
    if (isTrivialPolicy(this.config)) {
      if (prepared.contextId !== this.sessionId) throw new Error('Stale Policy response.');
      const advertised = response.raw?.lumine_rendering;
      if (advertised && (advertised.version !== 'action-only-v1' || advertised.response_format !== 'full' || advertised.generation_prefix !== ''))
        throw new Error('Trivial Policy requires the action-only-v1 rendering contract.');
      const configured = this.config.policyRendering;
      if (configured?.template_sha256 && advertised?.template_sha256 !== configured.template_sha256)
        throw new Error('Trivial rendering template identity mismatch.');
      const identity = advertised ? JSON.stringify(advertised) : null;
      if (this.renderIdentity && identity !== this.renderIdentity) throw new Error('Trivial rendering changed within a session.');
      if (this.effectiveMode === 'lumine_session') {
        if (!advertised || response.raw?.lumine_session_id !== prepared.contextId || response.raw?.lumine_request_id !== prepared.requestId)
          throw new Error('Trivial session rendering or correlation IDs missing or mismatched.');
        const interval = prepared.options.extraBody.lumine_ocr_history_reset_ms;
        const history = response.raw?.ocr_history;
        if (interval && (history?.version !== 'periodic-archive-v1' || history.interval_ms !== interval
          || typeof history.reset !== 'boolean' || !Number.isInteger(history.epoch) || history.epoch < 0))
          throw new Error('Trivial session did not acknowledge the OCR history reset contract; deploy the matching backend.');
        this.initialized = true;
      }
      response.deploymentConnected = this.renderIdentity === null;
      response.wireContent = response.content;
      response.rendering = advertised || null;
      response.generation = response.raw?.lumine_outputs?.[0] || null;
      this.renderIdentity = identity;
      return response;
    }
    const advertised = response.raw?.lumine_rendering;
    const configured = this.config.policyRendering || {};
    const rendering = advertised || configured;
    const format = rendering.response_format ?? 'full';
    if (format !== 'full') throw new Error('Policy requires complete assistant responses; select the matching deployment protocol.');
    if (advertised && advertised.version !== (configured.version || 'compact-think-v1'))
      throw new Error('Policy rendering version differs from the configured checkpoint contract.');
    if (this.effectiveMode === 'lumine_session' && !advertised)
      throw new Error('Session response is missing its rendering identity.');
    if (advertised && configured.template_sha256 && configured.template_sha256 !== advertised.template_sha256)
      throw new Error('Policy rendering template identity mismatch.');
    const identity = advertised ? JSON.stringify(advertised) : null;
    if (this.renderIdentity && identity && this.renderIdentity !== identity)
      throw new Error('Policy rendering changed within a session; create a new session.');
    if (prepared.contextId !== this.sessionId) throw new Error('Stale Policy session response.');
    if (prepared.options.extraBody.update_version != null
      && response.raw?.lumine_update_version !== prepared.options.extraBody.update_version)
      throw new Error('Policy response update version does not match the prepared context.');
    if (this.effectiveMode === 'lumine_session') {
      if (response.raw?.lumine_session_id !== prepared.contextId || response.raw?.lumine_request_id !== prepared.requestId)
        throw new Error('Policy response correlation IDs missing or mismatched.');
      this.initialized = true;
    }
    if (this.config.policyExecutionAcknowledgements) {
      const history = response.raw?.execution_history;
      if (history?.mode !== 'completed_ack_v1' || !Array.isArray(history.applied_receipts))
        throw Error('Deployment did not acknowledge completed-action history contract');
      for (const sent of prepared.executionReceipts || []) {
        if (!history.applied_receipts.some(r => r.request_id === sent.request_id && r.status === sent.status))
          throw Error('Missing execution receipt acknowledgement');
        if (JSON.stringify(this.executionReceipts.get(sent.request_id)) === JSON.stringify(sent))
          this.executionReceipts.delete(sent.request_id);
      }
    }
    if (this.config.policyStreaming && response.raw?.history_mode !== 'generated_dialogue_v1')
      throw Error('Deployment did not acknowledge generated dialogue history');
    const window = prepared.options.extraBody.lumine_window;
    if (window && (response.raw?.window_max !== window.max || response.raw?.window_keep !== window.keep))
      throw Error('Deployment did not acknowledge the configured history window');
    response.deploymentConnected = this.renderIdentity === null;
    response.wireContent = wire;
    response.rendering = advertised || { ...configured, response_format: 'full' };
    response.generation = response.raw?.lumine_outputs?.[0] || null;
    response.content = wire;
    this.renderIdentity = identity;

    return response;
  }
}
module.exports = { PolicyTransport };

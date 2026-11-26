const { randomUUID, createHash } = require('node:crypto');
const { PolicyTransport } = require('./policyTransport');
const ctx = require('./policyHandoffContext');
const clone = value => JSON.parse(JSON.stringify(value));
function freeze(value) { if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); } return value; }
const id = () => `policy-${randomUUID()}`;
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const protocolError = message => Object.assign(Error(message), { nonRetryablePolicyProtocol: true });

// One logical transcript owned by the harness; remote sessions are an internal
// target-specific transport detail. No history export/plan in the normal path.
class PolicySessionHandoffTransport {
  constructor(options = {}) {
    this.releases = new PolicyTransport(options);
    this.target = 'small'; this.epoch = 0; this.completedTurns = 0;
    this.accepted = new Map(); this.gptId = id(); this.small = this.newSlot();
  }
  newSlot() { return { id: id(), initialized: false, sent: false, dispatched: false,
    state: null, config: null, rendering: null, needsImport: false }; }
  get sessionId() { return this.target === 'small' ? this.small.id : this.gptId; }
  get effectiveMode() { return this.target === 'small' ? 'lumine_session' : 'stateless'; }
  get initialized() { return this.target === 'small' && this.small.initialized; }
  get sent() { return this.target === 'small' && this.small.sent; }
  set sent(value) { if (this.target === 'small') this.small.sent = value; }
  get dispatched() { return this.target === 'small' && this.small.dispatched; }
  set dispatched(value) { if (this.target === 'small') this.small.dispatched = value; }
  get renderIdentity() { return this.small.rendering; }
  retireSmall() {
    if (this.small.sent && this.small.config && !this.releases.retired.some(r => r.id === this.small.id))
      this.releases.retired.push({ id: this.small.id, config: this.small.config });
  }
  reset() {
    this.retireSmall(); this.small = this.newSlot(); this.gptId = id();
    this.completedTurns = 0; this.accepted.clear(); this.epoch++;
  }
  invalidateSmall() {
    const previous = this.small.id;
    this.retireSmall(); this.small = this.newSlot(); this.small.needsImport = true;
    this.epoch++;
    return { previous, next: this.small.id, completedTurns: this.completedTurns };
  }
  selectTarget(target) {
    if (!['small', 'gpt'].includes(target)) throw Error('Invalid Policy handoff target');
    if (target !== this.target) {
      if (target === 'small') this.small.needsImport = true;
      this.target = target; this.epoch++;
    }
  }
  releaseRetired() { return this.releases.releaseRetired(); }
  waitForRelease() { return this.releases.waitForRelease(); }
  acknowledgeExecution() { return false; }
  snapshot() { return { version: ctx.VERSION, target: this.target, contextId: this.sessionId,
    smallSessionId: this.small.id, completedTurns: this.completedTurns, epoch: this.epoch,
    smallInitialized: this.small.initialized, importPending: this.small.needsImport }; }

  prepare(config, logicalMessages, requestId = randomUUID()) {
    const target = config.policyHandoffTarget;
    if (target !== this.target) throw Error('Policy adapter target changed outside the handoff boundary');
    if (config.policyExecutionAcknowledgements || config.policyStreaming) throw Error('Session handoff requires non-streamed generated dialogue history');
    const messages = clone(logicalMessages), pairs = ctx.historyPairs(messages);
    if (pairs > this.completedTurns) throw Error('Logical history exceeds the committed global turn');
    const state = ctx.stateFromSystem(messages[0]?.content);
    this.config = config;
    let wireMessages, options, importing = false, replacing = false;
    if (target === 'small') {
      if (config.policyTransport !== 'lumine_session') throw Error('Small target requires lumine_session');
      if (this.small.state && JSON.stringify(state) !== JSON.stringify(this.small.state)) {
        this.invalidateSmall(); this.small.needsImport = true;
      }
      const slot = this.small;
      importing = slot.needsImport || (!slot.initialized && (pairs > 0 || this.completedTurns > 0));
      replacing = importing && slot.initialized;
      const all = ctx.nativeMessages(messages);
      wireMessages = importing ? all : [all.at(-1)];
      const extraBody = { stream: false, chat_template_kwargs: { enable_thinking: true },
        lumine_format_version: ctx.FORMAT, lumine_history_mode: 'generated_dialogue_v1',
        lumine_window: { max: 16, keep: 8 },
        ...((importing || !slot.initialized) ? { lumine_state: clone(state) } : {}),
        ...(importing ? { lumine_history_turn: this.completedTurns } : {}) };
      slot.sent = true; slot.config = config;
      options = { sessionId: slot.id, requestId, extraBody };
    } else {
      if (config.policyTransport !== 'stateless') throw Error('GPT target requires full-context requests');
      wireMessages = messages; options = { sessionMode: 'stateless', extraBody: {} };
    }
    const prepared = { contextId: this.sessionId, requestId, target, epoch: this.epoch,
      completeTurn: this.completedTurns, pairs, state, importing, replacing,
      messages: freeze(wireMessages), options: freeze(options),
      logicalMessagesSha256: hash(messages), wireMessagesSha256: hash(wireMessages) };
    return Object.freeze(prepared);
  }

  accept(prepared, response) {
    if (prepared.epoch !== this.epoch || prepared.target !== this.target || prepared.contextId !== this.sessionId)
      throw protocolError('Stale Policy handoff response');
    const raw = response.raw || {}, generation = raw.lumine_outputs?.[0];
    if (['length', 'max_tokens'].includes(response.finishReason) || generation?.truncated || response.truncated)
      throw protocolError('Truncated response is not committed handoff history');
    ctx.validateAssistant(response.content);
    // Responses preserves the provider body under raw.response; llmClient.model
    // alone is the requested ID and cannot prove the provider did not reroute.
    if ((raw.response?.model || raw.model || response.model) !== this.config.modelName)
      throw protocolError('Policy model identity changed');
    const connected = this.target === 'small' && !this.small.initialized;
    if (this.target === 'small') {
      if (raw.committed !== true || raw.lumine_session_id !== prepared.contextId || raw.lumine_request_id !== prepared.requestId)
        throw protocolError('Small Policy did not acknowledge committed response and correlation IDs');
      if (raw.history_mode !== 'generated_dialogue_v1' || raw.window_max !== 16 || raw.window_keep !== 8
        || raw.turn !== prepared.completeTurn + 1)
        throw protocolError('Small Policy generated history/window/global turn mismatch');
      const rendering = raw.lumine_rendering;
      if (!rendering || rendering.version !== ctx.RENDERING || rendering.response_format !== 'full'
        || generation?.protocol_valid !== true || generation?.full_content !== response.content)
        throw protocolError('Small Policy output does not match the frozen full-response protocol');
      const identity = JSON.stringify(rendering);
      if (this.small.rendering && identity !== this.small.rendering) throw protocolError('Small rendering changed within its session');
      if (prepared.importing) {
        const h = raw.history_import;
        if (!h || h.version !== ctx.TRANSFER || h.turn !== prepared.completeTurn
          || h.supplied_history_turns !== prepared.pairs || h.retained_history_turns !== prepared.pairs
          || !/^[a-f0-9]{64}$/.test(h.history_sha256 || '')
          || (prepared.replacing && (h.operation !== 'replace' || h.replace_version !== ctx.REPLACE)))
          throw protocolError('Small Policy did not acknowledge authoritative history import/replacement');
      }
      this.small.rendering = identity; this.small.state = clone(prepared.state);
      this.small.initialized = true; this.small.needsImport = false;
    }
    const key = prepared.requestId, contentHash = hash(response.content);
    if (this.accepted.has(key)) {
      if (this.accepted.get(key) !== contentHash) throw protocolError('Conflicting replayed Policy response');
    } else {
      if (prepared.completeTurn !== this.completedTurns) throw protocolError('Out-of-order Policy history commit');
      this.completedTurns++;
      this.accepted.set(key, contentHash);
      if (this.accepted.size > 256) this.accepted.delete(this.accepted.keys().next().value);
    }
    response.wireContent = response.content;
    response.rendering = raw.lumine_rendering || { version: ctx.RENDERING, response_format: 'full' };
    response.generation = generation || null; response.deploymentConnected = connected;
    response.policyHandoff = { ...this.snapshot(), requestId: prepared.requestId,
      imported: prepared.importing, replaced: prepared.replacing, suppliedHistoryTurns: prepared.pairs,
      logicalMessagesSha256: prepared.logicalMessagesSha256, wireMessagesSha256: prepared.wireMessagesSha256,
      commitSource: this.target === 'small' ? 'deployment_committed_true' : 'harness_frozen_protocol_validation' };
    return response;
  }
}
module.exports = { PolicySessionHandoffTransport };

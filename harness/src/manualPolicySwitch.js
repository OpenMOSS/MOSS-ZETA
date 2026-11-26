const fs = require('node:fs/promises');
const path = require('node:path');
const { randomUUID } = require('node:crypto');

const TARGET_FIELDS = ['baseUrl', 'apiKey', 'modelName', 'apiFormat', 'reasoningEffort', 'allowThinking', 'maxTokens', 'temperature', 'policyTransport'];
function settings(value = {}) {
  if (!value.enabled) return { enabled: false };
  const targets = {};
  for (const name of ['small', 'gpt']) {
    const raw = value.targets?.[name];
    if (!raw?.baseUrl || !raw.modelName) throw Error(`Manual Policy target ${name} requires endpoint and model`);
    if (Object.keys(raw).some(k => !TARGET_FIELDS.includes(k))) throw Error(`Unknown Manual Policy target field: ${name}`);
    const u = new URL(raw.baseUrl);
    if (!['http:', 'https:'].includes(u.protocol) || u.username || u.password) throw Error('Invalid Policy target URL');
    if (!['chat_completions', 'responses'].includes(raw.apiFormat || 'chat_completions')) throw Error('Unknown Policy target API format');
    if (raw.policyTransport && !['lumine_session', 'stateless'].includes(raw.policyTransport)) throw Error('Unknown Policy target transport');
    targets[name] = structuredClone(raw);
  }
  const initialTarget = value.initialTarget || 'small';
  if (!targets[initialTarget]) throw Error('Invalid initial Policy target');
  return { enabled: true, targets, initialTarget, controlFile: String(value.controlFile || ''), pauseDuringSwitch: value.pauseDuringSwitch !== false,
    imagePython: String(value.imagePython || (process.platform === 'win32' ? 'python' : 'python3')) };
}

// Logical messages stay with the harness. Target-specific adapters either append
// one native turn or send/import this same complete transcript.
class ManualPolicySwitch {
  constructor(agent) {
    this.agent = agent;
    this.settings = settings(agent.defaultConfig.policyManualSwitch);
    if (!this.settings.enabled) throw Error('Manual Policy switching is disabled');
    this.sessionBacked = require('./policyHandoffContext').enabled(agent.defaultConfig);
    if (agent.defaultConfig.policyStreaming || agent.defaultConfig.policyHotSwapEnabled
      || (!this.sessionBacked && agent.defaultConfig.policyTransport !== 'stateless') || agent.defaultConfig.policyMode === 'trivial')
      throw Error('Manual Policy switching requires non-streamed brain_policy with harness context ownership');
    if (this.sessionBacked) {
      const c = agent.defaultConfig;
      if (c.policyRendering?.version !== require('./policyHandoffContext').RENDERING || !c.allowThinking
        || c.policyWindowMaxRounds !== 16 || c.policyWindowKeepRounds !== 8 || !c.policyIncludeMinimap
        || c.policyExecutionAcknowledgements || Math.abs(c.hfPackageDurationMs - 200) > 0.001
        || Math.abs(c.hfChunkDurationMs * 6 - 200) > 0.001
        || Object.values(this.settings.targets).some(target => target.allowThinking === false))
        throw Error('Session/GPT handoff requires the frozen h16/k8 compact profile, paired images and six chunks totaling 200ms');
    }
    this.effective = this.settings.initialTarget;
    this.revision = 0;
    this.pending = null;
    this.phase = 'ready';
    this.stage = null;
    this.lastError = null;
    this.failedTarget = null;
    this.pollAt = 0;
    this.file = this.settings.controlFile || path.join(agent.sessionCaptureRoot, 'policy-control.json');
  }
  activeConfig() {
    return { ...this.agent.defaultConfig, ...this.settings.targets[this.effective],
      policyTransport: this.sessionBacked && this.effective === 'small' ? 'lumine_session' : 'stateless',
      policyStreaming: false, policyHotSwapEnabled: false, policyExecutionAcknowledgements: false,
      remoteServiceType: this.sessionBacked && this.effective === 'small' ? 'lumine' : 'openai', policyRequestTimeoutMs: 0,
      ...(this.sessionBacked ? { policyHandoffTarget: this.effective } : {}) };
  }
  status() { const desired = this.pending?.target || this.failedTarget || this.effective;
    return { effective: this.effective, desired,
    effectiveModel: this.settings.targets[this.effective].modelName,
    desiredModel: this.settings.targets[desired].modelName,
    targetModels: Object.fromEntries(Object.entries(this.settings.targets).map(([key, target]) => [key, target.modelName])),
    revision: this.revision, phase: this.phase === 'ready' && this.pending ? 'waiting' : this.phase,
    stage: this.stage, lastError: this.lastError, controlFile: this.file,
    ...(this.sessionBacked ? { logicalContext: this.agent.frameworkRuntime?.policyActor?.transport.snapshot?.() } : {}) }; }
  async initialize() {
    if (this.sessionBacked) {
      const c = { ...this.agent.defaultConfig, ...this.settings.targets.small };
      const health = await (this.agent.checkPolicyHandoffCapabilities || require('./llmClient').checkPolicyHandoffCapabilities)(c);
      const protocol = require('./policyHandoffContext');
      if (health.history_transfer_version !== protocol.TRANSFER || health.history_replace_version !== protocol.REPLACE)
        throw Error('Small deployment has not loaded the history import/replacement candidate');
      this.agent.frameworkRuntime.policyActor.transport = new (require('./policySessionHandoffTransport').PolicySessionHandoffTransport)();
      this.agent.frameworkRuntime.policyActor.transport.selectTarget(this.effective);
      await this.agent.recordHistoryEvent('policy_handoff_capabilities', { historyTransferVersion: health.history_transfer_version,
        historyReplaceVersion: health.history_replace_version });
    }
    await fs.mkdir(path.dirname(this.file), { recursive: true, mode: 0o700 });
    try {
      const prior = JSON.parse(await fs.readFile(this.file, 'utf8'));
      // A new episode never replays a previous episode's pending command.
      if (Number.isSafeInteger(prior.revision) && prior.revision >= 0) this.revision = prior.revision;
    } catch (e) { if (e.code !== 'ENOENT') throw e; }
    await this.write({ revision: this.revision, target: this.effective });
  }
  async write(command) {
    const tmp = `${this.file}.${randomUUID()}.tmp`;
    try { await fs.writeFile(tmp, JSON.stringify(command) + '\n', { mode: 0o600 }); await fs.rename(tmp, this.file); }
    finally { await fs.rm(tmp, { force: true }); }
  }
  validate(command) {
    if (!command || Object.keys(command).sort().join(',') !== 'revision,target'
      || !Number.isSafeInteger(command.revision) || command.revision < 0 || !this.settings.targets[command.target])
      throw Error('Policy control requires a nonnegative revision and target small/gpt');
    return command;
  }
  async request(target) {
    const command = this.validate({ revision: Math.max(this.revision, this.pending?.revision || 0) + 1, target });
    await this.write(command);
    this.pending = command;
    this.lastError = null;
    this.failedTarget = null;
    if (this.phase === 'failed') this.phase = 'ready';
    return this.status();
  }
  async poll() {
    if (Date.now() < this.pollAt) return;
    this.pollAt = Date.now() + 250;
    try {
      const command = this.validate(JSON.parse(await fs.readFile(this.file, 'utf8')));
      if (command.revision > Math.max(this.revision, this.pending?.revision || 0)) {
        this.pending = command; this.lastError = null; this.failedTarget = null;
        if (this.phase === 'failed') this.phase = 'ready';
      }
    } catch (error) {
      if (this.controlError !== error.message) {
        this.controlError = error.message;
        await this.agent.recordHistoryEvent('policy_control_rejected', { message: error.message });
      }
    }
  }
  async applyPending() {
    await this.poll();
    const command = this.pending;
    const a = this.agent;
    if (!command || a.stopRequested || a.paused || a.taskTransition || a.recoveryToolRunning || a.recoveryAwaitingBrain
      || a.brainCycle || a.pendingHighFrequencyBarrier || a.pendingAutomaticBrainFallback || a.frameworkRuntime?.brainRuntime.scheduler.inferencePromise
      || a.frameworkRuntime?.brainRuntime.scheduler.pending) return false;
    if (command.target === this.effective) { this.revision = command.revision; this.pending = null; this.stage = null; return false; }
    this.phase = 'switching';
    const previous = this.effective;
    let committed = false;
    try {
      await a.changePolicyTarget(command, () => { this.effective = command.target; committed = true; });
      this.revision = command.revision;
      this.phase = 'ready';
      this.stage = null;
      this.lastError = null;
      await a.recordHistoryEvent('policy_target_changed', { revision: command.revision, from: previous,
        to: this.effective, model: this.activeConfig().modelName, historyMessages: a.messages.length,
        contextId: a.frameworkRuntime.policyActor.transport.sessionId, contextOwner: 'harness' });
    } catch (error) {
      if (error.handoffDeferred) { this.phase = 'ready'; this.stage = 'waiting_brain'; return true; }
      if (!committed) this.effective = previous;
      this.revision = command.revision;
      this.phase = 'failed';
      this.stage = null;
      this.failedTarget = command.target;
      this.lastError = error.message;
      a.paused = true;
      await a.recordHistoryEvent('policy_target_change_failed', { revision: command.revision,
        from: previous, to: command.target, message: error.message, remainsPaused: true });
      if (!a.stopRequested) await a.pauseGame('policy_target_change_failed');
    } finally { if (this.pending === command && this.phase !== 'ready') this.pending = null;
      else if (this.pending === command && this.revision === command.revision) this.pending = null; }
    return true;
  }

  async recoverSession() {
    const recovery = this.agent.frameworkRuntime.policyActor.transport.invalidateSmall();
    this.rollbackObservation = true;
    await this.agent.recordHistoryEvent('policy_session_snapshot_recovery', recovery);
  }
}
module.exports = { ManualPolicySwitch, manualPolicySettings: settings };

const test = require('node:test');
const assert = require('node:assert/strict');
const { sanitizeConfig, publicConfig } = require('../src/config');
const { FrameworkRuntime } = require('../src/frameworkRuntime');
const { SessionState } = require('../src/sessionState');
const { HighFrequencyAgent } = require('../src/highFrequencyAgent');
const { startRunDeadline } = require('../src/runDeadline');
const { chatCompletionRequestBody } = require('../src/llmClient');
const { providerConfig } = require('../src/brainActor');

function fixture(input = {}, infer = async () => ({ memory: 'updated memory', think: 'continue' })) {
  let now = 0;
  const config = sanitizeConfig(input, {});
  const state = new SessionState({ sessionId: 'timer-test', agentMode: config.agentMode });
  const runtime = new FrameworkRuntime({ config, sessionState: state,
    brainActor: { infer, stop() {} }, monotonicNow: () => now });
  runtime.observation({ capturedAt: '2026-11-27T00:00:00Z' });
  runtime.brainRuntime.seedInitialRevision({ instruction: 'current stage', memory: '', think: '' });
  runtime.setBrainFallbackClockRunning(true);
  return { runtime, state, config, advance: ms => { now += ms; } };
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('timed memory fallback counts gameplay gaps and excludes paused wall time', () => {
  const f = fixture();
  f.advance(9000);
  assert.equal(f.runtime.checkAutomaticBrainFallback(), null);
  f.runtime.setBrainFallbackClockRunning(false);
  f.advance(120000);
  assert.equal(f.runtime.checkAutomaticBrainFallback(), null);
  f.runtime.setBrainFallbackClockRunning(true);
  f.advance(999);
  assert.equal(f.runtime.checkAutomaticBrainFallback(), null);
  f.advance(1);
  assert.equal(f.state.timeline.actionSeq, 0, 'no action completion is needed for the timer');
  assert.deepEqual(f.runtime.consumeAutomaticBrainTrigger(), null);
  assert.deepEqual(f.runtime.checkAutomaticBrainFallback(), { reason: 'active_time_safety_barrier', updateType: 'fallback' });
  f.runtime.stop();
  const elapsed = f.runtime.brainRuntime.getStatus().fallbackActiveGameMs;
  f.advance(60000);
  assert.equal(f.runtime.brainRuntime.getStatus().fallbackActiveGameMs, elapsed);
  assert.equal(f.runtime.checkAutomaticBrainFallback(), null);
});

test('only committing a memory revision resets the 10-second obligation', async () => {
  let resolve;
  const f = fixture({}, () => new Promise(r => { resolve = r; }));
  const scheduler = f.runtime.brainRuntime.scheduler;
  f.advance(10000);
  assert(f.runtime.checkAutomaticBrainFallback());
  f.runtime.consumeAutomaticBrainTrigger();
  scheduler.deferCommit = true;
  f.runtime.requestBrainReplan('active_time_safety_barrier', { updateType: 'fallback' });
  await tick();
  f.advance(5000);
  assert.equal(scheduler.lastMemoryRefreshActiveMs, 0);
  assert.equal(f.runtime.checkAutomaticBrainFallback(), null, 'no duplicate while inference is active');
  resolve({ memory: 'new evidence', think: 'continue current stage' });
  await tick();
  assert(scheduler.pending);
  assert.equal(f.runtime.checkAutomaticBrainFallback(), null, 'buffered state is not yet a refresh');
  assert.equal(scheduler.lastMemoryRefreshActiveMs, 0);
  scheduler.deferCommit = false;
  assert.equal(scheduler.tryRelease(), true);
  assert.equal(scheduler.lastMemoryRefreshActiveMs, 15000);
  assert.equal(f.state.currentPolicyInstruction.text, 'current stage');
  assert.equal(f.state.brain.memory, 'new evidence');
  f.advance(9999);
  assert.equal(f.runtime.checkAutomaticBrainFallback(), null);
  f.advance(1);
  assert.equal(f.runtime.checkAutomaticBrainFallback().updateType, 'fallback');
  f.runtime.stop();
});

test('failed or superseded Brain calls do not reset the refresh deadline', async () => {
  const f = fixture({}, async () => { throw Error('model unavailable'); });
  f.advance(10000);
  f.runtime.requestBrainReplan('active_time_safety_barrier', { updateType: 'fallback' });
  await tick();
  f.runtime.invalidateBrainContext('failed request');
  assert.equal(f.runtime.brainRuntime.scheduler.lastMemoryRefreshActiveMs, 0);
  assert.equal(f.runtime.checkAutomaticBrainFallback().updateType, 'fallback');
  f.runtime.stop();
});

test('instruction fallback is disabled while explicit full Brain calls still work', async () => {
  const requests = [];
  const f = fixture({}, async request => {
    requests.push(request);
    return { instruction: 'next stage', memory: 'new evidence', think: 'proceed' };
  });
  for (let n = 0; n < 300; n++) f.runtime.executedAction({ candidateId: 'P1', basedOnObservationId: 'O1', execution: { durationMs: 200 } });
  assert.equal(f.runtime.consumeAutomaticBrainTrigger(), null);
  const explicit = f.runtime.createFrameworkFullBrainTrigger('explicit full update');
  assert(f.runtime.isFrameworkFullBrainTrigger(explicit));
  f.runtime.requestBrainReplan('policy_request', { updateType: 'full' });
  await tick();
  assert.equal(requests[0].updateType, 'full');
  assert.equal(f.state.currentPolicyInstruction.text, 'next stage');
  f.runtime.stop();
});

test('count-based memory and automatic instruction refresh remain explicitly selectable', () => {
  const f = fixture({ brainScheduler: { memoryFallbackIntervalMs: 0,
    instructionFallbackEnabled: true, maxActionsWithoutMemoryUpdate: 2, maxActionsWithoutInstructionUpdate: 3 } });
  f.runtime.executedAction({ candidateId: 'P1', basedOnObservationId: 'O1' });
  assert.equal(f.runtime.consumeAutomaticBrainTrigger(), null);
  f.runtime.executedAction({ candidateId: 'P1', basedOnObservationId: 'O1' });
  assert.equal(f.runtime.consumeAutomaticBrainTrigger().reason, 'action_count_safety_barrier');
  f.runtime.executedAction({ candidateId: 'P1', basedOnObservationId: 'O1' });
  const trigger = f.runtime.consumeAutomaticBrainTrigger();
  assert(f.runtime.isAutomaticInstructionRefresh(trigger));
  assert.equal(trigger.updateType, 'full');
  f.runtime.stop();
});

test('the host polls timed fallback without waiting for a completed action', () => {
  const f = fixture();
  const host = Object.create(HighFrequencyAgent.prototype);
  Object.assign(host, { frameworkRuntime: f.runtime, fallbackClockReady: true, taskState: 'NORMAL_GAMEPLAY',
    atActionLimit: () => false });
  host.updateBrainFallbackClock();
  f.advance(10000);
  host.updateBrainFallbackClock();
  assert.equal(host.pendingAutomaticBrainFallback.reason, 'active_time_safety_barrier');
  host.paused = true;
  host.updateBrainFallbackClock();
  const before = f.runtime.brainRuntime.getStatus().fallbackActiveGameMs;
  f.advance(60000);
  assert.equal(f.runtime.brainRuntime.getStatus().fallbackActiveGameMs, before);
  host.paused = false;
  for (const flag of ['gamePaused', 'recoveryMode', 'taskTransition', 'recoveryToolRunning', 'recoveryAwaitingBrain', 'finalizing', 'stopRequested']) {
    host[flag] = true;
    host.updateBrainFallbackClock();
    assert.equal(f.runtime.brainRuntime.getStatus().fallbackClockRunning, false, flag);
    host[flag] = false;
  }
  f.runtime.stop();
});

test('new defaults round-trip and explicit input still overrides environment', () => {
  const env = { AGI_GTA_BRAIN_MEMORY_FALLBACK_INTERVAL_MS: '15000', AGI_GTA_BRAIN_INSTRUCTION_FALLBACK_ENABLED: 'true',
    AGI_GTA_RUN_DURATION_MS: '60000', DEFAULT_MAX_STEPS_UNLIMITED: 'false', AGI_GTA_MAX_TOKENS: '400', AGI_GTA_BRAIN_MAX_TOKENS: '6000' };
  const fromEnv = sanitizeConfig({}, env);
  assert.equal(fromEnv.brainScheduler.memoryFallbackIntervalMs, 15000);
  assert.equal(fromEnv.brainScheduler.instructionFallbackEnabled, true);
  assert.equal(fromEnv.runDurationMs, 60000);
  assert.equal(fromEnv.maxStepsUnlimited, false);
  assert.equal(fromEnv.maxTokens, 400);
  assert.equal(fromEnv.brainProvider.maxTokens, 6000);
  const config = sanitizeConfig({ brainScheduler: { memoryFallbackIntervalMs: 0, instructionFallbackEnabled: false },
    maxTokens: null, brainProvider: { maxTokens: null }, runDurationMs: 0, maxStepsUnlimited: true }, env);
  assert.equal(config.brainScheduler.memoryFallbackIntervalMs, 0);
  assert.equal(config.brainScheduler.instructionFallbackEnabled, false);
  assert.equal(config.maxTokens, null);
  assert.equal(config.brainProvider.maxTokens, null);
  assert.equal(config.runDurationMs, 0);
  assert.equal(config.maxStepsUnlimited, true);
  const restored = sanitizeConfig(publicConfig(config), {});
  for (const key of ['brainScheduler', 'maxTokens', 'brainProvider', 'runDurationMs', 'maxStepsUnlimited'])
    assert.deepEqual(restored[key], config[key], key);
});

test('separate Policy and Brain HTTP requests receive their respective output budgets', () => {
  const config = sanitizeConfig({}, {});
  assert.equal(chatCompletionRequestBody(config, 'policy', []).max_tokens, null);
  assert.equal(chatCompletionRequestBody(providerConfig(config), 'brain', []).max_tokens, 16384);
});

function timers() {
  let callback, cleared = false, delay;
  return { setTimer: (fn, ms) => { callback = fn; delay = ms; return { unref() {} }; },
    clearTimer: () => { cleared = true; }, fire: () => callback?.(),
    get delay() { return delay; }, get cleared() { return cleared; } };
}

test('the 40-minute deadline uses the existing stop path while paused or waiting for a model', () => {
  const clock = timers(), calls = [];
  const host = Object.create(HighFrequencyAgent.prototype);
  Object.assign(host, { running: true, paused: true, stopRequested: false, contextEpoch: 0,
    policyOcr: { reset() {} }, resolvePauseWaiters() {}, getStatus: () => ({}), emitStatus() {}, emit() {},
    abortController: new AbortController(), recordHistoryEvent: async () => {},
    frameworkRuntime: { setBrainFallbackClockRunning: value => calls.push(['clock', value]), stop: reason => calls.push(['framework', reason]) },
    executor: { stop: async ({ reason }) => calls.push(['executor', reason]) } });
  host.cancelRunDeadline = startRunDeadline(host, sanitizeConfig({}, {}).runDurationMs, clock);
  assert.equal(clock.delay, 2400000);
  clock.fire();
  assert.equal(host.runDurationLimitReached, true);
  assert.equal(host.stopRequested, true);
  assert.equal(host.abortController.signal.aborted, true);
  assert(calls.some(([name, reason]) => name === 'executor' && reason === 'run_duration_limit'));
  assert(calls.some(([name, reason]) => name === 'framework' && reason === 'run_duration_limit'));
  assert.equal(clock.cleared, true);
  const count = calls.length;
  clock.fire();
  assert.equal(calls.length, count, 'deadline stops once');
});

test('deadline cancellation cannot stop a later session, and zero disables the time limit', () => {
  const clock = timers(), calls = [];
  const host = { running: true, stop: reason => calls.push(reason) };
  const cancel = startRunDeadline(host, 100, clock);
  cancel();
  clock.fire();
  assert.deepEqual(calls, []);
  assert.equal(clock.cleared, true);
  const disabled = timers();
  startRunDeadline(host, 0, disabled)();
  assert.equal(disabled.delay, undefined);
  assert.throws(() => startRunDeadline(host, -1), /runDurationMs/);
});

test('the response-count limit is opt-in and remains independent of the wall-clock limit', () => {
  const host = Object.create(HighFrequencyAgent.prototype);
  Object.assign(host, { defaultConfig: sanitizeConfig({}, {}), responseCount: 100000, actionCount: 100000 });
  assert.equal(host.atActionLimit(), false);
  host.defaultConfig = sanitizeConfig({ maxSteps: 25, maxStepsUnlimited: false }, {});
  assert.equal(host.atActionLimit(), true);
});

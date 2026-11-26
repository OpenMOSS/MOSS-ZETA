const test = require('node:test');
const assert = require('node:assert/strict');
const { sanitizeConfig } = require('../src/config');
const { PolicyActor } = require('../src/policyActor');
const { AgentRuntime } = require('../src/agentRuntime');
const { InputController } = require('../src/inputController');
const { ActionRuntime } = require('../src/actionRuntime');
const { ActionRuntime: SharedActionRuntime, neutralControllerState } = require('../../cadence/src/actionRuntime');
const { validateOcrEndpoints } = require('../src/policyOcrConfig');
const { recognizeScreenshotText } = require('../src/ocrClient');
const { detectTaskResult } = require('../src/taskResultOcrClient');

test('enabled OCR requires explicit endpoints while disabled OCR stays optional', () => {
  assert.throws(() => validateOcrEndpoints(sanitizeConfig({}, {})), /AGI_GTA_POLICY_OCR_URL/);
  assert.doesNotThrow(() => validateOcrEndpoints(sanitizeConfig({ ocrEnabled: false }, {})));
  const config = sanitizeConfig({ policyOcrUrl: 'http://ocr.test/ocr' }, {});
  assert.doesNotThrow(() => validateOcrEndpoints(config));
  config.taskResultOcrEnabled = true;
  assert.throws(() => validateOcrEndpoints(config), /AGI_GTA_TASK_RESULT_OCR_URL/);
  config.taskResultOcrUrl = 'file:///tmp/result';
  assert.throws(() => validateOcrEndpoints(config), /expected an HTTP\(S\) URL/);
  config.taskResultOcrUrl = 'http://result-ocr.test/task-result';
  assert.doesNotThrow(() => validateOcrEndpoints(config));
  assert.throws(() => validateOcrEndpoints(sanitizeConfig({ policyOcrProfile: 'legacy' }, {})), /AGI_GTA_OCR_URL/);
});

test('missing OCR endpoint rejects startup before a session or input begins', async () => {
  const { HighFrequencyAgent } = require('../src/highFrequencyAgent');
  const host = new HighFrequencyAgent({ defaultConfig: { policyOcrUrl: '', taskResultOcrEnabled: false } });
  await assert.rejects(host.start(), /AGI_GTA_POLICY_OCR_URL/);
  assert.equal(host.running, false);
  assert.equal(host.sessionId, null);
  assert.equal(host.runPromise, null);
});

test('OCR clients never contact a fallback address when their URL is missing', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url) => {
    calls.push(url);
    return { ok: true, text: async () => JSON.stringify({ ok: true,
      results: [{ roi: 'full_screen', pipeline: 'full', box: [0, 0, 1280, 720], lines: [] }] }) };
  });
  const observation = { imagePath: 'fixture.png' };
  const config = sanitizeConfig({ taskResultOcrEnabled: true }, {});
  assert.match((await recognizeScreenshotText(config, observation)).error, /AGI_GTA_POLICY_OCR_URL/);
  assert.match((await detectTaskResult(config, observation)).error, /AGI_GTA_TASK_RESULT_OCR_URL/);
  assert.deepEqual(calls, []);
  config.ocrUrl = 'http://ocr.test/ocr';
  config.taskResultOcrUrl = 'http://result-ocr.test/task-result';
  assert.equal((await recognizeScreenshotText(config, observation)).ok, true);
  assert.equal((await detectTaskResult(config, observation)).ok, true);
  assert.deepEqual(calls, [config.ocrUrl, config.taskResultOcrUrl]);
  const cleared = await detectTaskResult(config, observation, { url: '' });
  assert.match(cleared.error, /no default URL/);
  assert.equal(calls.length, 2);
});

const rendering = { version: 'think-memory-minimap-compact-v1', response_format: 'full', generation_prefix: '<think>' };
const fixed = { baseUrl: 'http://policy.test:8000/v1', modelName: 'fixture', policyTransport: 'lumine_session',
  allowThinking: true, hfCaptureIntervalMs: 200, hfScheduling: 'fixed_cadence', hfSubmissionOffsetMs: 80, policyStreaming: true };

test('equivalent JSON and environment rendering select the same fixed-cadence protocol', () => {
  const json = sanitizeConfig({ ...fixed, policyRendering: rendering }, {});
  const env = sanitizeConfig(fixed, { AGI_GTA_POLICY_RENDERING_JSON: JSON.stringify(rendering) });
  assert.deepEqual(env, json);
  assert.equal(json.hfPackageDurationMs, 200);
  assert.equal(json.hfChunkDurationMs * 6, 200);
  assert.equal(json.policyWindowMaxRounds, 16);
  assert.equal(json.policyWindowKeepRounds, 8);
  assert.equal(json.effectivePolicyOcrProfile, 'training_pbck_v2');
});
test('fixed cadence rejects incompatible action duration; legacy duration remains explicit', () => {
  assert.throws(() => sanitizeConfig({ ...fixed, policyRendering: rendering, hfPackageDurationMs: 400 }, {}), /hfPackageDurationMs=200/);
  assert.throws(() => sanitizeConfig({ ...fixed, policyRendering: rendering, hfPackageDurationMs: 'oops' }, {}), /hfPackageDurationMs=200/);
  assert.equal(sanitizeConfig({ hfScheduling: 'legacy', hfPackageDurationMs: 400,
    hfSubmissionOffsetMs: 0, policyStreaming: false }, {}).hfPackageDurationMs, 400);
});

function actorFor(content) {
  const messages = [{ role: 'system', content: 'fixture' }, { role: 'user', content: 'observation' }];
  const evidence = [], candidates = [];
  const actor = new PolicyActor({ getConfig: () => sanitizeConfig({ modelName: 'fixture', policyStreaming: false }, {}), getMessages: () => messages,
    appendMessage: m => messages.push(m), createCandidate: c => { const row = { ...c, candidateId: 'P' + (candidates.length + 1) }; candidates.push(row); return row; },
    recordModelResponse: async r => evidence.push(r),
    requestModelWithRetry: async () => ({ response: { model: 'fixture', content, responseMessage: { role: 'assistant', content }, raw: { choices: [{ message: { role: 'assistant', content } }] } }, attempts: 1 }) });
  return { actor, messages, evidence, candidates };
}
test('invalid actions stay in diagnostic evidence and never enter generated dialogue', async () => {
  const raw = '<think></think><a>+0+0+0+0A;A</a>';
  const f = actorFor(raw);
  assert.equal(await f.actor.infer({ frameworkObservationId: 'O1' }), null);
  assert.equal(f.messages.filter(m => m.role === 'assistant').length, 0);
  assert.equal(f.candidates[0].valid, false);
  assert.equal(f.evidence[0].raw.choices[0].message.content, raw);
  assert.match(f.evidence[0].parseError, /exactly 6 chunks/);
  assert.equal(f.evidence[0].responseMessage.content, raw);
  assert.equal(f.evidence[0].actionCode, '<a>+0+0+0+0A;A</a>');
});
test('valid assistant history preserves original bytes after validation', async () => {
  const raw = '<think></think><a>+0+0+0+0A;A;A;B;B;B</a>';
  const f = actorFor(raw), result = await f.actor.infer({ frameworkObservationId: 'O1' });
  assert(result?.candidate.valid);
  assert.deepEqual(f.messages.filter(m => m.role === 'assistant'), [{ role: 'assistant', content: raw }]);
});

function pausedHost(checks) {
  const host = Object.create(AgentRuntime.prototype), events = [], calls = [];
  const caps = new InputController({ platform: 'win32' }).capabilities();
  Object.assign(host, { gamePaused: false, recoveryMode: false, stopRequested: false, taskState: 'NORMAL_GAMEPLAY',
    defaultConfig: { ocrEnabled: true, modelWaitResumeDelayMs: 0 },
    inputController: { capabilities: () => caps, gamepadButtonDown: async () => {}, gamepadButtonUp: async () => {} },
    frameworkRuntime: { actionRuntime: { suspendCarry: async () => calls.push('release') },
      pauseHarness: { transition: (phase, paused) => events.push({ phase, paused }) } },
    tapGamepadButton: async () => calls.push('START'),
    verifyGameResumed: async () => { calls.push('before'); return checks.before || { conclusive: true, menuOpen: false }; },
    verifyPauseBeforeResume: async () => { calls.push('confirm'); return checks.after; },
    resetController: async () => calls.push('reset'), recordHistoryEvent: async () => {}, emit: () => {}, emitStatus: () => {} });
  return { host, events, calls };
}
test('local Windows pause becomes authoritative only after positive confirmation', async () => {
  const f = pausedHost({ after: { conclusive: true, menuOpen: true } });
  assert.equal(await f.host.pauseGame('full_brain'), true);
  assert.deepEqual(f.calls, ['release', 'before', 'START', 'confirm']);
  assert.equal(f.host.gamePaused, true);
  assert.deepEqual(f.events.at(-1), { phase: 'PAUSED_INFERENCE', paused: true });
});
test('an ignored pause key cannot enter the Brain pause state', async () => {
  const f = pausedHost({ after: { conclusive: true, menuOpen: false } });
  await assert.rejects(f.host.pauseGame('full_brain'), /did not open/);
  assert.equal(f.calls.filter(x => x === 'START').length, 3);
  assert.equal(f.host.gamePaused, false);
  assert(!f.events.some(e => e.paused));
  assert.equal(f.calls.at(-1), 'reset');
});
test('uncertain OCR before pause sends no key and records no confirmed pause', async () => {
  const f = pausedHost({ before: { conclusive: false, menuOpen: false } });
  await assert.rejects(f.host.pauseGame('full_brain'), /state unavailable before START/);
  assert(!f.calls.includes('START'));
  assert.equal(f.host.gamePaused, false);
});
test('uncertain OCR after pause fails closed without a second toggle', async () => {
  const f = pausedHost({ after: { conclusive: false, menuOpen: false } });
  await assert.rejects(f.host.pauseGame('full_brain'), /state unavailable after START/);
  assert.equal(f.calls.filter(x => x === 'START').length, 1);
  assert.equal(f.host.gamePaused, false);
});
test('an already open pause menu is verified without toggling it closed', async () => {
  const f = pausedHost({ before: { conclusive: true, menuOpen: true } });
  await f.host.pauseGame('full_brain');
  assert(!f.calls.includes('START')); assert.equal(f.host.gamePaused, true);
});
test('native pause observations continue using the native verification path', async () => {
  const f = pausedHost({});
  f.host.supportsNativePauseMenuObservation = () => true;
  f.host.enterVerifiedNativePause = async () => f.calls.push('native_confirm');
  await f.host.pauseGame('full_brain');
  assert.deepEqual(f.calls, ['release', 'native_confirm']);
});

test('Harness runtime reuses the shared base while preserving continuity behavior', async () => {
  const writes = [], resets = [];
  const runtime = new ActionRuntime({ inputController: { gamepadState: async s => writes.push(s), gamepadReset: async () => resets.push(true) } });
  assert(runtime instanceof SharedActionRuntime);
  const state = neutralControllerState(); state.buttons = ['a'];
  assert.equal(await runtime.beginPackage(state, { packageEndHoldMs: 100 }), true);
  const receipt = await runtime.applyChunk(state, {});
  await runtime.completeChunk(receipt, () => false);
  await runtime.finishPackage(state, { packageEndHoldMs: 100 });
  assert.equal(runtime.getStatus().continuity.enabled, true);
  assert(writes.length > 0);
  await runtime.reset('test_done');
  assert.equal(runtime.packageEndHold, null);
  assert(resets.length > 0);
});
test('recovery input uses the shared whole-package implementation', async () => {
  const calls = [];
  const runtime = new ActionRuntime({ inputController: { supportsControllerRuntime: () => true,
    executePausedPackage: async (states, options) => { calls.push({ states, options }); return { runtime: { carryActive: false } }; } } });
  assert.equal(await runtime.executeRecoveryPackage([neutralControllerState()], { chunkDurationMs: 100 }), true);
  assert.equal(calls.length, 1); assert.equal(calls[0].options.chunkDurationMs, 100);
});

const test = require('node:test');
const assert = require('node:assert/strict');
const { sanitizeConfig, publicConfig } = require('../src/config');
const report = require('../configs/report.example.json');

test('service endpoints and credentials stay empty without user configuration', () => {
  for (const input of [{}, report, { policyMode: 'trivial' }]) {
    const config = sanitizeConfig(input, {});
    for (const key of ['baseUrl', 'apiKey', 'policyStandbyBaseUrl', 'policyStandbyApiKey',
      'ocrUrl', 'policyOcrUrl', 'taskResultOcrUrl']) assert.equal(config[key], '', key);
    assert.equal(config.brainProvider.baseUrl, '');
    assert.equal(config.brainProvider.apiKey, '');
    const restored = sanitizeConfig(publicConfig(config), {});
    assert.equal(restored.ocrUrl, '');
    assert.equal(restored.taskResultOcrUrl, '');
  }
});

test('explicit empty service settings clear environment values without fallback', () => {
  const env = { OPENAI_BASE_URL: 'https://policy.test/v1', OPENAI_API_KEY: 'fixture-only',
    AGI_GTA_BRAIN_BASE_URL: 'https://brain.test/v1', AGI_GTA_BRAIN_API_KEY: 'fixture-only',
    AGI_GTA_POLICY_STANDBY_BASE_URL: 'https://standby.test/v1', AGI_GTA_POLICY_STANDBY_API_KEY: 'fixture-only',
    AGI_GTA_OCR_URL: 'http://ocr.test/ocr', AGI_GTA_POLICY_OCR_URL: 'http://policy-ocr.test/ocr',
    AGI_GTA_TASK_RESULT_OCR_URL: 'http://result-ocr.test/task-result' };
  const configured = sanitizeConfig({}, env);
  assert.equal(configured.baseUrl, env.OPENAI_BASE_URL);
  assert.equal(configured.apiKey, env.OPENAI_API_KEY);
  assert.equal(configured.brainProvider.baseUrl, env.AGI_GTA_BRAIN_BASE_URL);
  assert.equal(configured.brainProvider.apiKey, env.AGI_GTA_BRAIN_API_KEY);
  assert.equal(configured.ocrUrl, env.AGI_GTA_POLICY_OCR_URL);
  assert.equal(configured.taskResultOcrUrl, env.AGI_GTA_TASK_RESULT_OCR_URL);
  for (const policyOcrProfile of ['auto', 'legacy']) {
    const config = sanitizeConfig({ policyOcrProfile, baseUrl: '', apiKey: '', brainBaseUrl: '', brainApiKey: '',
      policyStandbyBaseUrl: '', policyStandbyApiKey: '', ocrUrl: '', policyOcrUrl: '', taskResultOcrUrl: '' }, env);
    for (const key of ['baseUrl', 'apiKey', 'policyStandbyBaseUrl', 'policyStandbyApiKey',
      'ocrUrl', 'policyOcrUrl', 'taskResultOcrUrl']) assert.equal(config[key], '', key);
    assert.equal(config.brainProvider.baseUrl, '');
    assert.equal(config.brainProvider.apiKey, '');
  }
});

test('no-config launch resolves to the shipped reference configuration', () => {
  const config = sanitizeConfig({}, {});
  assert.deepEqual(config, sanitizeConfig(report, {}));
  assert.equal(config.hfCaptureIntervalMs, 200);
  assert.equal(config.hfPackageDurationMs, 200);
  assert.equal(config.hfChunkDurationMs * 6, 200);
  assert.equal(config.hfSubmissionOffsetMs, 80);
  assert.equal(config.hfScheduling, 'fixed_cadence');
  assert.equal(config.policyStreaming, true);
  assert.equal(config.policyTransport, 'lumine_session');
  assert.equal(config.policyRendering.version, 'think-memory-minimap-compact-v1');
  assert.equal(config.policyRendering.generation_prefix, '<think>');
  assert.equal(config.brainProvider.inheritPolicy, false);
  assert.equal(config.effectivePolicyOcrProfile, 'training_pbck_v2');
  assert.equal(config.ocrUrl, '');
  assert.equal(config.ocrManagedService, false);
  assert.deepEqual(config.ocrRois, ['full_screen']);
  assert.equal(config.taskResultOcrEnabled, false);
  assert.equal(config.brainPrefillEnabled, false);
  assert.equal(config.policyHotSwapEnabled, false);
  assert.equal(config.brainScheduler.memoryFallbackIntervalMs, 10000);
  assert.equal(config.brainScheduler.instructionFallbackEnabled, false);
  assert.equal(config.brainProvider.maxTokens, 16384);
  assert.equal(config.maxTokens, null);
  assert.equal(config.maxStepsUnlimited, true);
  assert.equal(config.runDurationMs, 40 * 60 * 1000);
  assert.equal(config.brainMaxRetries, 1);
  assert.equal(config.policyMaxInvalidRetries, 3);
});

const overrides = {
  AGI_GTA_HF_SUBMISSION_OFFSET_MS: '40',
  AGI_GTA_BRAIN_INHERIT_POLICY: 'true',
  AGI_GTA_BRAIN_PREFILL_ENABLED: 'true',
  AGI_GTA_BRAIN_FALLBACK_SCOPE: 'full',
  AGI_GTA_BRAIN_MAX_TOKENS: '4096',
  AGI_GTA_POLICY_OCR_URL: 'http://127.0.0.1:19000/ocr',
  AGI_GTA_POLICY_OCR_MANAGED_SERVICE: 'true'
};

test('environment overrides still apply when no configuration file is supplied', () => {
  const config = sanitizeConfig({}, overrides);
  assert.equal(config.hfSubmissionOffsetMs, 40);
  assert.equal(config.brainProvider.inheritPolicy, true);
  assert.equal(config.brainPrefillEnabled, true);
  assert.equal(config.brainFallbackScope, 'full');
  assert.equal(config.brainProvider.maxTokens, 4096);
  assert.equal(config.policyOcrUrl, overrides.AGI_GTA_POLICY_OCR_URL);
  assert.equal(config.ocrUrl, overrides.AGI_GTA_POLICY_OCR_URL);
  assert.equal(config.ocrManagedService, true);
});

test('explicit configuration overrides environment values and report defaults', () => {
  const config = sanitizeConfig({ hfSubmissionOffsetMs: 60,
    brainProvider: { inheritPolicy: false, maxTokens: 8192 },
    brainPrefillEnabled: false, brainFallbackScope: 'memory_think',
    policyOcrUrl: 'http://127.0.0.1:19001/ocr', policyOcrManagedService: false }, overrides);
  assert.equal(config.hfSubmissionOffsetMs, 60);
  assert.equal(config.brainProvider.inheritPolicy, false);
  assert.equal(config.brainProvider.maxTokens, 8192);
  assert.equal(config.brainPrefillEnabled, false);
  assert.equal(config.brainFallbackScope, 'memory_think');
  assert.equal(config.ocrUrl, 'http://127.0.0.1:19001/ocr');
  assert.equal(config.ocrManagedService, false);
});

test('legacy stateless operation remains explicitly selectable', () => {
  const config = sanitizeConfig({ hfScheduling: 'legacy', hfCaptureIntervalMs: 400,
    hfPackageDurationMs: 400, hfSubmissionOffsetMs: 0,
    policyStreaming: false, policyTransport: 'stateless', policyRendering: null }, {});
  assert.equal(config.hfScheduling, 'legacy');
  assert.equal(config.hfPackageDurationMs, 400);
  assert.equal(config.hfCaptureIntervalMs, 400);
  assert.equal(config.policyTransport, 'stateless');
  assert.equal(config.policyRendering, null);
  assert.equal(config.policyStreaming, false);
  assert.equal(config.hfSubmissionMaxLatenessMs, 0);
  assert.equal(config.hfWatchdogAutoResume, false);
});

test('explicit trivial Policy keeps its action-only transport and history contract', () => {
  const config = sanitizeConfig({ policyMode: 'trivial' }, {});
  assert.equal(config.policyStreaming, false);
  assert.equal(config.policyTransport, 'stateless');
  assert.equal(config.policyRendering, null);
  assert.equal(config.allowThinking, false);
  assert.equal(config.hfCaptureIntervalMs, 200);
  assert.equal(config.hfPackageDurationMs, 200);
  assert.equal(config.policyWindowMaxRounds, 5);
  assert.equal(config.policyWindowKeepRounds, 3);
});

test('resolved configurations cannot mutate the default rendering contract', () => {
  const first = sanitizeConfig({}, {});
  first.policyRendering.version = 'modified';
  assert.equal(sanitizeConfig({}, {}).policyRendering.version, 'think-memory-minimap-compact-v1');
  assert.equal(report.policyRendering.version, 'think-memory-minimap-compact-v1');
});

test('public configuration preserves the default protocol through normalization', () => {
  const config = sanitizeConfig({}, {});
  const exposed = publicConfig(config);
  const restored = sanitizeConfig(exposed, {});
  for (const key of ['policyRendering', 'policyStreaming', 'policyTransport', 'allowThinking',
    'hfScheduling', 'hfSubmissionOffsetMs', 'hfCaptureIntervalMs', 'hfPackageDurationMs',
    'hfWatchdogAutoResume', 'brainProvider', 'policyOcrUrl', 'policyOcrManagedService',
    'brainScheduler', 'maxTokens', 'maxStepsUnlimited', 'runDurationMs']) {
    assert.deepEqual(restored[key], config[key], key);
  }
  const fallback = publicConfig({});
  assert.equal(fallback.hfPackageDurationMs, 200);
  assert.equal(fallback.hfCaptureIntervalMs, 200);
  assert.equal(fallback.policyStreaming, true);
  assert.equal(fallback.policyRendering.version, 'think-memory-minimap-compact-v1');
  assert.equal(fallback.brainProvider.inheritPolicy, false);
});

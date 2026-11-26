const REPORT_DEFAULTS = require('../configs/report.example.json');

// This profile owns the Policy OCR endpoint as well as the selector. The
// legacy shared sidecar may omit parts of the frame and cannot supply PBCK.
function policyOcrConfig(input = {}, env = {}) {
  const requested = input.policyOcrProfile ?? env.AGI_GTA_POLICY_OCR_PROFILE ?? 'auto';
  if (!['auto', 'legacy', 'training_pbck_v2'].includes(requested)) throw Error('Invalid policyOcrProfile');
  const profile = requested === 'auto'
    ? (input.policyRendering?.version === 'think-memory-minimap-compact-v1' ? 'training_pbck_v2' : 'legacy')
    : requested;
  const url = String(input.policyOcrUrl ?? env.AGI_GTA_POLICY_OCR_URL ?? '').trim();
  const managed = input.policyOcrManagedService ?? env.AGI_GTA_POLICY_OCR_MANAGED_SERVICE ?? REPORT_DEFAULTS.policyOcrManagedService;
  const manage = ![false, 'false', '0', 'no', 'off'].includes(typeof managed === 'boolean' ? managed : String(managed).toLowerCase());
  const fields = { policyOcrProfile: requested, effectivePolicyOcrProfile: profile,
    policyOcrUrl: url, policyOcrManagedService: manage };
  if (profile === 'legacy') return fields;
  // An unconfigured profile can be inspected, but cannot start OCR requests.
  if (url && !['http:', 'https:'].includes(new URL(url).protocol)) throw Error('Invalid policyOcrUrl');
  return { ...fields, ocrUrl: url, ocrManagedService: manage,
    ocrRois: ['full_screen'], ocrMinScore: 0.6, ocrPipeline: 'full', ocrDiffGate: false };
}

function isTrainingPolicyOcr(config) {
  return (config?.effectivePolicyOcrProfile ?? config?.policyOcrProfile) === 'training_pbck_v2';
}

function validateOcrEndpoints(config) {
  const training = isTrainingPolicyOcr(config);
  const endpoints = [
    [config.ocrEnabled, config.ocrUrl, training ? 'policyOcrUrl / AGI_GTA_POLICY_OCR_URL' : 'ocrUrl / AGI_GTA_OCR_URL'],
    [config.taskResultOcrEnabled, config.taskResultOcrUrl, 'taskResultOcrUrl / AGI_GTA_TASK_RESULT_OCR_URL']
  ];
  for (const [enabled, value, setting] of endpoints) {
    if (!enabled) continue;
    if (!String(value || '').trim()) throw Error(`Set ${setting}; enabled OCR has no default URL.`);
    let endpoint;
    try { endpoint = new URL(value); } catch { throw Error(`Invalid ${setting}; expected an HTTP(S) URL.`); }
    if (!['http:', 'https:'].includes(endpoint.protocol)) throw Error(`Invalid ${setting}; expected an HTTP(S) URL.`);
  }
}

module.exports = { policyOcrConfig, isTrainingPolicyOcr, validateOcrEndpoints };

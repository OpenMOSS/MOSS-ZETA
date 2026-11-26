const path = require("node:path");
const { normalizeRelativeSubpath } = require("./projectPaths");
// Keep the no-config launch on the report profile. Explicit input and environment
// values still take precedence over these fallbacks.
const REPORT_DEFAULTS = require("../configs/report.example.json");

const RESOLUTIONS = new Set(["1280x720", "1920x1080"]);
const DEFAULT_OCR_ROIS = ["tutorial_prompt", "mission_objective"];
const DEFAULT_TASK_RESULT_OCR_ROIS = ["full_screen"];
const PLAN_HORIZON = 5;
const DEFAULT_MAX_TOKENS = REPORT_DEFAULTS.maxTokens;
const MAX_COMPLETION_TOKENS = 65536;
// Explicit null means no application output-token cap. Undefined retains the
// existing defaults; null must survive sanitize/publicConfig round trips.
function completionTokenLimitFrom(candidates, fallback) {
  const value = candidates.find(candidate => candidate !== undefined);
  return value === undefined ? fallback : value === null ? null : boundedIntegerFrom(value, fallback, 1, MAX_COMPLETION_TOKENS);
}
const DEFAULT_TEMPERATURE = 0.2;
const MIN_TEMPERATURE = 0;
const MAX_TEMPERATURE = 2;
const REASONING_EFFORTS = new Set(["low", "medium", "high", "max"]);
const DEFAULT_POLICY_REASONING_EFFORT = "low";
const DEFAULT_BRAIN_REASONING_EFFORT = "low";
const DEFAULT_MODEL_WAIT_RESUME_DELAY_MS = 400;
const DEFAULT_HF_CARRY_ENABLED = false;
const DEFAULT_HF_CARRY_MAX_MS = 5000;
const DEFAULT_NAVIGATION_MINIMAP_ENABLED = false;
const DEFAULT_NAVIGATION_MINIMAP_CROP = {
  left: 12,
  top: 570,
  width: 215,
  height: 145,
  coordinateSpace: "image"
};
const DEFAULT_BRAIN_MIN_NEW_ACTIONS_BEFORE_START = 5;
const DEFAULT_BRAIN_SIMULATED_LAG_ACTIONS = 10;
const DEFAULT_BRAIN_MAX_IN_FLIGHT = 1;
const DEFAULT_BRAIN_TRAJECTORY_OBSERVATION_LIMIT = 8;
const DEFAULT_BRAIN_MAX_ACTIONS_WITHOUT_MEMORY_UPDATE = REPORT_DEFAULTS.brainScheduler.maxActionsWithoutMemoryUpdate;
const DEFAULT_BRAIN_MAX_ACTIONS_WITHOUT_INSTRUCTION_UPDATE = REPORT_DEFAULTS.brainScheduler.maxActionsWithoutInstructionUpdate;
const OCR_DEVICES = new Set(["auto", "cpu", "gpu"]);
const OCR_PIPELINES = new Set(["recognizer", "full"]);
const DEFAULT_AGENT_MODE = "high_frequency";
const INFERENCE_BACKENDS = new Set(["openai_http"]);
const DEFAULT_INFERENCE_BACKEND = "openai_http";
const REMOTE_SERVICE_TYPES = new Set(["stock", "lumine"]);
const DEFAULT_REMOTE_SERVICE_TYPE = "stock";
const DEFAULT_POLICY_MAX_INVALID_RETRIES = 3;
const DEFAULT_HF_CAPTURE_INTERVAL_MS = REPORT_DEFAULTS.hfCaptureIntervalMs;
const DEFAULT_HF_CAPTURE_HZ = 1000 / DEFAULT_HF_CAPTURE_INTERVAL_MS;
const DEFAULT_HF_CHUNKS_PER_PACKAGE = 6;
const DEFAULT_HF_CHUNK_DURATION_MS = REPORT_DEFAULTS.hfPackageDurationMs / DEFAULT_HF_CHUNKS_PER_PACKAGE;
const DEFAULT_HF_PACKAGE_END_HOLD_MS = 100;
const DEFAULT_HF_LONG_HOLD_THRESHOLD_CHUNKS = 12;
const DEFAULT_HF_LONG_HOLD_RELEASE_MS = 100;
const DEFAULT_HF_MAX_PENDING_PACKAGES = 1;
const DEFAULT_HF_CONTEXT_OBSERVATIONS = 5;
const DEFAULT_HF_REMOTE_MAX_OCR_TOKENS = 8192;
const DEFAULT_HF_AUTO_REPAIR_PACKAGES = false;
const DEFAULT_HF_CAPTURE_ARCHIVE_HZ = 1;
const DEFAULT_HF_CAPTURE_ARCHIVE_FORMAT = "jpeg";
const DEFAULT_HF_CAPTURE_ARCHIVE_QUALITY = 90;
const DEFAULT_HF_CAPTURE_SCRATCH_SECONDS = 60;
const DEFAULT_HF_CAPTURE_MAX_GB = 50;
const DEFAULT_HF_CAPTURE_SAVE_ON_EVENT = true;
const DEFAULT_SESSION_OUTPUT_SUBPATH = "";
const HF_CAPTURE_ARCHIVE_FORMATS = new Set(["jpeg", "webp", "png"]);

function numberFrom(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function booleanFrom(value, fallback = false) {
  if (typeof value === "boolean") {
    return value;
  }

  if (value === undefined || value === null || value === "") {
    return fallback;
  }

  const text = String(value).trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(text)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(text)) {
    return false;
  }

  return fallback;
}

function reasoningEffortFrom(value, fallback = DEFAULT_POLICY_REASONING_EFFORT) {
  const effort = String(value || fallback).trim().toLowerCase();
  return REASONING_EFFORTS.has(effort) ? effort : fallback;
}

function boundedNumberFrom(value, fallback, min, max) {
  return Math.max(min, Math.min(max, numberFrom(value, fallback)));
}

function boundedIntegerFrom(value, fallback, min, max) {
  return Math.trunc(boundedNumberFrom(value, fallback, min, max));
}

function stringListFrom(value, fallback = []) {
  const raw = Array.isArray(value) ? value : String(value || "").split(",");
  const items = raw
    .map((item) => String(item || "").trim())
    .filter(Boolean)
    .filter((item) => /^[a-zA-Z0-9_-]+$/.test(item));

  return items.length > 0 ? items : fallback.slice();
}

function ocrDeviceFrom(value, fallback = "gpu") {
  const device = String(value || fallback).trim().toLowerCase();
  return OCR_DEVICES.has(device) ? device : fallback;
}

function ocrPipelineFrom(value, fallback = "recognizer") {
  const pipeline = String(value || fallback).trim().toLowerCase();
  return OCR_PIPELINES.has(pipeline) ? pipeline : fallback;
}

function remoteServiceTypeFrom(value, fallback = DEFAULT_REMOTE_SERVICE_TYPE) {
  const type = String(value || fallback).trim().toLowerCase();
  return REMOTE_SERVICE_TYPES.has(type) ? type : fallback;
}

function minimapCropFrom(value, fallback = DEFAULT_NAVIGATION_MINIMAP_CROP) {
  let parsed = value;
  if (typeof value === "string") {
    try {
      parsed = JSON.parse(value);
    } catch {
      parsed = null;
    }
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return { ...fallback };
  }
  const coordinateSpace = String(parsed.coordinateSpace || fallback.coordinateSpace).trim().toLowerCase();
  return {
    left: numberFrom(parsed.left, fallback.left),
    top: numberFrom(parsed.top, fallback.top),
    width: numberFrom(parsed.width, fallback.width),
    height: numberFrom(parsed.height, fallback.height),
    coordinateSpace: coordinateSpace === "normalized" ? "normalized" : "image"
  };
}

function hfCaptureArchiveFormatFrom(value, fallback = DEFAULT_HF_CAPTURE_ARCHIVE_FORMAT) {
  const format = String(value || fallback).trim().toLowerCase();
  const normalized = format === "jpg" ? "jpeg" : format;
  return HF_CAPTURE_ARCHIVE_FORMATS.has(normalized) ? normalized : fallback;
}

function normalizeSessionOutputSubpathForPublic(value) {
  try {
    return normalizeRelativeSubpath(value).split(path.sep).join("/");
  } catch {
    return DEFAULT_SESSION_OUTPUT_SUBPATH;
  }
}

function sanitizeConfig(input = {}, env = process.env) {
  const policyRendering = input.policyRendering !== undefined ? input.policyRendering
    : env.AGI_GTA_POLICY_RENDERING_JSON ? JSON.parse(env.AGI_GTA_POLICY_RENDERING_JSON)
      : structuredClone(REPORT_DEFAULTS.policyRendering);
  const localOcrLockedDisabled = booleanFrom(env.AGI_GTA_LOCK_LOCAL_OCR_DISABLED, false);
  const resolution = RESOLUTIONS.has(String(input.resolution || "")) ? String(input.resolution) : env.DEFAULT_RESOLUTION || "1280x720";
  const safeResolution = RESOLUTIONS.has(resolution) ? resolution : "1280x720";
  const maxSteps = Math.max(1, Math.min(10000, Math.trunc(numberFrom(input.maxSteps ?? input.maxActions ?? env.DEFAULT_MAX_STEPS ?? env.DEFAULT_MAX_ACTIONS, 25))));
  const maxStepsUnlimited = booleanFrom(input.maxStepsUnlimited ?? input.maxActionsUnlimited ?? env.DEFAULT_MAX_STEPS_UNLIMITED ?? env.DEFAULT_MAX_ACTIONS_UNLIMITED, REPORT_DEFAULTS.maxStepsUnlimited);
  const runDurationMs = boundedIntegerFrom(input.runDurationMs ?? env.AGI_GTA_RUN_DURATION_MS, REPORT_DEFAULTS.runDurationMs, 0, 2147483647);
  const autoRefreshSessionOnMaxSteps = booleanFrom(input.autoRefreshSessionOnMaxSteps ?? env.DEFAULT_AUTO_REFRESH_SESSION_ON_MAX_STEPS, false);
  const ocrRois = stringListFrom(input.ocrRois ?? env.AGI_GTA_OCR_ROIS, DEFAULT_OCR_ROIS);
  const taskResultOcrRois = stringListFrom(input.taskResultOcrRois ?? env.AGI_GTA_TASK_RESULT_OCR_ROIS, DEFAULT_TASK_RESULT_OCR_ROIS);
  const planEvery = boundedIntegerFrom(input.planEvery ?? env.AGI_GTA_PLAN_EVERY, 0, 0, 1000);
  const maxTokens = completionTokenLimitFrom([input.maxTokens, env.AGI_GTA_MAX_TOKENS], DEFAULT_MAX_TOKENS);
  const temperature = boundedNumberFrom(input.temperature ?? env.AGI_GTA_TEMPERATURE, DEFAULT_TEMPERATURE, MIN_TEMPERATURE, MAX_TEMPERATURE);
  const modelWaitResumeDelayMs = boundedIntegerFrom(input.modelWaitResumeDelayMs ?? env.AGI_GTA_MODEL_WAIT_RESUME_DELAY_MS, DEFAULT_MODEL_WAIT_RESUME_DELAY_MS, 0, 5000);
  const resumeVerificationTailMs = boundedIntegerFrom(
    input.resumeVerificationTailMs ?? input.pausedPackageResumeCarryWindowMs ?? env.AGI_GTA_RESUME_VERIFICATION_TAIL_MS
      ?? env.AGI_GTA_PAUSED_PACKAGE_RESUME_CARRY_WINDOW_MS, 200, 0, 5000
  );
  const hfCarryEnabled = booleanFrom(
    input.hfCarryEnabled ?? input.pausedPackageCarryEnabled ?? env.AGI_GTA_HF_CARRY_ENABLED ?? env.AGI_GTA_PAUSED_PACKAGE_CARRY_ENABLED,
    DEFAULT_HF_CARRY_ENABLED
  );
  const hfCarryMaxMs = boundedIntegerFrom(
    input.hfCarryMaxMs ?? input.pausedPackageCarryMaxMs ?? env.AGI_GTA_HF_CARRY_MAX_MS ?? env.AGI_GTA_PAUSED_PACKAGE_CARRY_MAX_MS,
    DEFAULT_HF_CARRY_MAX_MS, 100, 30000
  );
  const navigationMinimapEnabled = booleanFrom(input.navigationMinimapEnabled ?? input.navigationMinimap?.enabled ?? env.AGI_GTA_NAVIGATION_MINIMAP_ENABLED, DEFAULT_NAVIGATION_MINIMAP_ENABLED);
  const navigationMinimapCrop = minimapCropFrom(input.navigationMinimapCrop ?? input.navigationMinimap?.crop ?? env.AGI_GTA_NAVIGATION_MINIMAP_CROP_JSON);
  const brainMinNewActionsBeforeStart = boundedIntegerFrom(input.brainMinNewActionsBeforeStart ?? input.brainScheduler?.minNewActionsBeforeStart ?? env.AGI_GTA_BRAIN_MIN_NEW_ACTIONS_BEFORE_START, DEFAULT_BRAIN_MIN_NEW_ACTIONS_BEFORE_START, 1, 10000);
  const brainSimulatedLagActions = boundedIntegerFrom(input.brainSimulatedLagActions ?? input.brainScheduler?.simulatedLagActions ?? env.AGI_GTA_BRAIN_SIMULATED_LAG_ACTIONS, DEFAULT_BRAIN_SIMULATED_LAG_ACTIONS, 0, 10000);
  const brainMaxInFlight = boundedIntegerFrom(input.brainMaxInFlight ?? input.brainScheduler?.maxInFlight ?? env.AGI_GTA_BRAIN_MAX_IN_FLIGHT, DEFAULT_BRAIN_MAX_IN_FLIGHT, 1, 1);
  const brainFallbackScope = String(input.brainFallbackScope ?? env.AGI_GTA_BRAIN_FALLBACK_SCOPE ?? 'memory_think');
  if (!['full', 'memory_think'].includes(brainFallbackScope)) throw Error('Unknown Brain fallback scope');
  const brainTrajectoryObservationLimit = boundedIntegerFrom(input.brainTrajectoryObservationLimit ?? env.AGI_GTA_BRAIN_TRAJECTORY_OBSERVATION_LIMIT, DEFAULT_BRAIN_TRAJECTORY_OBSERVATION_LIMIT, 1, 50);
  const brainPrefillEnabled = booleanFrom(input.brainPrefillEnabled ?? env.AGI_GTA_BRAIN_PREFILL_ENABLED, false);
  const brainPrefillIntervalMs = boundedIntegerFrom(input.brainPrefillIntervalMs ?? env.AGI_GTA_BRAIN_PREFILL_INTERVAL_MS, 1000, 200, 60000);
  const brainPrefillTimeoutMs = boundedIntegerFrom(input.brainPrefillTimeoutMs ?? env.AGI_GTA_BRAIN_PREFILL_TIMEOUT_MS, 10000, 1000, 120000);
  const brainMaxActionsWithoutMemoryUpdate = boundedIntegerFrom(input.brainMaxActionsWithoutMemoryUpdate ?? input.brainScheduler?.maxActionsWithoutMemoryUpdate ?? env.AGI_GTA_BRAIN_MAX_ACTIONS_WITHOUT_MEMORY_UPDATE, DEFAULT_BRAIN_MAX_ACTIONS_WITHOUT_MEMORY_UPDATE, 1, 100000);
  const brainMaxActionsWithoutInstructionUpdate = boundedIntegerFrom(input.brainMaxActionsWithoutInstructionUpdate ?? input.brainScheduler?.maxActionsWithoutInstructionUpdate ?? env.AGI_GTA_BRAIN_MAX_ACTIONS_WITHOUT_INSTRUCTION_UPDATE, DEFAULT_BRAIN_MAX_ACTIONS_WITHOUT_INSTRUCTION_UPDATE, 1, 100000);
  const brainMemoryFallbackIntervalMs = boundedIntegerFrom(input.brainMemoryFallbackIntervalMs ?? input.brainScheduler?.memoryFallbackIntervalMs ?? env.AGI_GTA_BRAIN_MEMORY_FALLBACK_INTERVAL_MS, REPORT_DEFAULTS.brainScheduler.memoryFallbackIntervalMs, 0, 3600000);
  const brainInstructionFallbackEnabled = booleanFrom(input.brainInstructionFallbackEnabled ?? input.brainScheduler?.instructionFallbackEnabled ?? env.AGI_GTA_BRAIN_INSTRUCTION_FALLBACK_ENABLED, REPORT_DEFAULTS.brainScheduler.instructionFallbackEnabled);
  const brainInheritPolicy = booleanFrom(input.brainInheritPolicy ?? input.brainProvider?.inheritPolicy ?? env.AGI_GTA_BRAIN_INHERIT_POLICY, REPORT_DEFAULTS.brainProvider.inheritPolicy);
  const requestedMode = String(input.agentMode ?? env.AGI_GTA_AGENT_MODE ?? DEFAULT_AGENT_MODE).trim().toLowerCase();
  if (requestedMode !== DEFAULT_AGENT_MODE) throw Error('Only high_frequency mode is supported');
  const policyMode = String(input.policyMode ?? env.AGI_GTA_POLICY_MODE ?? "brain_policy").trim().toLowerCase();
  if (!["brain_policy", "trivial"].includes(policyMode)) throw Error("Unknown Policy mode");
  const trivialPolicy = policyMode === "trivial";
  const requestedOcrReset = input.trivialOcrHistoryResetMs ?? env.AGI_GTA_TRIVIAL_OCR_HISTORY_RESET_MS ?? 10000;
  const trivialOcrHistoryResetMs = Number(requestedOcrReset);
  if ((!['number', 'string'].includes(typeof requestedOcrReset))
    || String(requestedOcrReset).trim() === '' || !Number.isInteger(trivialOcrHistoryResetMs)
    || (trivialOcrHistoryResetMs !== 0 && (trivialOcrHistoryResetMs < 1000 || trivialOcrHistoryResetMs > 3600000))) {
    throw Error('trivialOcrHistoryResetMs must be 0 or an integer in [1000, 3600000]ms');
  }
  const inferenceBackend = DEFAULT_INFERENCE_BACKEND;
  const policyWindowMaxRounds = trivialPolicy ? 5 : boundedIntegerFrom(input.policyWindowMaxRounds, 16, 2, 50);
  const policyWindowKeepRounds = trivialPolicy ? 3 : boundedIntegerFrom(input.policyWindowKeepRounds, Math.min(8, policyWindowMaxRounds - 1), 1, policyWindowMaxRounds - 1);
  const policyManualSwitch = require('./manualPolicySwitch').manualPolicySettings(input.policyManualSwitch);
  const mobilityRecovery = require('./mobilityRecovery').recoverySettings(input.mobilityRecovery);
  const brainMaxOverrides = boundedIntegerFrom(input.brainMaxOverrides ?? env.AGI_GTA_BRAIN_MAX_OVERRIDES, 3, 0, 100);
  const brainMaxWaitMs = boundedIntegerFrom(input.brainMaxWaitMs ?? env.AGI_GTA_BRAIN_MAX_WAIT_MS, 15000, 1, 300000);
  const policyMaxInvalidRetries = boundedIntegerFrom(
    input.policyMaxInvalidRetries ?? input.pausedPackageMaxRetries ?? env.AGI_GTA_POLICY_MAX_INVALID_RETRIES
      ?? env.AGI_GTA_PAUSED_PACKAGE_MAX_RETRIES, DEFAULT_POLICY_MAX_INVALID_RETRIES, 0, 10
  );
  const legacyHz = input.hfCaptureHz ?? env.AGI_GTA_HF_CAPTURE_HZ;
  const intervalSource = input.hfCaptureIntervalMs
    ?? (input.hfCaptureHz != null ? 1000 / boundedNumberFrom(input.hfCaptureHz, DEFAULT_HF_CAPTURE_HZ, 0.1, 30) : undefined)
    ?? env.AGI_GTA_HF_CAPTURE_INTERVAL_MS
    ?? (legacyHz != null ? 1000 / boundedNumberFrom(legacyHz, DEFAULT_HF_CAPTURE_HZ, 0.1, 30) : undefined);
  const hfCaptureIntervalMs = boundedIntegerFrom(intervalSource, trivialPolicy ? 200 : DEFAULT_HF_CAPTURE_INTERVAL_MS, 33, 10000);
  const hfCaptureHz = 1000 / hfCaptureIntervalMs;
  const hfSubmissionOffsetMs = boundedIntegerFrom(input.hfSubmissionOffsetMs ?? env.AGI_GTA_HF_SUBMISSION_OFFSET_MS, trivialPolicy ? 0 : REPORT_DEFAULTS.hfSubmissionOffsetMs, 0, 199);
  const hfScheduling = input.hfScheduling ?? env.AGI_GTA_HF_SCHEDULING ?? (trivialPolicy ? 'auto' : REPORT_DEFAULTS.hfScheduling);
  if (!['auto', 'legacy', 'fixed_cadence'].includes(hfScheduling)) throw Error('Invalid hfScheduling');
  const trainedThinkCadence = ['think-memory-minimap-v1', 'think-memory-minimap-compact-v1'].includes(policyRendering?.version);
  if (hfSubmissionOffsetMs && (hfScheduling === 'legacy' || !(trivialPolicy || trainedThinkCadence) || hfCaptureIntervalMs !== 200)) throw Error('hfSubmissionOffsetMs requires fixed 200ms trivial or trained think/minimap scheduling');
  if (hfScheduling === 'fixed_cadence' && (!(trivialPolicy || trainedThinkCadence) || hfCaptureIntervalMs !== 200)) {
    throw Error('fixed_cadence requires high_frequency + a supported Policy contract with hfCaptureIntervalMs=200');
  }
  const legacyRemoteService = remoteServiceTypeFrom(input.remoteServiceType ?? env.AGI_GTA_REMOTE_SERVICE_TYPE);
  const requestedTransport = input.policyTransport ?? env.AGI_GTA_POLICY_TRANSPORT;
  const policyTransport = ["stateless", "lumine_session"].includes(requestedTransport)
    ? requestedTransport : (trivialPolicy ? (legacyRemoteService === "lumine" ? "lumine_session" : "stateless") : REPORT_DEFAULTS.policyTransport);
  const policyStreaming = booleanFrom(input.policyStreaming, !trivialPolicy && REPORT_DEFAULTS.policyStreaming);
  const hfActionStartDelayMs = boundedIntegerFrom(input.hfActionStartDelayMs, 200, 1, 400);
  const hfSubmissionMaxLatenessMs = boundedIntegerFrom(input.hfSubmissionMaxLatenessMs, policyStreaming ? 100 : 0, 0, 100);
  const hfValidActionTimeoutMs = boundedIntegerFrom(input.hfValidActionTimeoutMs, 1000, 500, 1000);
  const hfWatchdogAutoResume = booleanFrom(input.hfWatchdogAutoResume, policyStreaming);
  const policyHotSwapEnabled = booleanFrom(input.policyHotSwapEnabled, false);
  const policyStandbyBaseUrl = String(input.policyStandbyBaseUrl ?? env.AGI_GTA_POLICY_STANDBY_BASE_URL ?? '').trim();
  const policyStandbyApiKey = String(input.policyStandbyApiKey ?? env.AGI_GTA_POLICY_STANDBY_API_KEY ?? '');
  const policyHandoffPrepareTimeoutMs = boundedIntegerFrom(input.policyHandoffPrepareTimeoutMs, 15000, 1000, 120000);
  const policyHandoffMaxObservationAgeMs = boundedIntegerFrom(input.policyHandoffMaxObservationAgeMs, 400, 200, 400);
  if (policyHotSwapEnabled) {
    if (!policyStreaming || !policyStandbyBaseUrl) throw Error('Policy hot swap requires streaming and a standby service');
    const primary = new URL(String(input.baseUrl ?? env.OPENAI_BASE_URL ?? ''));
    const standby = new URL(policyStandbyBaseUrl);
    if (!['http:', 'https:'].includes(standby.protocol) || primary.origin === standby.origin)
      throw Error('Policy hot swap requires two distinct HTTP service origins');
  }
  if (hfSubmissionMaxLatenessMs && !policyStreaming) throw Error('Elastic submission requires Policy streaming');
  if (policyStreaming && !(hfScheduling === 'fixed_cadence'
    && hfSubmissionOffsetMs > 0 && policyTransport === 'lumine_session' && policyRendering?.version === 'think-memory-minimap-compact-v1'))
    throw Error('Policy streaming requires fixed 200ms compact think session transport');
  if (policyStreaming && booleanFrom(input.policyExecutionAcknowledgements, false))
    throw Error('Generated dialogue streaming and completed-ack history are mutually exclusive');
  const policyExecutionAcknowledgements = booleanFrom(input.policyExecutionAcknowledgements, false);
  if (policyExecutionAcknowledgements && !(hfScheduling === 'fixed_cadence'
    && hfSubmissionOffsetMs > 0 && policyTransport === 'lumine_session' && policyRendering?.version === 'think-memory-minimap-compact-v1'))
    throw Error('Execution acknowledgements require fixed-cadence compact think session transport with a positive submission offset');
  const hfChunksPerPackage = 6;
  const requestedHfDuration = input.hfPackageDurationMs ?? env.AGI_GTA_HF_PACKAGE_DURATION_MS;
  const fixedCadence = hfScheduling === 'fixed_cadence'
    || (hfScheduling === 'auto' && trivialPolicy && hfCaptureIntervalMs === 200);
  if (fixedCadence && requestedHfDuration != null && Number(requestedHfDuration) !== 200)
    throw Error('Fixed cadence requires hfPackageDurationMs=200');
  const hfPackageDurationMs = fixedCadence || trivialPolicy ? 200 : boundedNumberFrom(requestedHfDuration, 400, 6, 30000);
  const hfChunkDurationMs = hfPackageDurationMs / 6;
  const hfPackageEndHoldMs = boundedIntegerFrom(input.hfPackageEndHoldMs ?? env.AGI_GTA_HF_PACKAGE_END_HOLD_MS, DEFAULT_HF_PACKAGE_END_HOLD_MS, 0, 60000);
  const hfLongHoldThresholdChunks = boundedIntegerFrom(input.hfLongHoldThresholdChunks ?? env.AGI_GTA_HF_LONG_HOLD_THRESHOLD_CHUNKS, DEFAULT_HF_LONG_HOLD_THRESHOLD_CHUNKS, 1, 100000);
  const hfLongHoldReleaseMs = boundedIntegerFrom(input.hfLongHoldReleaseMs ?? env.AGI_GTA_HF_LONG_HOLD_RELEASE_MS, DEFAULT_HF_LONG_HOLD_RELEASE_MS, 0, 60000);
  const hfAllowBarePackage = booleanFrom(input.hfAllowBarePackage ?? env.AGI_GTA_HF_ALLOW_BARE_PACKAGE, false);
  const hfMaxPendingPackages = boundedIntegerFrom(input.hfMaxPendingPackages ?? env.AGI_GTA_HF_MAX_PENDING_PACKAGES, DEFAULT_HF_MAX_PENDING_PACKAGES, 1, 20);
  const hfContextObservations = boundedIntegerFrom(input.hfContextObservations ?? env.AGI_GTA_HF_CONTEXT_OBSERVATIONS, DEFAULT_HF_CONTEXT_OBSERVATIONS, 1, 20);
  const hfRemoteMaxOcrTokens = boundedIntegerFrom(input.hfRemoteMaxOcrTokens ?? env.AGI_GTA_HF_REMOTE_MAX_OCR_TOKENS, DEFAULT_HF_REMOTE_MAX_OCR_TOKENS, 0, 1048576);
  const hfAutoRepairPackages = booleanFrom(input.hfAutoRepairPackages ?? env.AGI_GTA_HF_AUTO_REPAIR_PACKAGES, DEFAULT_HF_AUTO_REPAIR_PACKAGES);
  const hfCaptureArchiveHz = boundedNumberFrom(input.hfCaptureArchiveHz ?? env.AGI_GTA_HF_CAPTURE_ARCHIVE_HZ, DEFAULT_HF_CAPTURE_ARCHIVE_HZ, 0, 30);
  const hfCaptureArchiveFormat = hfCaptureArchiveFormatFrom(input.hfCaptureArchiveFormat ?? env.AGI_GTA_HF_CAPTURE_ARCHIVE_FORMAT);
  const hfCaptureArchiveQuality = boundedIntegerFrom(input.hfCaptureArchiveQuality ?? env.AGI_GTA_HF_CAPTURE_ARCHIVE_QUALITY, DEFAULT_HF_CAPTURE_ARCHIVE_QUALITY, 1, 100);
  const hfCaptureScratchSeconds = boundedIntegerFrom(input.hfCaptureScratchSeconds ?? env.AGI_GTA_HF_CAPTURE_SCRATCH_SECONDS, DEFAULT_HF_CAPTURE_SCRATCH_SECONDS, 1, 3600);
  const hfCaptureMaxGb = boundedNumberFrom(input.hfCaptureMaxGb ?? env.AGI_GTA_HF_CAPTURE_MAX_GB, DEFAULT_HF_CAPTURE_MAX_GB, 0.1, 100000);
  const hfCaptureSaveOnEvent = booleanFrom(input.hfCaptureSaveOnEvent ?? env.AGI_GTA_HF_CAPTURE_SAVE_ON_EVENT, DEFAULT_HF_CAPTURE_SAVE_ON_EVENT);
  const sessionOutputSubpath = normalizeRelativeSubpath(
    input.sessionOutputSubpath ?? env.AGI_GTA_SESSION_OUTPUT_SUBPATH ?? DEFAULT_SESSION_OUTPUT_SUBPATH
  );
  const requestedOcrDevice = ocrDeviceFrom(input.ocrDevice ?? env.AGI_GTA_OCR_DEVICE, "gpu");
  const ocrDevice = requestedOcrDevice;
  const requestedTaskResultOcrDevice = ocrDeviceFrom(input.taskResultOcrDevice ?? env.AGI_GTA_TASK_RESULT_OCR_DEVICE ?? input.ocrDevice ?? env.AGI_GTA_OCR_DEVICE, "gpu");
  const taskResultOcrDevice = requestedTaskResultOcrDevice;

  return {
    agentMode: DEFAULT_AGENT_MODE,
    policyMode,
    inferenceBackend,
    remoteServiceType: legacyRemoteService,
    policyTransport,
    policyExecutionAcknowledgements,
    policyStreaming, hfActionStartDelayMs, hfSubmissionMaxLatenessMs, hfValidActionTimeoutMs, hfWatchdogAutoResume,
    policyHotSwapEnabled, policyStandbyBaseUrl, policyStandbyApiKey,
    policyHandoffPrepareTimeoutMs, policyHandoffMaxObservationAgeMs,
    trivialOcrHistoryResetMs,
    policyRendering: trivialPolicy ? null : policyRendering,
    baseUrl: String(input.baseUrl ?? env.OPENAI_BASE_URL ?? "").trim(),
    apiFormat: String(input.apiFormat ?? env.OPENAI_API_FORMAT ?? "chat_completions").trim(),
    apiKey: String(input.apiKey ?? env.OPENAI_API_KEY ?? ""),
    modelName: String(input.modelName ?? env.OPENAI_MODEL ?? "").trim(),
    displayId: input.displayId === undefined || input.displayId === null ? "" : String(input.displayId),
    resolution: safeResolution,
    maxSteps,
    maxActions: maxSteps,
    maxStepsUnlimited,
    runDurationMs,
    maxActionsUnlimited: maxStepsUnlimited,
    autoRefreshSessionOnMaxSteps,
    allowThinking: trivialPolicy ? false : booleanFrom(input.allowThinking ?? env.AGI_GTA_ALLOW_THINKING, REPORT_DEFAULTS.allowThinking),
    reasoningEffort: reasoningEffortFrom(
      input.reasoningEffort ?? env.AGI_GTA_REASONING_EFFORT,
      DEFAULT_POLICY_REASONING_EFFORT
    ),
    planEvery,
    planHorizon: PLAN_HORIZON,
    hfPackageDurationMs,
    policyWindowMaxRounds, policyWindowKeepRounds, brainMaxOverrides, brainMaxWaitMs,
    policyManualSwitch, mobilityRecovery,
    policyIncludeMinimap: !trivialPolicy && booleanFrom(input.policyIncludeMinimap, policyManualSwitch.enabled),
    policyMaxInvalidRetries,
    hfScheduling,
    hfSubmissionOffsetMs,
    hfCaptureIntervalMs,
    hfCaptureHz,
    hfChunksPerPackage,
    hfChunkDurationMs,
    hfPackageEndHoldMs,
    hfLongHoldThresholdChunks,
    hfLongHoldReleaseMs,
    hfAllowBarePackage,
    hfMaxPendingPackages,
    hfContextObservations,
    hfRemoteMaxOcrTokens,
    hfAutoRepairPackages,
    hfCaptureArchiveHz,
    hfCaptureArchiveFormat,
    hfCaptureArchiveQuality,
    hfCaptureScratchSeconds,
    hfCaptureMaxGb,
    hfCaptureSaveOnEvent,
    sessionOutputSubpath,
    temperature,
    modelWaitResumeDelayMs,
    resumeVerificationTailMs,
    hfCarryEnabled,
    hfCarryMaxMs,
    navigationMinimap: {
      enabled: trivialPolicy ? false : navigationMinimapEnabled,
      crop: navigationMinimapCrop
    },
    brainScheduler: {
      minNewActionsBeforeStart: brainMinNewActionsBeforeStart,
      simulatedLagActions: brainSimulatedLagActions,
      maxInFlight: brainMaxInFlight
      ,
      maxActionsWithoutMemoryUpdate: brainMaxActionsWithoutMemoryUpdate,
      maxActionsWithoutInstructionUpdate: brainMaxActionsWithoutInstructionUpdate,
      memoryFallbackIntervalMs: brainMemoryFallbackIntervalMs,
      instructionFallbackEnabled: brainInstructionFallbackEnabled
    },
    brainFallbackScope,
    brainTrajectoryObservationLimit,
    brainPrefillEnabled,
    brainPrefillIntervalMs,
    brainPrefillTimeoutMs,
    policyRequestTimeoutMs: boundedIntegerFrom(input.policyRequestTimeoutMs ?? env.AGI_GTA_POLICY_REQUEST_TIMEOUT_MS, 120000, 0, 2147483647),
    hfPauseAllBrainUpdates: booleanFrom(input.hfPauseAllBrainUpdates ?? env.AGI_GTA_HF_PAUSE_ALL_BRAIN_UPDATES, false),
    brainMaxRetries: boundedIntegerFrom(input.brainMaxRetries ?? env.AGI_GTA_BRAIN_MAX_RETRIES, 1, 0, 5),
    brainRetryDelayMs: boundedIntegerFrom(input.brainRetryDelayMs ?? env.AGI_GTA_BRAIN_RETRY_DELAY_MS, 500, 0, 5000),
    brainRequestTimeoutMs: boundedIntegerFrom(input.brainRequestTimeoutMs ?? env.AGI_GTA_BRAIN_REQUEST_TIMEOUT_MS, 0, 0, 2147483647),
    brainProvider: {
      inheritPolicy: brainInheritPolicy,
      apiFormat: String(input.brainApiFormat ?? input.brainProvider?.apiFormat ?? env.AGI_GTA_BRAIN_API_FORMAT ?? "chat_completions").trim(),
      baseUrl: String(input.brainBaseUrl ?? input.brainProvider?.baseUrl ?? env.AGI_GTA_BRAIN_BASE_URL ?? "").trim(),
      apiKey: String(input.brainApiKey ?? input.brainProvider?.apiKey ?? env.AGI_GTA_BRAIN_API_KEY ?? ""),
      modelName: String(input.brainModelName ?? input.brainProvider?.modelName ?? env.AGI_GTA_BRAIN_MODEL ?? "").trim(),
      allowThinking: booleanFrom(input.brainAllowThinking ?? input.brainProvider?.allowThinking ?? env.AGI_GTA_BRAIN_ALLOW_THINKING, true),
      reasoningEffort: reasoningEffortFrom(
        input.brainReasoningEffort ?? input.brainProvider?.reasoningEffort ?? env.AGI_GTA_BRAIN_REASONING_EFFORT,
        DEFAULT_BRAIN_REASONING_EFFORT
      ),
      maxTokens: completionTokenLimitFrom([input.brainMaxTokens, input.brainProvider?.maxTokens, env.AGI_GTA_BRAIN_MAX_TOKENS], REPORT_DEFAULTS.brainProvider.maxTokens),
      temperature: boundedNumberFrom(input.brainTemperature ?? input.brainProvider?.temperature ?? env.AGI_GTA_BRAIN_TEMPERATURE, DEFAULT_TEMPERATURE, MIN_TEMPERATURE, MAX_TEMPERATURE)
    },
    continuationHandoffPath: trivialPolicy ? "" : String(input.continuationHandoffPath ?? env.AGI_GTA_CONTINUATION_HANDOFF_PATH ?? "").trim(),
    maxTokens,
    ocrEnabled: localOcrLockedDisabled
      ? false
      : booleanFrom(input.ocrEnabled ?? env.AGI_GTA_OCR_ENABLED, true),
    ocrManagedService: booleanFrom(input.ocrManagedService ?? env.AGI_GTA_OCR_MANAGED_SERVICE, true),
    ocrUrl: String(input.ocrUrl ?? env.AGI_GTA_OCR_URL ?? "").trim(),
    ocrRois,
    ocrTimeoutMs: boundedIntegerFrom(input.ocrTimeoutMs ?? env.AGI_GTA_OCR_TIMEOUT_MS, 500, 1, 30000),
    ocrMinScore: boundedNumberFrom(input.ocrMinScore ?? env.AGI_GTA_OCR_MIN_SCORE, 0.5, 0, 1),
    ocrDevice,
    ocrPipeline: ocrPipelineFrom(input.ocrPipeline ?? env.AGI_GTA_OCR_PIPELINE, "recognizer"),
    ocrWarmup: booleanFrom(input.ocrWarmup ?? env.AGI_GTA_OCR_WARMUP, true),
    ocrDiffGate: booleanFrom(input.ocrDiffGate ?? env.AGI_GTA_OCR_DIFF_GATE, true),
    ocrDiffThreshold: boundedNumberFrom(input.ocrDiffThreshold ?? env.AGI_GTA_OCR_DIFF_THRESHOLD, 1.0, 0, 255),
    // Retry through detector+recognizer when recognizer-only returns an empty
    // result. This covers Paddle GPU runtimes that emit no rec_text values.
    ocrFallbackFull: booleanFrom(input.ocrFallbackFull ?? env.AGI_GTA_OCR_FALLBACK_FULL, true),
    ocrPython: String(input.ocrPython ?? env.AGI_GTA_OCR_PYTHON ?? "").trim(),
    ocrGpuVenv: String(input.ocrGpuVenv ?? env.AGI_GTA_OCR_GPU_VENV ?? "").trim(),
    ocrCpuThreads: boundedIntegerFrom(input.ocrCpuThreads ?? env.AGI_GTA_OCR_CPU_THREADS, 4, 1, 32),
    ocrDetLimitSideLen: boundedIntegerFrom(input.ocrDetLimitSideLen ?? env.AGI_GTA_OCR_DET_LIMIT_SIDE_LEN, 960, 64, 4096),
    ocrEnableMkldnn: booleanFrom(input.ocrEnableMkldnn ?? env.AGI_GTA_OCR_ENABLE_MKLDNN, false),
    ...require('./policyOcrConfig').policyOcrConfig({ ...input,
      policyRendering: trivialPolicy ? null : policyRendering }, env),
    taskResultOcrEnabled: localOcrLockedDisabled
      ? false
      : booleanFrom(input.taskResultOcrEnabled ?? env.AGI_GTA_TASK_RESULT_OCR_ENABLED, REPORT_DEFAULTS.taskResultOcrEnabled),
    taskResultOcrManagedService: booleanFrom(input.taskResultOcrManagedService ?? env.AGI_GTA_TASK_RESULT_OCR_MANAGED_SERVICE, true),
    taskResultOcrUrl: String(input.taskResultOcrUrl ?? env.AGI_GTA_TASK_RESULT_OCR_URL ?? "").trim(),
    taskResultOcrRois,
    taskResultOcrTimeoutMs: boundedIntegerFrom(input.taskResultOcrTimeoutMs ?? env.AGI_GTA_TASK_RESULT_OCR_TIMEOUT_MS, 3000, 1, 60000),
    taskResultOcrMinScore: boundedNumberFrom(input.taskResultOcrMinScore ?? env.AGI_GTA_TASK_RESULT_OCR_MIN_SCORE, 0.5, 0, 1),
    taskResultOcrDevice,
    taskResultOcrWarmup: booleanFrom(input.taskResultOcrWarmup ?? env.AGI_GTA_TASK_RESULT_OCR_WARMUP ?? input.ocrWarmup ?? env.AGI_GTA_OCR_WARMUP, true),
    taskResultOcrCpuThreads: boundedIntegerFrom(input.taskResultOcrCpuThreads ?? env.AGI_GTA_TASK_RESULT_OCR_CPU_THREADS ?? input.ocrCpuThreads ?? env.AGI_GTA_OCR_CPU_THREADS, 4, 1, 32),
    taskResultOcrDetLimitSideLen: boundedIntegerFrom(input.taskResultOcrDetLimitSideLen ?? env.AGI_GTA_TASK_RESULT_OCR_DET_LIMIT_SIDE_LEN, 1280, 64, 4096),
    taskResultOcrEnableMkldnn: booleanFrom(input.taskResultOcrEnableMkldnn ?? env.AGI_GTA_TASK_RESULT_OCR_ENABLE_MKLDNN ?? input.ocrEnableMkldnn ?? env.AGI_GTA_OCR_ENABLE_MKLDNN, false),
    taskResultOcrPython: String(input.taskResultOcrPython ?? env.AGI_GTA_TASK_RESULT_OCR_PYTHON ?? "").trim(),
    taskResultOcrGpuVenv: String(input.taskResultOcrGpuVenv ?? env.AGI_GTA_TASK_RESULT_OCR_GPU_VENV ?? "").trim(),
    taskResultAutoStopOnSuccess: booleanFrom(input.taskResultAutoStopOnSuccess ?? env.AGI_GTA_TASK_RESULT_AUTO_STOP_ON_SUCCESS, false),
    taskResultAutoStopOnFailure: booleanFrom(input.taskResultAutoStopOnFailure ?? env.AGI_GTA_TASK_RESULT_AUTO_STOP_ON_FAILURE, false)
  };
}

function publicConfig(config) {
  const trivialPolicy = config.policyMode === 'trivial';
  const policyStreaming = config.policyStreaming ?? (!trivialPolicy && REPORT_DEFAULTS.policyStreaming);
  const policyRendering = config.policyRendering !== undefined ? config.policyRendering
    : trivialPolicy ? null : REPORT_DEFAULTS.policyRendering;
  return {
    agentMode: DEFAULT_AGENT_MODE,
    policyMode: config.policyMode || "brain_policy",
    inferenceBackend: DEFAULT_INFERENCE_BACKEND,
    remoteServiceType: remoteServiceTypeFrom(config.remoteServiceType),
    policyTransport: config.policyTransport || (trivialPolicy ? (config.remoteServiceType === "lumine" ? "lumine_session" : "stateless") : REPORT_DEFAULTS.policyTransport),
    policyExecutionAcknowledgements: Boolean(config.policyExecutionAcknowledgements),
    policyStreaming: Boolean(policyStreaming), hfActionStartDelayMs: config.hfActionStartDelayMs ?? 200,
    hfSubmissionMaxLatenessMs: config.hfSubmissionMaxLatenessMs ?? (policyStreaming ? 100 : 0),
    hfValidActionTimeoutMs: config.hfValidActionTimeoutMs ?? 1000,
    hfWatchdogAutoResume: Boolean(config.hfWatchdogAutoResume ?? policyStreaming),
    policyHotSwapEnabled: Boolean(config.policyHotSwapEnabled),
    policyStandbyBaseUrl: config.policyStandbyBaseUrl || '',
    policyStandbyApiKeySet: Boolean(config.policyStandbyApiKey),
    policyHandoffPrepareTimeoutMs: config.policyHandoffPrepareTimeoutMs ?? 15000,
    policyHandoffMaxObservationAgeMs: config.policyHandoffMaxObservationAgeMs ?? 400,
    trivialOcrHistoryResetMs: config.trivialOcrHistoryResetMs ?? 10000,
    // Preserve the rendering contract without recording private provider settings.
    policyRendering: policyRendering ? Object.fromEntries(
      ['version', 'response_format', 'template_sha256', 'generation_prefix']
        .filter(key => policyRendering[key] != null)
        .map(key => [key, policyRendering[key]])
    ) : null,
    baseUrl: config.baseUrl,
    apiFormat: config.apiFormat || "chat_completions",
    modelName: config.modelName,
    displayId: config.displayId,
    resolution: config.resolution,
    maxSteps: config.maxSteps ?? config.maxActions,
    maxActions: config.maxSteps ?? config.maxActions,
    maxStepsUnlimited: Boolean(config.maxStepsUnlimited ?? config.maxActionsUnlimited ?? REPORT_DEFAULTS.maxStepsUnlimited),
    runDurationMs: config.runDurationMs ?? REPORT_DEFAULTS.runDurationMs,
    maxActionsUnlimited: Boolean(config.maxStepsUnlimited ?? config.maxActionsUnlimited ?? REPORT_DEFAULTS.maxStepsUnlimited),
    autoRefreshSessionOnMaxSteps: Boolean(config.autoRefreshSessionOnMaxSteps),
    allowThinking: Boolean(config.allowThinking ?? (!trivialPolicy && REPORT_DEFAULTS.allowThinking)),
    reasoningEffort: reasoningEffortFrom(config.reasoningEffort, DEFAULT_POLICY_REASONING_EFFORT),
    planEvery: config.planEvery ?? 0,
    planHorizon: PLAN_HORIZON,
    hfPackageDurationMs: config.hfPackageDurationMs ?? REPORT_DEFAULTS.hfPackageDurationMs,
    policyWindowMaxRounds: config.policyWindowMaxRounds ?? 16,
    policyWindowKeepRounds: config.policyWindowKeepRounds ?? 8,
    policyIncludeMinimap: Boolean(config.policyIncludeMinimap),
    mobilityRecovery: config.mobilityRecovery,
    policyManualSwitch: config.policyManualSwitch?.enabled ? {
      ...config.policyManualSwitch, targets: Object.fromEntries(Object.entries(config.policyManualSwitch.targets)
        .map(([name, target]) => { const { apiKey, ...safe } = target; return [name, { ...safe, apiKeySet: Boolean(apiKey) }]; }))
    } : { enabled: false },
    brainMaxOverrides: config.brainMaxOverrides ?? 3,
    brainMaxWaitMs: config.brainMaxWaitMs ?? 15000,
    policyMaxInvalidRetries: config.policyMaxInvalidRetries ?? DEFAULT_POLICY_MAX_INVALID_RETRIES,
    hfScheduling: config.hfScheduling ?? (trivialPolicy ? 'auto' : REPORT_DEFAULTS.hfScheduling),
    hfSubmissionOffsetMs: config.hfSubmissionOffsetMs ?? (trivialPolicy ? 0 : REPORT_DEFAULTS.hfSubmissionOffsetMs),
    hfCaptureIntervalMs: config.hfCaptureIntervalMs ?? Math.round(1000 / (config.hfCaptureHz || DEFAULT_HF_CAPTURE_HZ)),
    hfCaptureHz: config.hfCaptureHz ?? DEFAULT_HF_CAPTURE_HZ,
    hfChunksPerPackage: config.hfChunksPerPackage ?? DEFAULT_HF_CHUNKS_PER_PACKAGE,
    hfChunkDurationMs: config.hfChunkDurationMs ?? DEFAULT_HF_CHUNK_DURATION_MS,
    hfPackageEndHoldMs: config.hfPackageEndHoldMs ?? DEFAULT_HF_PACKAGE_END_HOLD_MS,
    hfLongHoldThresholdChunks: config.hfLongHoldThresholdChunks ?? DEFAULT_HF_LONG_HOLD_THRESHOLD_CHUNKS,
    hfLongHoldReleaseMs: config.hfLongHoldReleaseMs ?? DEFAULT_HF_LONG_HOLD_RELEASE_MS,
    hfAllowBarePackage: Boolean(config.hfAllowBarePackage),
    hfMaxPendingPackages: config.hfMaxPendingPackages ?? DEFAULT_HF_MAX_PENDING_PACKAGES,
    hfContextObservations: config.hfContextObservations ?? DEFAULT_HF_CONTEXT_OBSERVATIONS,
    hfRemoteMaxOcrTokens: config.hfRemoteMaxOcrTokens ?? DEFAULT_HF_REMOTE_MAX_OCR_TOKENS,
    hfAutoRepairPackages: Boolean(config.hfAutoRepairPackages),
    hfCaptureArchiveHz: config.hfCaptureArchiveHz ?? DEFAULT_HF_CAPTURE_ARCHIVE_HZ,
    hfCaptureArchiveFormat: config.hfCaptureArchiveFormat ?? DEFAULT_HF_CAPTURE_ARCHIVE_FORMAT,
    hfCaptureArchiveQuality: config.hfCaptureArchiveQuality ?? DEFAULT_HF_CAPTURE_ARCHIVE_QUALITY,
    hfCaptureScratchSeconds: config.hfCaptureScratchSeconds ?? DEFAULT_HF_CAPTURE_SCRATCH_SECONDS,
    hfCaptureMaxGb: config.hfCaptureMaxGb ?? DEFAULT_HF_CAPTURE_MAX_GB,
    hfCaptureSaveOnEvent: Boolean(config.hfCaptureSaveOnEvent),
    sessionOutputSubpath: normalizeSessionOutputSubpathForPublic(config.sessionOutputSubpath),
    temperature: config.temperature,
    modelWaitResumeDelayMs: config.modelWaitResumeDelayMs ?? DEFAULT_MODEL_WAIT_RESUME_DELAY_MS,
    resumeVerificationTailMs: config.resumeVerificationTailMs ?? 200,
    hfCarryEnabled: Boolean(config.hfCarryEnabled),
    hfCarryMaxMs: config.hfCarryMaxMs ?? DEFAULT_HF_CARRY_MAX_MS,
    navigationMinimap: {
      enabled: Boolean(config.navigationMinimap?.enabled),
      crop: minimapCropFrom(config.navigationMinimap?.crop)
    },
    brainScheduler: {
      minNewActionsBeforeStart: config.brainScheduler?.minNewActionsBeforeStart ?? DEFAULT_BRAIN_MIN_NEW_ACTIONS_BEFORE_START,
      simulatedLagActions: config.brainScheduler?.simulatedLagActions ?? DEFAULT_BRAIN_SIMULATED_LAG_ACTIONS,
      maxInFlight: DEFAULT_BRAIN_MAX_IN_FLIGHT
      ,
      maxActionsWithoutMemoryUpdate: config.brainScheduler?.maxActionsWithoutMemoryUpdate ?? DEFAULT_BRAIN_MAX_ACTIONS_WITHOUT_MEMORY_UPDATE,
      maxActionsWithoutInstructionUpdate: config.brainScheduler?.maxActionsWithoutInstructionUpdate ?? DEFAULT_BRAIN_MAX_ACTIONS_WITHOUT_INSTRUCTION_UPDATE,
      memoryFallbackIntervalMs: config.brainScheduler?.memoryFallbackIntervalMs ?? REPORT_DEFAULTS.brainScheduler.memoryFallbackIntervalMs,
      instructionFallbackEnabled: config.brainScheduler?.instructionFallbackEnabled ?? REPORT_DEFAULTS.brainScheduler.instructionFallbackEnabled
    },
    brainFallbackScope: config.brainFallbackScope || 'memory_think',
    brainTrajectoryObservationLimit: config.brainTrajectoryObservationLimit ?? DEFAULT_BRAIN_TRAJECTORY_OBSERVATION_LIMIT,
    brainPrefillEnabled: config.brainPrefillEnabled === true,
    brainPrefillIntervalMs: config.brainPrefillIntervalMs ?? 1000,
    brainPrefillTimeoutMs: config.brainPrefillTimeoutMs ?? 10000,
    policyRequestTimeoutMs: config.policyRequestTimeoutMs ?? 120000,
    hfPauseAllBrainUpdates: Boolean(config.hfPauseAllBrainUpdates),
    brainMaxRetries: config.brainMaxRetries ?? 1,
    brainRetryDelayMs: config.brainRetryDelayMs ?? 500,
    brainRequestTimeoutMs: config.brainRequestTimeoutMs ?? 0,
    brainProvider: {
      inheritPolicy: config.brainProvider?.inheritPolicy ?? REPORT_DEFAULTS.brainProvider.inheritPolicy,
      apiFormat: config.brainProvider?.apiFormat || "chat_completions",
      baseUrl: String(config.brainProvider?.baseUrl || ""),
      modelName: String(config.brainProvider?.modelName || ""),
      allowThinking: config.brainProvider?.allowThinking !== false,
      reasoningEffort: reasoningEffortFrom(config.brainProvider?.reasoningEffort, DEFAULT_BRAIN_REASONING_EFFORT),
      maxTokens: config.brainProvider?.maxTokens === undefined ? REPORT_DEFAULTS.brainProvider.maxTokens : config.brainProvider.maxTokens,
      temperature: config.brainProvider?.temperature ?? DEFAULT_TEMPERATURE,
      apiKeySet: Boolean(config.brainProvider?.apiKey)
    },
    continuationHandoffPath: String(config.continuationHandoffPath || ""),
    maxTokens: config.maxTokens === undefined ? DEFAULT_MAX_TOKENS : config.maxTokens,
    ocrEnabled: Boolean(config.ocrEnabled),
    policyOcrProfile: config.policyOcrProfile || 'auto',
    effectivePolicyOcrProfile: config.effectivePolicyOcrProfile,
    policyOcrUrl: config.policyOcrUrl,
    policyOcrManagedService: config.policyOcrManagedService,
    ocrManagedService: Boolean(config.ocrManagedService),
    ocrUrl: config.ocrUrl,
    ocrRois: Array.isArray(config.ocrRois) ? config.ocrRois : DEFAULT_OCR_ROIS.slice(),
    ocrTimeoutMs: config.ocrTimeoutMs,
    ocrMinScore: config.ocrMinScore,
    ocrDevice: config.ocrDevice,
    ocrPipeline: config.ocrPipeline,
    ocrWarmup: Boolean(config.ocrWarmup),
    ocrDiffGate: Boolean(config.ocrDiffGate),
    ocrDiffThreshold: config.ocrDiffThreshold,
    ocrFallbackFull: Boolean(config.ocrFallbackFull),
    taskResultOcrEnabled: Boolean(config.taskResultOcrEnabled),
    taskResultOcrManagedService: Boolean(config.taskResultOcrManagedService),
    taskResultOcrUrl: config.taskResultOcrUrl,
    taskResultOcrRois: Array.isArray(config.taskResultOcrRois) ? config.taskResultOcrRois : DEFAULT_TASK_RESULT_OCR_ROIS.slice(),
    taskResultOcrTimeoutMs: config.taskResultOcrTimeoutMs,
    taskResultOcrMinScore: config.taskResultOcrMinScore,
    taskResultOcrDevice: config.taskResultOcrDevice,
    taskResultOcrWarmup: Boolean(config.taskResultOcrWarmup),
    taskResultOcrCpuThreads: config.taskResultOcrCpuThreads,
    taskResultOcrDetLimitSideLen: config.taskResultOcrDetLimitSideLen,
    taskResultOcrEnableMkldnn: Boolean(config.taskResultOcrEnableMkldnn),
    taskResultAutoStopOnSuccess: Boolean(config.taskResultAutoStopOnSuccess),
    taskResultAutoStopOnFailure: Boolean(config.taskResultAutoStopOnFailure),
    apiKeySet: Boolean(config.apiKey)
  };
}

function resolutionSize(resolution) {
  if (resolution === "1920x1080") {
    return { width: 1920, height: 1080 };
  }

  return { width: 1280, height: 720 };
}

module.exports = {
  INFERENCE_BACKENDS,
  REMOTE_SERVICE_TYPES,
  DEFAULT_REMOTE_SERVICE_TYPE,
  DEFAULT_INFERENCE_BACKEND,
  DEFAULT_AGENT_MODE,
  DEFAULT_POLICY_MAX_INVALID_RETRIES,
  DEFAULT_HF_CAPTURE_INTERVAL_MS,
  DEFAULT_HF_CAPTURE_HZ,
  DEFAULT_HF_CHUNKS_PER_PACKAGE,
  DEFAULT_HF_CHUNK_DURATION_MS,
  DEFAULT_HF_PACKAGE_END_HOLD_MS,
  DEFAULT_HF_LONG_HOLD_THRESHOLD_CHUNKS,
  DEFAULT_HF_LONG_HOLD_RELEASE_MS,
  DEFAULT_HF_MAX_PENDING_PACKAGES,
  DEFAULT_HF_REMOTE_MAX_OCR_TOKENS,
  DEFAULT_HF_CAPTURE_ARCHIVE_HZ,
  DEFAULT_HF_CAPTURE_ARCHIVE_FORMAT,
  DEFAULT_HF_CAPTURE_ARCHIVE_QUALITY,
  DEFAULT_HF_CAPTURE_SCRATCH_SECONDS,
  DEFAULT_HF_CAPTURE_MAX_GB,
  DEFAULT_HF_CAPTURE_SAVE_ON_EVENT,
  DEFAULT_MAX_TOKENS,
  DEFAULT_TEMPERATURE,
  MIN_TEMPERATURE,
  MAX_TEMPERATURE,
  DEFAULT_MODEL_WAIT_RESUME_DELAY_MS,
  DEFAULT_HF_CARRY_ENABLED,
  DEFAULT_HF_CARRY_MAX_MS,
  DEFAULT_BRAIN_MIN_NEW_ACTIONS_BEFORE_START,
  DEFAULT_BRAIN_SIMULATED_LAG_ACTIONS,
  DEFAULT_BRAIN_MAX_IN_FLIGHT,
  DEFAULT_BRAIN_TRAJECTORY_OBSERVATION_LIMIT,
  DEFAULT_BRAIN_MAX_ACTIONS_WITHOUT_MEMORY_UPDATE,
  DEFAULT_BRAIN_MAX_ACTIONS_WITHOUT_INSTRUCTION_UPDATE,
  DEFAULT_POLICY_REASONING_EFFORT,
  DEFAULT_BRAIN_REASONING_EFFORT,
  REASONING_EFFORTS,
  DEFAULT_OCR_ROIS,
  DEFAULT_TASK_RESULT_OCR_ROIS,
  MAX_COMPLETION_TOKENS,
  PLAN_HORIZON,
  RESOLUTIONS,
  sanitizeConfig,
  publicConfig,
  resolutionSize
};

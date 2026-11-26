const { detectTaskResult: defaultDetectTaskResult } = require("./taskResultOcrClient");

const TASK_RESULT_AUTO_STOP_GRACE_MS = 10000;

function taskResultAutoStopEnabled(config = {}) {
  return Boolean(config.taskResultOcrEnabled)
    && (Boolean(config.taskResultAutoStopOnSuccess) || Boolean(config.taskResultAutoStopOnFailure));
}

function shouldStopForTaskResult(config = {}, result = {}) {
  return (result.signal === "success" && Boolean(config.taskResultAutoStopOnSuccess))
    || (result.signal === "failure" && Boolean(config.taskResultAutoStopOnFailure));
}

function compactText(value, limit = 240) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length <= limit ? text : `${text.slice(0, limit - 3)}...`;
}

function compactResult(result = {}) {
  return {
    ok: Boolean(result.ok),
    skipped: Boolean(result.skipped),
    signal: result.signal || "none",
    state: result.state || result.signal || "none",
    terminal: Boolean(result.terminal),
    success: Boolean(result.success),
    failure: Boolean(result.failure),
    latencyMs: result.latencyMs ?? null,
    requestLatencyMs: result.requestLatencyMs ?? null,
    imagePath: result.imagePath || "",
    capturedAt: result.capturedAt || "",
    rois: Array.isArray(result.rois) ? result.rois : [],
    text: compactText(result.text || "", 500),
    score: result.score ?? null,
    matches: result.matches || { success: [], failure: [] },
    error: compactText(result.error || "", 240)
  };
}

function timestampMs(value, fallback) {
  const parsed = Date.parse(value || "");
  return Number.isFinite(parsed) ? parsed : fallback;
}

class TaskResultMonitor {
  constructor(options = {}) {
    this.now = typeof options.now === "function" ? options.now : () => Date.now();
    this.getConfig = typeof options.getConfig === "function" ? options.getConfig : () => ({});
    this.onTerminal = typeof options.onTerminal === "function" ? options.onTerminal : () => {};
    this.onResult = typeof options.onResult === "function" ? options.onResult : () => {};
    this.onLog = typeof options.onLog === "function" ? options.onLog : () => {};
    this.onStatus = typeof options.onStatus === "function" ? options.onStatus : () => {};
    this.detectTaskResult = typeof options.detectTaskResult === "function" ? options.detectTaskResult : defaultDetectTaskResult;
    this.isEnabled = typeof options.isEnabled === "function"
      ? options.isEnabled
      : (config) => taskResultAutoStopEnabled(config);
    this.inFlight = false;
    this.pending = null;
    this.sequence = 0;
    this.lastResult = null;
    this.firstTerminal = null;
    this.lastError = "";
    this.lastObservationAt = null;
    this.startupGraceUntilMs = this.now() + TASK_RESULT_AUTO_STOP_GRACE_MS;
    this.startupGraceUntil = new Date(this.startupGraceUntilMs).toISOString();
    this.suppressedObservations = 0;
    this.lastSuppressedAt = null;
  }

  reset() {
    this.pending = null;
    this.sequence = 0;
    this.lastResult = null;
    this.firstTerminal = null;
    this.lastError = "";
    this.lastObservationAt = null;
    this.startupGraceUntilMs = this.now() + TASK_RESULT_AUTO_STOP_GRACE_MS;
    this.startupGraceUntil = new Date(this.startupGraceUntilMs).toISOString();
    this.suppressedObservations = 0;
    this.lastSuppressedAt = null;
    this.emitStatus();
  }

  getStatus() {
    const config = this.getConfig();
    return {
      enabled: Boolean(this.isEnabled(config)),
      autoStopOnSuccess: Boolean(config.taskResultAutoStopOnSuccess),
      autoStopOnFailure: Boolean(config.taskResultAutoStopOnFailure),
      inFlight: this.inFlight,
      pending: Boolean(this.pending),
      sequence: this.sequence,
      lastObservationAt: this.lastObservationAt,
      lastResult: this.lastResult,
      firstTerminal: this.firstTerminal,
      lastError: this.lastError,
      startupGraceMs: TASK_RESULT_AUTO_STOP_GRACE_MS,
      startupGraceUntil: this.startupGraceUntil,
      startupGraceActive: this.now() < this.startupGraceUntilMs,
      suppressedObservations: this.suppressedObservations,
      lastSuppressedAt: this.lastSuppressedAt
    };
  }

  emitStatus() {
    this.onStatus(this.getStatus());
  }

  observe(observation = {}, metadata = {}) {
    const config = this.getConfig();
    if (!this.isEnabled(config)) {
      return false;
    }

    const observedAtMs = timestampMs(observation.capturedAt, this.now());
    if (observedAtMs < this.startupGraceUntilMs) {
      this.suppressedObservations += 1;
      this.lastObservationAt = observation.capturedAt || new Date(observedAtMs).toISOString();
      this.lastSuppressedAt = this.lastObservationAt;
      this.emitStatus();
      return false;
    }

    const imagePath = observation.ocrImagePath || observation.imagePath;
    const imageBuffer = Buffer.isBuffer(observation.buffer) ? observation.buffer : null;
    if (!imagePath && !imageBuffer) {
      this.lastError = "task-result OCR requires a screenshot image buffer or saved image path";
      this.onLog(this.lastError);
      this.emitStatus();
      return false;
    }

    const sequence = this.sequence + 1;
    this.sequence = sequence;
    this.lastObservationAt = observation.capturedAt || new Date(observedAtMs).toISOString();
    this.pending = {
      sequence,
      observation: {
        imagePath,
        buffer: imageBuffer,
        mimeType: observation.mimeType || "image/jpeg",
        capturedAt: observation.capturedAt
      },
      metadata
    };
    this.emitStatus();
    void this.pump();
    return true;
  }

  async pump() {
    if (this.inFlight || !this.pending) {
      return;
    }

    const item = this.pending;
    this.pending = null;
    this.inFlight = true;
    this.emitStatus();

    try {
      const config = this.getConfig();
      if (!this.isEnabled(config)) {
        return;
      }

      const result = await this.detectTaskResult(config, item.observation);
      const compact = compactResult({
        ...result,
        capturedAt: item.observation.capturedAt
      });
      this.lastResult = {
        ...compact,
        sequence: item.sequence,
        checkedAt: new Date().toISOString()
      };
      this.lastError = compact.error || "";
      if (!this.firstTerminal && result.ok && !result.skipped && ['success', 'failure'].includes(result.signal)) {
        this.firstTerminal = { ...this.lastResult };
        await this.onTerminal(result, item);
      }
      await this.onResult(result, item);
    } catch (error) {
      this.lastError = compactText(error.message || error, 240);
      this.onLog(`task-result OCR unavailable: ${this.lastError}`);
    } finally {
      this.inFlight = false;
      this.emitStatus();
      if (this.pending) {
        void this.pump();
      }
    }
  }

  async waitForIdle() {
    while (this.inFlight || this.pending) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  }
}

module.exports = {
  TaskResultMonitor,
  TASK_RESULT_AUTO_STOP_GRACE_MS,
  shouldStopForTaskResult,
  taskResultAutoStopEnabled
};

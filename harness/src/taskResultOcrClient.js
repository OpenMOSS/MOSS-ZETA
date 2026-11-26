const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { performance } = require("node:perf_hooks");

const DEFAULT_TASK_RESULT_OCR_ROIS = ["full_screen"];
const WINDOWS_OCR_PATH_LIMIT = 240;

function isLocalTaskResultOcrUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:"
      && (parsed.hostname === "127.0.0.1" || parsed.hostname === "localhost" || parsed.hostname === "::1");
  } catch {
    return false;
  }
}

async function stageImagePathForTaskResultOcr(imagePath, url) {
  if (process.platform !== "win32"
    || imagePath.length < WINDOWS_OCR_PATH_LIMIT
    || !isLocalTaskResultOcrUrl(url)) {
    return { imagePath, cleanup: null };
  }

  const stagingDirectory = await fs.mkdtemp(path.join(os.tmpdir(), "bvalanche-task-result-ocr-"));
  const extension = path.extname(imagePath).toLowerCase() === ".png" ? ".png" : ".jpg";
  const stagedImagePath = path.join(stagingDirectory, `image${extension}`);
  await fs.copyFile(imagePath, stagedImagePath);

  return {
    imagePath: stagedImagePath,
    cleanup: async () => {
      await fs.rm(stagingDirectory, { recursive: true, force: true });
    }
  };
}

function elapsedMs(startedAt) {
  return Math.round((performance.now() - startedAt) * 100) / 100;
}

function compactText(value, limit = 500) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (text.length <= limit) {
    return text;
  }

  return `${text.slice(0, Math.max(0, limit - 3))}...`;
}

function numericOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function taskResultRoisFromConfig(config = {}, options = {}) {
  const source = options.rois ?? config.taskResultOcrRois ?? DEFAULT_TASK_RESULT_OCR_ROIS;
  const raw = Array.isArray(source) ? source : String(source || "").split(",");
  const rois = raw.map((item) => String(item || "").trim()).filter(Boolean);
  return rois.length > 0 ? rois : DEFAULT_TASK_RESULT_OCR_ROIS.slice();
}

function taskResultOcrEnabled(config = {}, options = {}) {
  if (Object.prototype.hasOwnProperty.call(options, "enabled")) {
    return Boolean(options.enabled);
  }
  if (Object.prototype.hasOwnProperty.call(config, "taskResultOcrEnabled")) {
    return Boolean(config.taskResultOcrEnabled);
  }
  return true;
}

function skippedResult({ rois = DEFAULT_TASK_RESULT_OCR_ROIS } = {}) {
  return {
    ok: false,
    skipped: true,
    signal: "none",
    state: "none",
    success: false,
    failure: false,
    terminal: false,
    latencyMs: 0,
    requestLatencyMs: 0,
    rois,
    text: "",
    results: [],
    error: ""
  };
}

function failureResult({ startedAt, rois, error }) {
  const latencyMs = elapsedMs(startedAt);
  return {
    ok: false,
    skipped: false,
    signal: "none",
    state: "none",
    success: false,
    failure: false,
    terminal: false,
    latencyMs,
    requestLatencyMs: latencyMs,
    rois,
    text: "",
    results: [],
    error: compactText(error?.message || error || "task-result OCR request failed", 220)
  };
}

function normalizeLine(line = {}) {
  return {
    text: compactText(line.text || "", 400),
    score: numericOrNull(line.score),
    box: line.box ?? null
  };
}

function normalizeResult(result = {}) {
  const lines = Array.isArray(result.lines)
    ? result.lines.map(normalizeLine).filter((line) => line.text)
    : [];
  return {
    roi: String(result.roi || ""),
    box: Array.isArray(result.box) ? result.box : null,
    text: compactText(result.text || lines.map((line) => line.text).join(" "), 800),
    score: numericOrNull(result.score),
    latencyMs: numericOrNull(result.latencyMs),
    pipeline: result.pipeline ? String(result.pipeline) : "full",
    lines
  };
}

function normalizeSignal(value) {
  const signal = String(value || "none").trim().toLowerCase();
  return ["success", "failure", "none", "ambiguous"].includes(signal) ? signal : "none";
}

function normalizeResponse(body, { startedAt, rois }) {
  const requestLatencyMs = elapsedMs(startedAt);
  const results = Array.isArray(body?.results)
    ? body.results.map(normalizeResult).filter((result) => result.roi)
    : [];
  const signal = normalizeSignal(body?.signal || body?.state);

  return {
    ok: Boolean(body?.ok),
    skipped: false,
    signal,
    state: signal,
    success: Boolean(body?.success),
    failure: Boolean(body?.failure),
    terminal: signal === "success" || signal === "failure",
    latencyMs: numericOrNull(body?.latencyMs) ?? requestLatencyMs,
    requestLatencyMs,
    imagePath: body?.imagePath ? String(body.imagePath) : "",
    width: Number.isFinite(Number(body?.width)) ? Math.trunc(Number(body.width)) : null,
    height: Number.isFinite(Number(body?.height)) ? Math.trunc(Number(body.height)) : null,
    rois: Array.isArray(body?.rois) ? body.rois.map(String) : rois,
    text: compactText(body?.text || results.map((result) => result.text).join(" "), 1200),
    score: numericOrNull(body?.score),
    matches: {
      success: Array.isArray(body?.matches?.success) ? body.matches.success.map(String) : [],
      failure: Array.isArray(body?.matches?.failure) ? body.matches.failure.map(String) : []
    },
    results,
    error: compactText(body?.error || "", 220)
  };
}

async function fetchJsonWithTimeout(url, payload, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => {
    controller.abort(new Error(`task-result OCR request timed out after ${timeoutMs} ms`));
  }, timeoutMs);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json"
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    const text = await response.text();
    const body = text ? JSON.parse(text) : null;

    if (!response.ok) {
      throw new Error(body?.error || response.statusText || `HTTP ${response.status}`);
    }

    return body;
  } finally {
    clearTimeout(timeout);
  }
}

async function detectTaskResult(config = {}, observation = {}, options = {}) {
  const rois = taskResultRoisFromConfig(config, options);
  if (!taskResultOcrEnabled(config, options)) {
    return skippedResult({ rois });
  }

  const startedAt = performance.now();
  const imagePath = options.imagePath || observation?.ocrImagePath || observation?.imagePath;
  const imageBuffer = Buffer.isBuffer(options.buffer)
    ? options.buffer
    : Buffer.isBuffer(observation?.buffer)
      ? observation.buffer
      : null;
  if (!imagePath && !imageBuffer) {
    return failureResult({
      startedAt,
      rois,
      error: "task-result OCR requires a screenshot image buffer or saved image path"
    });
  }

  const url = String(options.url ?? config.taskResultOcrUrl ?? '').trim();
  if (!url) return failureResult({ startedAt, rois,
    error: 'Set taskResultOcrUrl / AGI_GTA_TASK_RESULT_OCR_URL; task-result OCR has no default URL.' });
  let stagedImage = null;
  try {
    const payload = {
      rois,
      minScore: options.minScore ?? config.taskResultOcrMinScore
    };
    if (imageBuffer && isLocalTaskResultOcrUrl(url)) {
      payload.imageData = `data:${observation?.mimeType || "image/jpeg"};base64,${imageBuffer.toString("base64")}`;
    } else {
      stagedImage = await stageImagePathForTaskResultOcr(imagePath, url);
      payload.imagePath = stagedImage.imagePath;
    }
    const body = await fetchJsonWithTimeout(
      url,
      payload,
      options.timeoutMs || config.taskResultOcrTimeoutMs || 3000
    );

    const normalized = normalizeResponse(body, { startedAt, rois });
    if (!normalized.ok && !normalized.error) {
      normalized.error = "task-result OCR service returned ok=false";
    }
    return normalized;
  } catch (error) {
    return failureResult({ startedAt, rois, error });
  } finally {
    if (stagedImage?.cleanup) {
      await stagedImage.cleanup().catch(() => {});
    }
  }
}

function detectTaskResultInBackground(config = {}, observation = {}, options = {}) {
  const onResult = typeof options.onResult === "function" ? options.onResult : null;
  const onError = typeof options.onError === "function" ? options.onError : null;
  const promise = detectTaskResult(config, observation, options)
    .then((result) => {
      if (onResult) {
        onResult(result);
      }
      return result;
    })
    .catch((error) => {
      if (onError) {
        onError(error);
      }
      return failureResult({
        startedAt: performance.now(),
        rois: taskResultRoisFromConfig(config, options),
        error
      });
    });

  return promise;
}

module.exports = {
  DEFAULT_TASK_RESULT_OCR_ROIS,
  detectTaskResult,
  detectTaskResultInBackground,
  isLocalTaskResultOcrUrl,
  stageImagePathForTaskResultOcr,
  taskResultRoisFromConfig
};

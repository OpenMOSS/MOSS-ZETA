const { performance } = require("node:perf_hooks");

function elapsedMs(startedAt) {
  return Math.round((performance.now() - startedAt) * 100) / 100;
}

function compactText(value, limit = 260) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (text.length <= limit) {
    return text;
  }

  return `${text.slice(0, Math.max(0, limit - 3))}...`;
}

function roisFromConfig(config = {}) {
  if (Array.isArray(config.ocrRois)) {
    return config.ocrRois.map((item) => String(item || "").trim()).filter(Boolean);
  }

  return String(config.ocrRois || "subtitle_tight")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function isLocalOcrUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "http:"
      && ["127.0.0.1", "localhost", "::1"].includes(url.hostname);
  } catch {
    return false;
  }
}

function disabledResult() {
  return {
    ok: false,
    skipped: true,
    latencyMs: 0,
    requestLatencyMs: 0,
    rois: [],
    results: [],
    error: ""
  };
}

function failureResult({ startedAt, rois, error }) {
  const latencyMs = elapsedMs(startedAt);
  return {
    ok: false,
    skipped: false,
    latencyMs,
    requestLatencyMs: latencyMs,
    rois,
    results: [],
    error: compactText(error?.message || error || "OCR request failed", 180)
  };
}

function numericOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function averageScore(lines) {
  const scores = lines
    .map((line) => numericOrNull(line.score))
    .filter((score) => score !== null);
  if (scores.length === 0) {
    return null;
  }

  return Math.round((scores.reduce((sum, score) => sum + score, 0) / scores.length) * 1000) / 1000;
}

function normalizeLine(line) {
  return {
    text: compactText(line?.text || "", 400),
    score: numericOrNull(line?.score),
    box: line?.box ?? null
  };
}

function normalizeResult(result) {
  const lines = Array.isArray(result?.lines) ? result.lines.map(normalizeLine).filter((line) => line.text) : [];
  const text = compactText(result?.text || lines.map((line) => line.text).join(" "), 600);
  return {
    roi: String(result?.roi || ""),
    box: Array.isArray(result?.box) ? result.box : null,
    text,
    score: numericOrNull(result?.score) ?? averageScore(lines),
    latencyMs: numericOrNull(result?.latencyMs),
    pipeline: result?.pipeline ? String(result.pipeline) : "",
    cached: Boolean(result?.cached),
    diffScore: numericOrNull(result?.diffScore),
    diffMs: numericOrNull(result?.diffMs),
    proposalMs: numericOrNull(result?.proposalMs),
    recognitionMs: numericOrNull(result?.recognitionMs),
    proposalCount: Number.isFinite(Number(result?.proposalCount)) ? Math.trunc(Number(result.proposalCount)) : null,
    fallback: Boolean(result?.fallback),
    lines
  };
}

function normalizeResponse(body, { startedAt, rois }) {
  const requestLatencyMs = elapsedMs(startedAt);
  const results = Array.isArray(body?.results)
    ? body.results.map(normalizeResult).filter((result) => result.roi)
    : [];

  return {
    ok: Boolean(body?.ok),
    ...(body?.policyFields ? { policyFields: body.policyFields, policyOcrAudit: body.policyOcrAudit } : {}),
    skipped: false,
    latencyMs: numericOrNull(body?.latencyMs) ?? requestLatencyMs,
    requestLatencyMs,
    rois: Array.isArray(body?.rois) ? body.rois.map(String) : rois,
    results,
    error: compactText(body?.error || "", 180)
  };
}

async function fetchJsonWithTimeout(url, payload, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => {
    controller.abort(new Error(`OCR request timed out after ${timeoutMs} ms`));
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
    let body = null;
    if (text) {
      body = JSON.parse(text);
    }

    if (!response.ok) {
      throw new Error(body?.error || response.statusText || `HTTP ${response.status}`);
    }

    return body;
  } finally {
    clearTimeout(timeout);
  }
}

async function recognizeScreenshotText(config = {}, observation = {}) {
  if (!config.ocrEnabled) {
    return disabledResult();
  }

  const rois = roisFromConfig(config);
  const startedAt = performance.now();
  const imagePath = observation?.ocrImagePath || observation?.imagePath;
  const imageBuffer = Buffer.isBuffer(observation?.buffer) ? observation.buffer : null;
  if (!imagePath && !imageBuffer) {
    return failureResult({
      startedAt,
      rois,
      error: "OCR requires a screenshot image buffer or saved image path"
    });
  }

  try {
    if (!String(config.ocrUrl || '').trim()) {
      const setting = require('./policyOcrConfig').isTrainingPolicyOcr(config)
        ? 'policyOcrUrl / AGI_GTA_POLICY_OCR_URL' : 'ocrUrl / AGI_GTA_OCR_URL';
      throw new Error(`Set ${setting}; OCR has no default URL.`);
    }
    const payload = {
      rois,
      minScore: config.ocrMinScore
    };
    if (imageBuffer && isLocalOcrUrl(config.ocrUrl)) {
      payload.imageData = `data:${observation.mimeType || "image/jpeg"};base64,${imageBuffer.toString("base64")}`;
    } else {
      payload.imagePath = imagePath;
    }
    let body = await fetchJsonWithTimeout(config.ocrUrl, payload, config.ocrTimeoutMs || 500);
    if (body?.ok && require('./policyOcrConfig').isTrainingPolicyOcr(config)) {
      // Selection must precede legacy display-text truncation and must see all
      // regions so that training's first-region ownership remains identical.
      body = require('./trainingPolicyOcr').adaptTrainingOcrResult(body);
    }

    const normalized = normalizeResponse(body, { startedAt, rois });
    if (!normalized.ok && !normalized.error) {
      normalized.error = "OCR service returned ok=false";
    }
    return normalized;
  } catch (error) {
    return failureResult({ startedAt, rois, error });
  }
}

function formatScore(score) {
  const number = numericOrNull(score);
  return number === null ? "" : ` (score ${number.toFixed(3)})`;
}

function formatOcrForObserve(ocr, options = {}) {
  if (!ocr || ocr.skipped) {
    return "";
  }

  if (!ocr.ok) {
    return options.includeFailures && ocr.error ? `OCR unavailable: ${compactText(ocr.error, 160)}` : "";
  }

  const rois = Array.isArray(ocr.rois) && ocr.rois.length > 0
    ? ocr.rois
    : [...new Set((ocr.results || []).map((result) => result.roi).filter(Boolean))];
  if (rois.length === 0) {
    return "";
  }

  const resultByRoi = new Map((ocr.results || []).map((result) => [result.roi, result]));
  const lines = ["OCR:"];
  for (const roi of rois) {
    const result = resultByRoi.get(roi);
    const text = compactText(result?.text || "", 320);
    lines.push(text ? `${roi}: ${text}${formatScore(result?.score)}` : `${roi}: none`);
  }

  return lines.join("\n");
}

const SCREEN_OCR_SLOT_BY_ROI = Object.freeze({
  top_prompt: "P",
  hud_left_bottom: "L",
  subtitle_tight: "B",
  subtitle: "B"
});
const SCREEN_OCR_SLOT_ORDER = ["P", "U", "L", "B", "R", "C"];

function formatOcrForScreenOcr(ocr, previousState = null) {
  if (!ocr || ocr.skipped || !ocr.ok) {
    return {
      text: "",
      state: previousState || null,
      changed: false
    };
  }

  const rois = Array.isArray(ocr.rois) && ocr.rois.length > 0
    ? ocr.rois
    : [...new Set((ocr.results || []).map((result) => result.roi).filter(Boolean))];
  const resultByRoi = new Map((ocr.results || []).map((result) => [result.roi, result]));
  const observedSlots = new Set();
  const valuesBySlot = new Map();

  for (const roi of rois) {
    const slot = SCREEN_OCR_SLOT_BY_ROI[String(roi || "").trim()];
    if (!slot) {
      continue;
    }
    observedSlots.add(slot);
    const text = compactText(resultByRoi.get(roi)?.text || "", 320);
    if (!text) {
      continue;
    }
    const values = valuesBySlot.get(slot) || [];
    if (!values.includes(text)) {
      values.push(text);
    }
    valuesBySlot.set(slot, values);
  }

  if (observedSlots.size === 0) {
    return {
      text: "",
      state: previousState || null,
      changed: false
    };
  }

  const nextState = { ...(previousState || {}) };
  const changedSlots = [];
  for (const slot of SCREEN_OCR_SLOT_ORDER) {
    if (!observedSlots.has(slot)) {
      continue;
    }
    const nextValue = (valuesBySlot.get(slot) || []).join("|");
    const previousValue = String(previousState?.[slot] || "");
    if (nextValue === previousValue) {
      continue;
    }
    if (nextValue) {
      nextState[slot] = nextValue;
    } else {
      delete nextState[slot];
    }
    changedSlots.push([slot, nextValue]);
  }

  if (changedSlots.length === 0) {
    return {
      text: "",
      state: nextState,
      changed: false
    };
  }

  const isInitialState = !previousState;
  const lines = [isInitialState ? "S" : "D"];
  if (isInitialState) {
    for (const slot of SCREEN_OCR_SLOT_ORDER) {
      if (nextState[slot]) {
        lines.push(`${slot}=${nextState[slot]}`);
      }
    }
  } else {
    for (const [slot, value] of changedSlots) {
      lines.push(`${slot}=${value}`);
    }
  }

  return {
    text: lines.join("\n"),
    state: nextState,
    changed: true
  };
}

module.exports = {
  formatOcrForObserve,
  formatOcrForScreenOcr,
  isLocalOcrUrl,
  recognizeScreenshotText,
  roisFromConfig
};

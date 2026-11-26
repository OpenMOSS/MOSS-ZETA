"use strict";

// Port of pretrain/joint_think_regional_ocr.py followed by
// pretrain/training_ocr_input.py. Keep the eight-region ownership pass before
// projecting to P/B/C/K: discarded U/L/R/I still shadow overlapping regions.
const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");

const PROFILE = "training_pbck_v2";
const FIELDS = Object.freeze(["P", "B", "C", "K"]);
const REGION_ORDER = ["P", "U", "L", "B", "R", "C", "I", "K"];
const ROI_NAMES = Object.freeze({
  P: "tutorial_prompt", B: "mission_objective", C: "central_result", K: "key_prompt"
});
const configBytes = fs.readFileSync(path.join(__dirname, "..", "prompts", "training-ocr-regions.json"));
const RULES_SHA256 = createHash("sha256").update(configBytes).digest("hex");
const config = JSON.parse(configBytes.toString("utf8"));
// Python str/re whitespace differs from JavaScript \s (notably U+0085,
// U+001C..1F and U+FEFF). NFKC runs before whitespace collapsing in both ports.
const PYTHON_WHITESPACE = /[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/gu;
const cjk = "\\u3400-\\u9fff";

function requireValid(condition, message) {
  if (!condition) throw new Error(`Training OCR: ${message}`);
}

function normalizeText(value) {
  return String(value || "").normalize("NFKC").replace(PYTHON_WHITESPACE, " ")
    .replace(/^ +| +$/g, "");
}

function pythonLength(value) {
  return Array.from(value).length;
}

function escapeValue(value) {
  return value.replace(/\\/g, "\\\\").replace(/\|/g, "\\|");
}

function splitLines(value) {
  if (!value) return [];
  const result = [];
  let start = 0;
  let escaped = false;
  for (let i = 0; i < value.length; i += 1) {
    if (escaped) escaped = false;
    else if (value[i] === "\\") escaped = true;
    else if (value[i] === "|") {
      result.push(value.slice(start, i));
      start = i + 1;
    }
  }
  result.push(value.slice(start));
  return result;
}

function normalizePolicyOcrValue(value) {
  const lines = [];
  for (const encoded of splitLines(String(value || ""))) {
    let text = normalizeText(encoded.replace(/\\([\\|])/g, "$1"));
    text = text.replace(/\.{2,}/g, "…").replace(/([!?。])\1+/g, "$1")
      .replace(new RegExp(`(?<=[${cjk}]) +(?=[${cjk}A-Za-z0-9])`, "gu"), "")
      .replace(new RegExp(`(?<=[A-Za-z0-9]) +(?=[${cjk}])`, "gu"), "")
      .replace(/ +(?=[,;:!?。…])/g, "")
      .replace(new RegExp(`(?<=[,;:!?。…]) +(?=[${cjk}])`, "gu"), "");
    if (text) lines.push(escapeValue(text));
  }
  return lines.join("|");
}

// The frozen rules use Python scoped (?i:...) groups, unavailable in Node 22.
// Expand their ASCII literals instead of applying /i to the entire pattern.
// Python's Unicode IGNORECASE additionally matches these four code points.
function compilePythonPattern(pattern) {
  const scopes = [false];
  let out = "";
  let inClass = false;
  for (let i = 0; i < pattern.length; i += 1) {
    const ch = pattern[i];
    if (ch === "\\") {
      requireValid(i + 1 < pattern.length, "invalid rule escape");
      out += ch + pattern[++i];
      continue;
    }
    if (ch === "[") inClass = true;
    if (!inClass && ch === "(") {
      if (pattern.startsWith("(?i:", i)) {
        scopes.push(true);
        out += "(?:";
        i += 3;
        continue;
      }
      scopes.push(scopes[scopes.length - 1]);
    } else if (!inClass && ch === ")") scopes.pop();
    if (scopes[scopes.length - 1] && /[A-Za-z]/.test(ch)) {
      requireValid(!inClass, "unsupported case-insensitive character class");
      const lower = ch.toLowerCase();
      const extra = { i: "İı", s: "ſ", k: "K" }[lower] || "";
      out += `[${lower}${lower.toUpperCase()}${extra}]`;
    } else out += ch;
    if (ch === "]") inClass = false;
  }
  requireValid(scopes.length === 1 && !inClass, "invalid rule grouping");
  return new RegExp(out, "u");
}

requireValid(JSON.stringify(config.regions.map((region) => region.id)) === JSON.stringify(REGION_ORDER),
  "region order differs from training protocol");
const regions = config.regions.map((region) => {
  const [x1, y1, x2, y2] = region.rect;
  requireValid(0 <= x1 && x1 < x2 && x2 <= 1 && 0 <= y1 && y1 < y2 && y2 <= 1,
    "invalid rule geometry");
  return {
    ...region,
    include: compilePythonPattern(region.include_pattern),
    activate: region.activation_pattern ? compilePythonPattern(region.activation_pattern) : null
  };
});

function contains(region, candidate) {
  const [x1, y1, x2, y2] = region.rect;
  return x1 <= candidate.x && candidate.x <= x2 && y1 <= candidate.y && candidate.y <= y2;
}

function selectTrainingPolicyOcr(raw) {
  requireValid(raw && typeof raw === "object", "missing OCR source frame");
  requireValid(raw.ok === undefined || Boolean(raw.ok), "OCR source frame failed; do not encode a false disappearance");
  const width = raw.image_width;
  const height = raw.image_height;
  requireValid(typeof width === "number" && Number.isFinite(width) && width > 0
    && typeof height === "number" && Number.isFinite(height) && height > 0, "invalid OCR dimensions");
  const sourceLines = raw.lines === undefined ? [] : raw.lines;
  requireValid(Array.isArray(sourceLines), "OCR lines must be an array");
  const stats = { raw_lines: sourceLines.length };
  const count = (key) => { stats[key] = (stats[key] || 0) + 1; };
  const candidates = [];
  for (let index = 0; index < sourceLines.length; index += 1) {
    const line = sourceLines[index];
    requireValid(line && typeof line === "object", "invalid OCR line");
    const text = normalizeText(line.text);
    const score = Number(line.score === undefined ? 0 : line.score);
    const box = line.box_xyxy;
    const length = pythonLength(text);
    const threshold = length === 1 ? config.single_char_min_score : config.min_score;
    if (!text || !Number.isFinite(score) || score < threshold || !Array.isArray(box) || box.length !== 4) {
      count("low_quality_lines");
      continue;
    }
    requireValid(box.every((value) => typeof value === "number" && Number.isFinite(value)), "nonfinite OCR box");
    const [x1, y1, x2, y2] = box;
    if (x1 >= x2 || y1 >= y2) {
      count("invalid_boxes");
      continue;
    }
    candidates.push({ text, score, x: (x1 + x2) / 2 / width, y: (y1 + y2) / 2 / height,
      box, sourceIndex: index });
  }
  const active = Object.fromEntries(regions.map((region) => [region.id,
    !region.activate || candidates.some((candidate) => contains(region, candidate) && region.activate.test(candidate.text))]));
  const groups = Object.fromEntries(REGION_ORDER.map((key) => [key, []]));
  for (const candidate of candidates) {
    const owner = regions.find((region) => active[region.id] && contains(region, candidate) && region.include.test(candidate.text));
    if (owner) groups[owner.id].push(candidate);
    else count("outside_useful_regions_or_filtered");
  }
  const state = Object.fromEntries(REGION_ORDER.map((key) => [key, ""]));
  const selectedLines = [];
  for (const [key, items] of Object.entries(groups)) {
    const seen = new Set();
    const texts = [];
    let chars = 0;
    items.sort((a, b) => a.box[1] - b.box[1] || a.box[0] - b.box[0] || a.sourceIndex - b.sourceIndex);
    for (const item of items) {
      if (seen.has(item.text)) {
        count("duplicate_lines");
        continue;
      }
      seen.add(item.text);
      const length = pythonLength(item.text);
      if (texts.length >= config.max_lines_per_region || chars + length > config.max_chars_per_region) {
        count("over_budget_lines");
        continue;
      }
      texts.push(escapeValue(item.text));
      chars += length;
      selectedLines.push({ region: key, text: item.text, score: item.score, box_xyxy: [...item.box] });
      count(`selected_${key}`);
      count("selected_lines");
    }
    state[key] = texts.join("|");
  }
  return {
    fields: Object.fromEntries(FIELDS.map((key) => [key, normalizePolicyOcrValue(state[key])])),
    selectedLines, stats, profile: PROFILE, rulesSha256: RULES_SHA256
  };
}

function polygonBox(polygon) {
  if (polygon == null) return null;
  requireValid(Array.isArray(polygon) && polygon.length >= 3
    && polygon.every((point) => Array.isArray(point) && point.length === 2
      && point.every((value) => typeof value === "number" && Number.isFinite(value))), "malformed full-screen OCR polygon");
  const xs = polygon.map((point) => point[0]);
  const ys = polygon.map((point) => point[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function adaptTrainingOcrResult(body) {
  requireValid(body && body.ok === true, "OCR source frame failed; do not encode a false disappearance");
  requireValid(!body.error, "OCR source returned an error");
  requireValid(Array.isArray(body.results), "missing full-screen OCR results");
  const fullResults = body.results.filter((result) => result?.roi === "full_screen");
  requireValid(fullResults.length === 1, "expected exactly one full_screen OCR result");
  const full = fullResults[0];
  requireValid(full.pipeline === "full" && (!body.pipeline || body.pipeline === "full"), "full OCR pipeline required");
  requireValid(Array.isArray(full.box) && full.box.length === 4
    && full.box.every((value) => typeof value === "number" && Number.isFinite(value))
    && full.box[0] === 0 && full.box[1] === 0 && full.box[2] > 0 && full.box[3] > 0,
  "invalid full_screen OCR dimensions or crop origin");
  const [, , width, height] = full.box;
  for (const [key, expected] of [["image_width", width], ["image_height", height], ["imageWidth", width], ["imageHeight", height]]) {
    if (body[key] !== undefined) requireValid(body[key] === expected, "full_screen OCR dimensions disagree with source image");
  }
  requireValid(Array.isArray(full.lines), "missing untruncated full-screen OCR lines");
  requireValid(!body.truncated && !full.truncated, "truncated full-screen OCR lines");
  const selection = selectTrainingPolicyOcr({
    ok: true, image_width: width, image_height: height,
    lines: full.lines.map((line) => {
      requireValid(line && typeof line === "object" && typeof line.text === "string", "malformed full-screen OCR line");
      requireValid(!line.truncated, "truncated full-screen OCR line");
      return { text: line.text, score: line.score, box_xyxy: polygonBox(line.box) };
    })
  });
  const results = FIELDS.map((key) => {
    const region = regions.find((item) => item.id === key);
    const box = region.rect.map((value, index) => value * (index % 2 === 0 ? width : height));
    const lines = selection.selectedLines.filter((line) => line.region === key).map((line) => {
      const [x1, y1, x2, y2] = line.box_xyxy;
      return { text: line.text, score: line.score,
        box: [[x1 - box[0], y1 - box[1]], [x2 - box[0], y1 - box[1]],
          [x2 - box[0], y2 - box[1]], [x1 - box[0], y2 - box[1]]] };
    });
    return { ...full, roi: ROI_NAMES[key], box, lines,
      text: lines.map((line) => line.text).join(" "),
      score: lines.length ? lines.reduce((sum, line) => sum + line.score, 0) / lines.length : null,
      inputScope: "full_screen", selectionProfile: PROFILE };
  });
  return {
    ...body, rois: FIELDS.map((key) => ROI_NAMES[key]), results,
    policyFields: selection.fields,
    policyOcrAudit: { profile: PROFILE, rulesSha256: RULES_SHA256,
      stats: selection.stats, selectedLines: selection.selectedLines }
  };
}

module.exports = { selectTrainingPolicyOcr, normalizePolicyOcrValue, adaptTrainingOcrResult };

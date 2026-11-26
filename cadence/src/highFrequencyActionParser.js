const DEFAULT_CHUNKS_PER_PACKAGE = 6;
const DEFAULT_CHUNK_DURATION_MS = 33;
const THINKING_TAG_PATTERN = /<(think|thinking|thought|thoughts|reasoning)\b[^>]*>[\s\S]*?<\/\1>/gi;
const STICK_ACTION_PATTERN = /^([LR])\s*\(\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*\)$/;
const COMPACT_COORDINATE_PATTERN = /^([+-])(\d)([+-])(\d)([+-])(\d)([+-])(\d)/;

const BUTTON_ACTIONS = new Map([
  ["LB", "lb"],
  ["RB", "rb"],
  ["L3", "left_thumb"],
  ["R3", "right_thumb"],
  ["view", "back"],
  ["menu", "start"],
  ["up", "dpad_up"],
  ["down", "dpad_down"],
  ["left", "dpad_left"],
  ["right", "dpad_right"],
  ["X", "x"],
  ["Y", "y"],
  ["B", "b"],
  ["A", "a"]
]);
const BUTTON_DISPLAY_NAMES = new Map(
  [...BUTTON_ACTIONS.entries()].map(([actionName, buttonName]) => [buttonName, actionName])
);

const ACTIONS = new Map([
  ["LT", { trigger: "leftTrigger", value: 1 }],
  ["RT", { trigger: "rightTrigger", value: 1 }],
  ...[...BUTTON_ACTIONS.entries()].map(([name, button]) => [name, { button }])
]);

function neutralControllerState() {
  return {
    leftStick: { x: 0, y: 0 },
    rightStick: { x: 0, y: 0 },
    leftTrigger: 0,
    rightTrigger: 0,
    buttons: []
  };
}

function positiveInteger(value, fallback, label) {
  const number = Math.trunc(Number(value ?? fallback));
  if (!Number.isFinite(number) || number <= 0) {
    throw new Error(`${label} must be a positive integer.`);
  }

  return number;
}

function stickNumber(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < -1 || number > 1) {
    throw new Error(`${label} must be a number from -1 to 1.`);
  }
  return number;
}

function compactStickValue(sign, digit, label) {
  if (sign === "-" && digit === "0") {
    throw new Error(`${label} must use +0 for a centered axis; -0 is not allowed.`);
  }

  const magnitude = Number(digit) / 9;
  return sign === "-" ? -magnitude : magnitude;
}

function compactActionTokens(chunkText, chunkIndex) {
  const text = String(chunkText || "");
  if (!text) {
    return [];
  }

  const names = [...ACTIONS.keys()].sort((left, right) => right.length - left.length);
  const tokens = [];
  let offset = 0;
  while (offset < text.length) {
    const token = names.find((name) => text.startsWith(name, offset));
    if (!token) {
      throw new Error(`Unknown compact high-frequency action token in chunk ${chunkIndex + 1} near "${text.slice(offset)}".`);
    }
    tokens.push(token);
    offset += token.length;
  }
  return tokens;
}

function parseCompactState(coordinates, buttons, index = 0) {
  const match = String(coordinates).match(COMPACT_COORDINATE_PATTERN);
  if (!match || coordinates.length !== 8) throw Error('Incomplete compact coordinates');
  const state = neutralControllerState();
  state.leftStick = { x: compactStickValue(match[1], match[2], 'left x'), y: compactStickValue(match[3], match[4], 'left y') };
  state.rightStick = { x: compactStickValue(match[5], match[6], 'right x'), y: compactStickValue(match[7], match[8], 'right y') };
  for (const action of compactActionTokens(buttons, index)) applyActionToState(state, action);
  return state;
}

function parseCompactPackage(trimmed, chunksPerPackage, chunkDurationMs) {
  const coordinateMatch = trimmed.match(COMPACT_COORDINATE_PATTERN);
  if (!coordinateMatch) {
    throw new Error("Compact high-frequency package must start with four signed one-digit stick coordinates.");
  }

  const leftStick = {
    x: compactStickValue(coordinateMatch[1], coordinateMatch[2], "left stick x"),
    y: compactStickValue(coordinateMatch[3], coordinateMatch[4], "left stick y")
  };
  const rightStick = {
    x: compactStickValue(coordinateMatch[5], coordinateMatch[6], "right stick x"),
    y: compactStickValue(coordinateMatch[7], coordinateMatch[8], "right stick y")
  };
  const actionText = trimmed.slice(coordinateMatch[0].length);
  if (/\s/.test(actionText)) {
    throw new Error("Compact high-frequency package cannot contain spaces or line breaks inside <a>...</a>.");
  }

  const originalChunkTexts = actionText.split(";");
  let chunkTexts = originalChunkTexts.slice();
  let normalizationReason = "";
  if (chunkTexts.length === chunksPerPackage + 1 && chunkTexts.at(-1) === "") {
    chunkTexts = chunkTexts.slice(0, -1);
    normalizationReason = "trailing_semicolon";
  }
  if (chunkTexts.length !== chunksPerPackage) {
    throw new Error(`High-frequency action package must contain exactly ${chunksPerPackage} chunks; got ${chunkTexts.length}.`);
  }

  const chunks = chunkTexts.map((chunkText, index) => {
    const actions = compactActionTokens(chunkText, index);
    const state = neutralControllerState();
    state.leftStick = { ...leftStick };
    state.rightStick = { ...rightStick };
    for (const action of actions) {
      applyActionToState(state, action);
    }
    return {
      index,
      raw: chunkText,
      actions,
      state,
      compact: true
    };
  });

  return {
    type: "high_frequency_package",
    raw: trimmed,
    trimmed,
    originalActionText: trimmed,
    actionText,
    normalized: Boolean(normalizationReason),
    normalizationReason: normalizationReason || null,
    barePackage: false,
    compact: true,
    compactCoordinates: {
      leftStick: { ...leftStick },
      rightStick: { ...rightStick },
      raw: coordinateMatch[0]
    },
    chunkDurationMs,
    chunksPerPackage,
    chunks,
    states: chunks.map((chunk) => chunk.state)
  };
}

function applyActionToState(state, actionName) {
  const stickMatch = String(actionName || "").trim().match(STICK_ACTION_PATTERN);
  if (stickMatch) {
    const stickName = stickMatch[1] === "L" ? "leftStick" : "rightStick";
    state[stickName] = {
      x: stickNumber(stickMatch[2], `${stickMatch[1]} x`),
      y: stickNumber(stickMatch[3], `${stickMatch[1]} y`)
    };
    return;
  }

  const action = ACTIONS.get(actionName);
  if (!action) {
    throw new Error(`Unknown high-frequency action: ${actionName}`);
  }

  if (action.stick) {
    state[action.stick] = { x: action.x, y: action.y };
    return;
  }

  if (action.trigger) {
    state[action.trigger] = action.value;
    return;
  }

  if (action.button && !state.buttons.includes(action.button)) {
    state.buttons.push(action.button);
  }
}

function parseChunk(chunkText, chunkIndex) {
  const trimmed = String(chunkText || "").trim();
  const state = neutralControllerState();
  if (!trimmed) {
    return {
      index: chunkIndex,
      raw: chunkText,
      actions: [],
      state
    };
  }

  const tokens = trimmed
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (const token of tokens) {
    if (!STICK_ACTION_PATTERN.test(token) && !/^[A-Za-z][A-Za-z0-9]*$/.test(token)) {
      throw new Error(`Invalid high-frequency action token in chunk ${chunkIndex + 1}: ${token}`);
    }
    applyActionToState(state, token);
  }

  return {
    index: chunkIndex,
    raw: chunkText,
    actions: tokens,
    state
  };
}

function stripThinkingText(text) {
  let stripped = String(text ?? "").replace(THINKING_TAG_PATTERN, "").trim();
  const lower = stripped.toLowerCase();
  const closingThinkIndex = lower.lastIndexOf("</think>");
  if (closingThinkIndex !== -1) {
    stripped = stripped.slice(closingThinkIndex + "</think>".length).trim();
  }
  return stripped;
}

function extractActionPackageText(text, options = {}) {
  const stripped = stripThinkingText(text);
  const openingTags = stripped.match(/<a>/g) || [];
  const closingTags = stripped.match(/<\/a>/g) || [];
  if (options.allowBarePackage && openingTags.length === 0 && closingTags.length === 0) {
    return {
      text: stripped,
      barePackage: true
    };
  }
  if (openingTags.length !== 1 || closingTags.length !== 1) {
    throw new Error("High-frequency action package must contain exactly one <a>...</a> block.");
  }

  const match = stripped.match(/<a>([\s\S]*?)<\/a>/);
  if (!match) {
    throw new Error("High-frequency action package must contain a complete <a>...</a> block.");
  }
  if (options.requireOnlyActionBlock && stripped !== match[0]) {
    throw new Error("High-frequency action package must contain only one <a>...</a> block and outer whitespace.");
  }

  return {
    text: match[1].trim(),
    barePackage: false
  };
}

function repairHighFrequencyPackageText(text, options = {}) {
  const chunksPerPackage = positiveInteger(options.chunksPerPackage, DEFAULT_CHUNKS_PER_PACKAGE, "chunksPerPackage");
  const raw = String(text ?? "");
  const extracted = extractActionPackageText(raw, options);
  let inner = extracted.text;
  const reasons = [];

  if (/^[+-]\d/.test(inner)) {
    let coordinateEnd = 0;
    let coordinateCount = 0;
    while (coordinateCount < 4 && /^[+-]\d/.test(inner.slice(coordinateEnd, coordinateEnd + 2))) {
      coordinateEnd += 2;
      coordinateCount += 1;
    }
    if (coordinateCount < 4) {
      inner = `${inner.slice(0, coordinateEnd)}${"+0".repeat(4 - coordinateCount)}${inner.slice(coordinateEnd)}`;
      reasons.push(`missing_stick_coordinates:${4 - coordinateCount}`);
    }
  }

  const coordinatePrefix = /^[+-]\d[+-]\d[+-]\d[+-]\d/.exec(inner)?.[0] || "";
  const actionText = coordinatePrefix ? inner.slice(coordinatePrefix.length) : inner;
  let chunks = actionText.split(";");
  if (chunks.length === chunksPerPackage + 1 && chunks.at(-1).trim() === "") {
    if (reasons.length === 0) {
      return {
        text: raw,
        repaired: false,
        reasons: [],
        originalText: inner
      };
    }
    chunks = chunks.slice(0, chunksPerPackage);
  }
  if (chunks.length > chunksPerPackage) {
    chunks = chunks.slice(0, chunksPerPackage);
    reasons.push(`truncated_chunks:${actionText.split(";").length - chunksPerPackage}`);
  } else if (chunks.length < chunksPerPackage) {
    chunks.push(...Array.from({ length: chunksPerPackage - chunks.length }, () => ""));
    reasons.push(`padded_chunks:${chunksPerPackage - actionText.split(";").length}`);
  }

  if (reasons.length === 0) {
    return {
      text: raw,
      repaired: false,
      reasons: [],
      originalText: inner
    };
  }

  const repairedInner = `${coordinatePrefix || ""}${chunks.join(";")}`;
  const stripped = stripThinkingText(raw);
  if (extracted.barePackage) {
    return {
      text: repairedInner,
      repaired: true,
      reasons,
      originalText: inner
    };
  }
  const match = stripped.match(/<a>([\s\S]*?)<\/a>/);
  return {
    text: `${stripped.slice(0, match.index)}<a>${repairedInner}</a>${stripped.slice(match.index + match[0].length)}`,
    repaired: true,
    reasons,
    originalText: inner
  };
}

function parseHighFrequencyPackage(text, options = {}) {
  const chunksPerPackage = positiveInteger(options.chunksPerPackage, DEFAULT_CHUNKS_PER_PACKAGE, "chunksPerPackage");
  const chunkDurationMs = Number(options.chunkDurationMs ?? DEFAULT_CHUNK_DURATION_MS);
  if (!Number.isFinite(chunkDurationMs) || chunkDurationMs <= 0) throw new Error("chunkDurationMs must be positive.");
  const originalRaw = String(text ?? "");
  const repair = options.autoRepair
    ? repairHighFrequencyPackageText(originalRaw, options)
    : null;
  const raw = repair?.repaired ? repair.text : originalRaw;
  const extracted = extractActionPackageText(raw, options);
  const trimmed = extracted.text;

  if (!trimmed) {
    throw new Error("High-frequency action package is empty.");
  }
  if (/^[+-]\d/.test(trimmed)) {
    const parsed = parseCompactPackage(trimmed, chunksPerPackage, chunkDurationMs);
    return repair?.repaired
      ? { ...parsed, repaired: true, repairReasons: repair.reasons, repairOriginalText: repair.originalText }
      : parsed;
  }
  if (/```/.test(trimmed) || /[{}[\]"]/g.test(trimmed)) {
    throw new Error("High-frequency action package block contains unsupported syntax.");
  }

  const originalActionText = trimmed;
  const chunkTexts = originalActionText.split(";");
  let normalizationReason = "";
  if (chunkTexts.length === chunksPerPackage + 1 && chunkTexts.at(-1).trim() === "") {
    chunkTexts.pop();
    normalizationReason = "trailing_semicolon";
  }
  if (chunkTexts.length !== chunksPerPackage) {
    throw new Error(`High-frequency action package must contain exactly ${chunksPerPackage} chunks; got ${chunkTexts.length}.`);
  }

  const chunks = chunkTexts.map((chunkText, index) => parseChunk(chunkText, index));
  const actionText = chunkTexts.join(";");

  return {
    type: "high_frequency_package",
    raw,
    trimmed,
    originalActionText,
    actionText,
    normalized: Boolean(normalizationReason),
    normalizationReason: normalizationReason || null,
    repaired: Boolean(repair?.repaired),
    repairReasons: repair?.reasons || [],
    repairOriginalText: repair?.repaired ? repair.originalText : null,
    barePackage: extracted.barePackage,
    chunkDurationMs,
    chunksPerPackage,
    chunks,
    states: chunks.map((chunk) => chunk.state)
  };
}

function describeHighFrequencyPackage(actionPackage) {
  const chunks = Array.isArray(actionPackage?.chunks) ? actionPackage.chunks : [];
  const duration = actionPackage?.chunkDurationMs || DEFAULT_CHUNK_DURATION_MS;
  const packageSticks = actionPackage?.compactCoordinates
    ? {
        leftStick: actionPackage.compactCoordinates.leftStick,
        rightStick: actionPackage.compactCoordinates.rightStick
      }
    : null;
  const summary = chunks
    .map((chunk, index) => `${index + 1}: ${describeHighFrequencyChunk(chunk, {
      includeSticks: !packageSticks,
      emptyLabel: packageSticks ? "none" : "neutral"
    })}`)
    .join("; ");
  const stickPrefix = packageSticks
    ? `${describeStickState(packageSticks)} | `
    : "";
  return `${stickPrefix}${chunks.length} chunks @${duration}ms: ${summary}`;
}

function formatAxis(value) {
  const number = Number(value || 0);
  return Number.isInteger(number) ? String(number) : String(Number(number.toFixed(3)));
}

function describeStickState(state) {
  const leftStick = state?.leftStick || { x: 0, y: 0 };
  const rightStick = state?.rightStick || { x: 0, y: 0 };
  return `L(${formatAxis(leftStick.x)},${formatAxis(leftStick.y)}) R(${formatAxis(rightStick.x)},${formatAxis(rightStick.y)})`;
}

function describeHighFrequencyChunk(chunk, options = {}) {
  const state = chunk?.state || chunk || neutralControllerState();
  const leftStick = state.leftStick || { x: 0, y: 0 };
  const rightStick = state.rightStick || { x: 0, y: 0 };
  const parts = [];
  if (options.includeSticks !== false && (leftStick.x || leftStick.y)) {
    parts.push(`L(${formatAxis(leftStick.x)},${formatAxis(leftStick.y)})`);
  }
  if (options.includeSticks !== false && (rightStick.x || rightStick.y)) {
    parts.push(`R(${formatAxis(rightStick.x)},${formatAxis(rightStick.y)})`);
  }
  if (state.leftTrigger) parts.push("LT");
  if (state.rightTrigger) parts.push("RT");
  for (const button of Array.isArray(state.buttons) ? state.buttons : []) {
    parts.push(BUTTON_DISPLAY_NAMES.get(button) || String(button));
  }
  return parts.length > 0 ? parts.join(" ") : (options.emptyLabel || "neutral");
}

module.exports = {
  DEFAULT_CHUNK_DURATION_MS,
  DEFAULT_CHUNKS_PER_PACKAGE,
  ACTIONS,
  parseCompactState,
  repairHighFrequencyPackageText,
  parseHighFrequencyPackage,
  describeHighFrequencyPackage,
  describeHighFrequencyChunk,
  neutralControllerState
};

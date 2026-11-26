const MAX_DELTA = 5000;
const MAX_WAIT_MS = 10000;
const MAX_DURATION_MS = 5000;
const MAX_STICK_VALUE = 1;
const MAX_TRIGGER_VALUE = 1;
const DEFAULT_GAMEPAD_PRESS_MS = 80;
const MOUSE_BUTTONS = new Set(["left", "right", "middle"]);
const GAMEPAD_STICK_ALIASES = new Map([
  ["left", "left"],
  ["l", "left"],
  ["ls", "left"],
  ["left_stick", "left"],
  ["right", "right"],
  ["r", "right"],
  ["rs", "right"],
  ["right_stick", "right"]
]);
const GAMEPAD_TRIGGER_ALIASES = new Map([
  ["left", "left"],
  ["l", "left"],
  ["lt", "left"],
  ["l2", "left"],
  ["left2", "left"],
  ["left_trigger", "left"],
  ["right", "right"],
  ["r", "right"],
  ["rt", "right"],
  ["r2", "right"],
  ["right2", "right"],
  ["right_trigger", "right"]
]);
const GAMEPAD_BUTTON_ALIASES = new Map([
  ["a", "a"],
  ["cross", "a"],
  ["x_cross", "a"],
  ["b", "b"],
  ["circle", "b"],
  ["o", "b"],
  ["x", "x"],
  ["square", "x"],
  ["y", "y"],
  ["triangle", "y"],
  ["lb", "lb"],
  ["l1", "lb"],
  ["left_bumper", "lb"],
  ["left_shoulder", "lb"],
  ["rb", "rb"],
  ["r1", "rb"],
  ["right_bumper", "rb"],
  ["right_shoulder", "rb"],
  ["back", "back"],
  ["select", "back"],
  ["view", "back"],
  ["share", "back"],
  ["create", "back"],
  ["touchpad", "back"],
  ["start", "start"],
  ["menu", "start"],
  ["options", "start"],
  ["guide", "guide"],
  ["xbox", "guide"],
  ["ps", "guide"],
  ["playstation", "guide"],
  ["left_thumb", "left_thumb"],
  ["left_stick_button", "left_thumb"],
  ["l3", "left_thumb"],
  ["right_thumb", "right_thumb"],
  ["right_stick_button", "right_thumb"],
  ["r3", "right_thumb"],
  ["dpad_up", "dpad_up"],
  ["up", "dpad_up"],
  ["dpad_down", "dpad_down"],
  ["down", "dpad_down"],
  ["dpad_left", "dpad_left"],
  ["left", "dpad_left"],
  ["dpad_right", "dpad_right"],
  ["right", "dpad_right"]
]);
const GAMEPAD_STICK_FUNCTIONS = new Set(["gamepad_stick", "controller_stick", "pad_stick", "stick"]);
const GAMEPAD_TRIGGER_FUNCTIONS = new Set(["gamepad_trigger", "controller_trigger", "pad_trigger", "trigger"]);
const GAMEPAD_BUTTON_DOWN_FUNCTIONS = new Set(["gamepad_down", "gamepad_button_down", "controller_down", "pad_down"]);
const GAMEPAD_BUTTON_UP_FUNCTIONS = new Set(["gamepad_up", "gamepad_button_up", "controller_up", "pad_up"]);
const GAMEPAD_BUTTON_PRESS_FUNCTIONS = new Set(["gamepad_press", "gamepad_button_press", "controller_press", "pad_press"]);
const GAMEPAD_RESET_FUNCTIONS = new Set(["gamepad_reset", "controller_reset", "pad_reset", "reset_gamepad"]);
const GTA_SHORTHAND_STICK_FUNCTIONS = new Map([
  ["lup", { stick: "left", x: 0, y: 1 }],
  ["ldown", { stick: "left", x: 0, y: -1 }],
  ["lleft", { stick: "left", x: -1, y: 0 }],
  ["lright", { stick: "left", x: 1, y: 0 }],
  ["rup", { stick: "right", x: 0, y: 1 }],
  ["rdown", { stick: "right", x: 0, y: -1 }],
  ["rleft", { stick: "right", x: -1, y: 0 }],
  ["rright", { stick: "right", x: 1, y: 0 }]
]);
const GTA_SHORTHAND_TRIGGER_FUNCTIONS = new Map([
  ["lt", "left"],
  ["rt", "right"]
]);
const GTA_SHORTHAND_BUTTON_FUNCTIONS = new Map([
  ["lb", "lb"],
  ["rb", "rb"],
  ["l3", "left_thumb"],
  ["r3", "right_thumb"],
  ["view", "back"],
  ["menu", "start"],
  ["up", "dpad_up"],
  ["down", "dpad_down"],
  ["left", "dpad_left"],
  ["right", "dpad_right"],
  ["x", "x"],
  ["y", "y"],
  ["b", "b"],
  ["a", "a"]
]);
const GTA_SHORTHAND_FUNCTIONS = new Set([
  ...GTA_SHORTHAND_STICK_FUNCTIONS.keys(),
  ...GTA_SHORTHAND_TRIGGER_FUNCTIONS.keys(),
  ...GTA_SHORTHAND_BUTTON_FUNCTIONS.keys()
]);
const THINKING_TAG_PATTERN = /<(think|thinking|thought|thoughts|reasoning)\b[^>]*>[\s\S]*?<\/\1>/gi;
const UNCLOSED_THINKING_BEFORE_CODE_PATTERN = /<(think|thinking|thought|thoughts|reasoning)\b[^>]*>[\s\S]*?(?=```python)/gi;
const ALLOWED_FUNCTIONS = new Set([
  "mouse_move",
  "move_mouse",
  "move",
  "mouse_down",
  "button_down",
  "mouse_up",
  "button_up",
  "click",
  "mouse_click",
  "key_down",
  "keydown",
  "key_up",
  "keyup",
  "key_press",
  "keypress",
  "press",
  "wait",
  "sleep",
  "wheel",
  "mouse_wheel",
  ...GAMEPAD_STICK_FUNCTIONS,
  ...GAMEPAD_TRIGGER_FUNCTIONS,
  ...GAMEPAD_BUTTON_DOWN_FUNCTIONS,
  ...GAMEPAD_BUTTON_UP_FUNCTIONS,
  ...GAMEPAD_BUTTON_PRESS_FUNCTIONS,
  ...GAMEPAD_RESET_FUNCTIONS,
  ...GTA_SHORTHAND_FUNCTIONS
]);

function clampNumber(value, min, max, fallback = 0) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }

  return Math.max(min, Math.min(max, number));
}

function normalizeType(type) {
  return String(type || "")
    .trim()
    .replace(/([a-z])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
}

function normalizeButton(button) {
  const normalized = String(button || "left").trim().toLowerCase();
  if (!MOUSE_BUTTONS.has(normalized)) {
    throw new Error(`Unsupported mouse button: ${button}`);
  }

  return normalized;
}

function normalizeAlias(value, aliases, label) {
  const normalized = normalizeType(value || "");
  const alias = aliases.get(normalized);
  if (!alias) {
    throw new Error(`Unsupported ${label}: ${value}`);
  }

  return alias;
}

function normalizeGamepadStick(stick) {
  return normalizeAlias(stick || "left", GAMEPAD_STICK_ALIASES, "gamepad stick");
}

function normalizeGamepadTrigger(trigger) {
  return normalizeAlias(trigger || "right", GAMEPAD_TRIGGER_ALIASES, "gamepad trigger");
}

function normalizeGamepadButton(button) {
  return normalizeAlias(button, GAMEPAD_BUTTON_ALIASES, "gamepad button");
}

function normalizeAction(action) {
  if (!action || typeof action !== "object") {
    throw new Error("Action must be an object.");
  }

  const type = normalizeType(action.type || action.action || action.name);

  if (type === "mouse_move" || type === "move_mouse" || type === "move") {
    const delta = action.delta && typeof action.delta === "object" ? action.delta : {};
    return [
      {
        type: "mouse_move",
        dx: Math.trunc(clampNumber(action.dx ?? delta.dx ?? delta.x, -MAX_DELTA, MAX_DELTA)),
        dy: Math.trunc(clampNumber(action.dy ?? delta.dy ?? delta.y, -MAX_DELTA, MAX_DELTA)),
        durationMs: Math.trunc(clampNumber(action.durationMs ?? action.duration_ms ?? action.ms, 0, MAX_DURATION_MS))
      }
    ];
  }

  if (type === "mouse_down" || type === "button_down") {
    return [
      {
        type: "mouse_down",
        button: normalizeButton(action.button)
      }
    ];
  }

  if (type === "mouse_up" || type === "button_up") {
    return [
      {
        type: "mouse_up",
        button: normalizeButton(action.button)
      }
    ];
  }

  if (type === "click" || type === "mouse_click") {
    const button = normalizeButton(action.button);
    return [
      { type: "mouse_down", button },
      { type: "mouse_up", button }
    ];
  }

  if (type === "key_down" || type === "keydown") {
    if (!action.key) {
      throw new Error("key_down requires key.");
    }

    return [
      {
        type: "key_down",
        key: String(action.key)
      }
    ];
  }

  if (type === "key_up" || type === "keyup") {
    if (!action.key) {
      throw new Error("key_up requires key.");
    }

    return [
      {
        type: "key_up",
        key: String(action.key)
      }
    ];
  }

  if (type === "key_press" || type === "keypress" || type === "press") {
    if (!action.key) {
      throw new Error("key_press requires key.");
    }

    const key = String(action.key);
    return [
      { type: "key_down", key },
      { type: "key_up", key }
    ];
  }

  if (type === "wait" || type === "sleep") {
    return [
      {
        type: "wait",
        ms: Math.trunc(clampNumber(action.ms ?? action.durationMs ?? action.duration_ms, 0, MAX_WAIT_MS, 100))
      }
    ];
  }

  if (type === "wheel" || type === "mouse_wheel") {
    return [
      {
        type: "wheel",
        amount: Math.trunc(clampNumber(action.amount ?? action.delta ?? action.dy, -2400, 2400))
      }
    ];
  }

  if (GAMEPAD_STICK_FUNCTIONS.has(type)) {
    return [
      {
        type: "gamepad_stick",
        stick: normalizeGamepadStick(action.stick),
        x: clampNumber(action.x ?? action.dx, -MAX_STICK_VALUE, MAX_STICK_VALUE),
        y: clampNumber(action.y ?? action.dy, -MAX_STICK_VALUE, MAX_STICK_VALUE),
        durationMs: Math.trunc(clampNumber(action.durationMs ?? action.duration_ms ?? action.ms, 0, MAX_DURATION_MS))
      }
    ];
  }

  if (GAMEPAD_TRIGGER_FUNCTIONS.has(type)) {
    return [
      {
        type: "gamepad_trigger",
        trigger: normalizeGamepadTrigger(action.trigger),
        value: clampNumber(action.value ?? action.amount, 0, MAX_TRIGGER_VALUE),
        durationMs: Math.trunc(clampNumber(action.durationMs ?? action.duration_ms ?? action.ms, 0, MAX_DURATION_MS))
      }
    ];
  }

  if (GAMEPAD_BUTTON_DOWN_FUNCTIONS.has(type)) {
    return [
      {
        type: "gamepad_button_down",
        button: normalizeGamepadButton(action.button)
      }
    ];
  }

  if (GAMEPAD_BUTTON_UP_FUNCTIONS.has(type)) {
    return [
      {
        type: "gamepad_button_up",
        button: normalizeGamepadButton(action.button)
      }
    ];
  }

  if (GAMEPAD_BUTTON_PRESS_FUNCTIONS.has(type)) {
    const button = normalizeGamepadButton(action.button);
    const durationMs = Math.trunc(clampNumber(action.durationMs ?? action.duration_ms ?? action.ms, 0, MAX_DURATION_MS, DEFAULT_GAMEPAD_PRESS_MS));
    const actions = [{ type: "gamepad_button_down", button }];
    if (durationMs > 0) {
      actions.push({ type: "wait", ms: durationMs });
    }
    actions.push({ type: "gamepad_button_up", button });
    return actions;
  }

  if (GAMEPAD_RESET_FUNCTIONS.has(type)) {
    return [
      {
        type: "gamepad_reset"
      }
    ];
  }

  throw new Error(`Unsupported action type: ${type || "(missing)"}`);
}

function stripComment(line) {
  let quote = null;
  let escaped = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];

    if (escaped) {
      escaped = false;
      continue;
    }

    if (character === "\\") {
      escaped = true;
      continue;
    }

    if (quote) {
      if (character === quote) {
        quote = null;
      }
      continue;
    }

    if (character === "'" || character === "\"") {
      quote = character;
      continue;
    }

    if (character === "#") {
      return line.slice(0, index);
    }
  }

  return line;
}

function stripThinkingBlocks(text) {
  return String(text || "")
    .replace(THINKING_TAG_PATTERN, "")
    .replace(UNCLOSED_THINKING_BEFORE_CODE_PATTERN, "")
    .trim();
}

function splitArguments(argsText) {
  const args = [];
  let quote = null;
  let escaped = false;
  let depth = 0;
  let start = 0;

  for (let index = 0; index < argsText.length; index += 1) {
    const character = argsText[index];

    if (escaped) {
      escaped = false;
      continue;
    }

    if (character === "\\") {
      escaped = true;
      continue;
    }

    if (quote) {
      if (character === quote) {
        quote = null;
      }
      continue;
    }

    if (character === "'" || character === "\"") {
      quote = character;
      continue;
    }

    if (character === "(" || character === "[" || character === "{") {
      depth += 1;
      continue;
    }

    if (character === ")" || character === "]" || character === "}") {
      depth -= 1;
      continue;
    }

    if (character === "," && depth === 0) {
      args.push(argsText.slice(start, index).trim());
      start = index + 1;
    }
  }

  const tail = argsText.slice(start).trim();
  if (tail) {
    args.push(tail);
  }

  return args.filter(Boolean);
}

function splitKeyword(argument) {
  let quote = null;
  let escaped = false;
  let depth = 0;

  for (let index = 0; index < argument.length; index += 1) {
    const character = argument[index];

    if (escaped) {
      escaped = false;
      continue;
    }

    if (character === "\\") {
      escaped = true;
      continue;
    }

    if (quote) {
      if (character === quote) {
        quote = null;
      }
      continue;
    }

    if (character === "'" || character === "\"") {
      quote = character;
      continue;
    }

    if (character === "(" || character === "[" || character === "{") {
      depth += 1;
      continue;
    }

    if (character === ")" || character === "]" || character === "}") {
      depth -= 1;
      continue;
    }

    if (character === "=" && depth === 0) {
      const key = argument.slice(0, index).trim();
      const value = argument.slice(index + 1).trim();
      if (/^[a-zA-Z_]\w*$/.test(key)) {
        return [key, value];
      }
    }
  }

  return null;
}

function parsePythonString(value) {
  const quote = value[0];
  const inner = value.slice(1, -1);
  let result = "";
  let escaped = false;

  for (const character of inner) {
    if (escaped) {
      if (character === "n") {
        result += "\n";
      } else if (character === "r") {
        result += "\r";
      } else if (character === "t") {
        result += "\t";
      } else {
        result += character;
      }
      escaped = false;
      continue;
    }

    if (character === "\\") {
      escaped = true;
      continue;
    }

    result += character;
  }

  if (escaped) {
    result += "\\";
  }

  return result;
}

function parsePythonValue(rawValue) {
  const value = rawValue.trim();

  if (!value) {
    throw new Error("Empty function argument.");
  }

  if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
    return parsePythonString(value);
  }

  if (/^[+-]?(?:\d+\.?\d*|\.\d+)$/.test(value)) {
    return Number(value);
  }

  if (/^(true|false)$/i.test(value)) {
    return /^true$/i.test(value);
  }

  if (/^(none|null)$/i.test(value)) {
    return null;
  }

  if (/^[a-zA-Z_][\w+-]*$/.test(value)) {
    return value;
  }

  throw new Error(`Unsupported Python argument value: ${rawValue}`);
}

function parseCallArguments(argsText) {
  const positional = [];
  const keyword = {};

  for (const argument of splitArguments(argsText)) {
    const keywordParts = splitKeyword(argument);
    if (keywordParts) {
      const [key, value] = keywordParts;
      keyword[normalizeType(key)] = parsePythonValue(value);
    } else {
      positional.push(parsePythonValue(argument));
    }
  }

  return {
    positional,
    keyword
  };
}

function argValue(args, names, position, fallback = undefined) {
  for (const name of names) {
    const normalized = normalizeType(name);
    if (Object.prototype.hasOwnProperty.call(args.keyword, normalized)) {
      return args.keyword[normalized];
    }
  }

  if (position < args.positional.length) {
    return args.positional[position];
  }

  return fallback;
}

function gamepadShortcutDuration(args, functionName) {
  const keywordNames = Object.keys(args.keyword);
  const unsupportedKeywords = keywordNames.filter((key) => key !== "duration_ms" && key !== "ms");
  if (unsupportedKeywords.length > 0) {
    throw new Error(`${functionName} only accepts duration_ms.`);
  }
  if (args.positional.length > 1 || (args.positional.length > 0 && keywordNames.length > 0)) {
    throw new Error(`${functionName} requires a single duration_ms argument.`);
  }

  const duration = argValue(args, ["duration_ms", "durationMs", "ms"], 0);
  if (duration === undefined) {
    throw new Error(`${functionName} requires duration_ms.`);
  }

  const number = Number(duration);
  if (!Number.isFinite(number) || number <= 0) {
    throw new Error(`${functionName} duration_ms must be a positive number.`);
  }

  return Math.trunc(clampNumber(number, 1, MAX_DURATION_MS));
}

function gamepadShortcutAction(name, args) {
  const type = normalizeType(name);
  const durationMs = gamepadShortcutDuration(args, name);
  const stick = GTA_SHORTHAND_STICK_FUNCTIONS.get(type);
  if (stick) {
    return normalizeAction({
      type: "gamepad_stick",
      stick: stick.stick,
      x: stick.x,
      y: stick.y,
      durationMs
    });
  }

  const trigger = GTA_SHORTHAND_TRIGGER_FUNCTIONS.get(type);
  if (trigger) {
    return normalizeAction({
      type: "gamepad_trigger",
      trigger,
      value: 1,
      durationMs
    });
  }

  const button = GTA_SHORTHAND_BUTTON_FUNCTIONS.get(type);
  if (button) {
    return normalizeAction({
      type: "gamepad_press",
      button,
      durationMs
    });
  }

  return null;
}

function actionFromPythonCall(name, args) {
  const type = normalizeType(name);

  if (GTA_SHORTHAND_FUNCTIONS.has(type)) {
    return gamepadShortcutAction(name, args);
  }

  if (type === "mouse_move" || type === "move_mouse" || type === "move") {
    return normalizeAction({
      type: "mouse_move",
      dx: argValue(args, ["dx", "x"], 0, 0),
      dy: argValue(args, ["dy", "y"], 1, 0),
      durationMs: argValue(args, ["duration_ms", "durationMs", "ms"], 2, 0)
    });
  }

  if (type === "mouse_down" || type === "button_down") {
    return normalizeAction({
      type: "mouse_down",
      button: argValue(args, ["button"], 0, "left")
    });
  }

  if (type === "mouse_up" || type === "button_up") {
    return normalizeAction({
      type: "mouse_up",
      button: argValue(args, ["button"], 0, "left")
    });
  }

  if (type === "click" || type === "mouse_click") {
    return normalizeAction({
      type: "click",
      button: argValue(args, ["button"], 0, "left")
    });
  }

  if (type === "key_down" || type === "keydown") {
    return normalizeAction({
      type: "key_down",
      key: argValue(args, ["key"], 0)
    });
  }

  if (type === "key_up" || type === "keyup") {
    return normalizeAction({
      type: "key_up",
      key: argValue(args, ["key"], 0)
    });
  }

  if (type === "key_press" || type === "keypress" || type === "press") {
    return normalizeAction({
      type: "key_press",
      key: argValue(args, ["key"], 0)
    });
  }

  if (type === "wait" || type === "sleep") {
    return normalizeAction({
      type: "wait",
      ms: argValue(args, ["ms", "duration_ms", "durationMs"], 0, 100)
    });
  }

  if (type === "wheel" || type === "mouse_wheel") {
    return normalizeAction({
      type: "wheel",
      amount: argValue(args, ["amount", "delta", "dy"], 0, 0)
    });
  }

  if (GAMEPAD_STICK_FUNCTIONS.has(type)) {
    return normalizeAction({
      type: "gamepad_stick",
      stick: argValue(args, ["stick"], 0, "left"),
      x: argValue(args, ["x", "dx"], 1, 0),
      y: argValue(args, ["y", "dy"], 2, 0),
      durationMs: argValue(args, ["duration_ms", "durationMs", "ms"], 3, 0)
    });
  }

  if (GAMEPAD_TRIGGER_FUNCTIONS.has(type)) {
    return normalizeAction({
      type: "gamepad_trigger",
      trigger: argValue(args, ["trigger"], 0, "right"),
      value: argValue(args, ["value", "amount"], 1, 0),
      durationMs: argValue(args, ["duration_ms", "durationMs", "ms"], 2, 0)
    });
  }

  if (GAMEPAD_BUTTON_DOWN_FUNCTIONS.has(type)) {
    return normalizeAction({
      type: "gamepad_down",
      button: argValue(args, ["button"], 0)
    });
  }

  if (GAMEPAD_BUTTON_UP_FUNCTIONS.has(type)) {
    return normalizeAction({
      type: "gamepad_up",
      button: argValue(args, ["button"], 0)
    });
  }

  if (GAMEPAD_BUTTON_PRESS_FUNCTIONS.has(type)) {
    return normalizeAction({
      type: "gamepad_press",
      button: argValue(args, ["button"], 0),
      durationMs: argValue(args, ["duration_ms", "durationMs", "ms"], 1, DEFAULT_GAMEPAD_PRESS_MS)
    });
  }

  if (GAMEPAD_RESET_FUNCTIONS.has(type)) {
    return normalizeAction({
      type: "gamepad_reset"
    });
  }

  throw new Error(`Unsupported Python GUI function: ${name}`);
}

function annotateActionSource(actions, sourceCall, displayCall = "") {
  for (const action of actions) {
    Object.defineProperty(action, "sourceCall", {
      value: sourceCall,
      enumerable: false,
      configurable: true
    });
    if (displayCall) {
      Object.defineProperty(action, "displayCall", {
        value: displayCall,
        enumerable: false,
        configurable: true
      });
    }
  }

  return actions;
}

function parsePythonFunctionCalls(text) {
  const actions = [];
  const errors = [];

  for (const line of String(text || "").split(/\r?\n/)) {
    const trimmed = stripComment(line).trim();
    if (!trimmed || trimmed === "```" || /^```[a-zA-Z0-9_-]*$/.test(trimmed)) {
      continue;
    }

    const match = trimmed.match(/^([a-zA-Z_]\w*)\s*\(([\s\S]*)\)\s*$/);
    if (!match) {
      continue;
    }

    const name = match[1];
    const type = normalizeType(name);
    if (!ALLOWED_FUNCTIONS.has(type)) {
      errors.push(`Unsupported function: ${name}`);
      continue;
    }

    try {
      const parsedActions = actionFromPythonCall(name, parseCallArguments(match[2]));
      const displayCall = GTA_SHORTHAND_FUNCTIONS.has(type) ? trimmed : "";
      actions.push(...annotateActionSource(parsedActions, trimmed, displayCall));
    } catch (error) {
      errors.push(error.message);
    }
  }

  if (actions.length > 0) {
    return actions;
  }

  if (errors.length > 0) {
    throw new Error(errors.join("; "));
  }

  return [];
}

function extractPythonCodeBlock(text) {
  const trimmed = stripThinkingBlocks(text);
  if (!trimmed) {
    throw new Error("Model response is empty.");
  }

  const match = trimmed.match(/```python[ \t]*\r?\n([\s\S]*?)\r?\n?```/i);
  if (!match) {
    throw new Error("Model response must contain a complete fenced ```python code block ending with ```.");
  }

  return match[1];
}

function validatePythonGuiOnly(code) {
  const errors = [];

  for (const line of String(code || "").split(/\r?\n/)) {
    const original = line.trim();
    const trimmed = stripComment(line).trim();
    if (!original) {
      continue;
    }

    if (!trimmed) {
      errors.push(`Unsupported Python GUI statement: ${original}`);
      continue;
    }

    if (trimmed !== original) {
      errors.push(`Unsupported Python GUI statement: ${original}`);
      continue;
    }

    const call = trimmed.match(/^([a-zA-Z_]\w*)\s*\(([\s\S]*)\)\s*$/);
    if (!call || !ALLOWED_FUNCTIONS.has(normalizeType(call[1]))) {
      errors.push(`Unsupported Python GUI statement: ${original}`);
    }
  }

  if (errors.length > 0) {
    throw new Error(errors.join("; "));
  }
}

function parseActions(rawText) {
  const code = extractPythonCodeBlock(rawText);
  validatePythonGuiOnly(code);
  const actions = parsePythonFunctionCalls(code);
  if (actions.length === 0) {
    throw new Error("No Python GUI function calls found in model response.");
  }

  return actions;
}

function describeAction(action) {
  if (!action || typeof action !== "object") {
    return String(action);
  }

  if (action.type === "mouse_move") {
    return `mouse_move dx=${action.dx} dy=${action.dy}`;
  }

  if (action.type === "mouse_down" || action.type === "mouse_up") {
    return `${action.type} ${action.button}`;
  }

  if (action.type === "key_down" || action.type === "key_up") {
    return `${action.type} ${action.key}`;
  }

  if (action.type === "wait") {
    return `wait ${action.ms}ms`;
  }

  if (action.type === "wheel") {
    return `wheel ${action.amount}`;
  }

  if (action.type === "gamepad_stick") {
    const duration = action.durationMs ? ` ${action.durationMs}ms` : "";
    return `gamepad_stick ${action.stick} x=${action.x} y=${action.y}${duration}`;
  }

  if (action.type === "gamepad_trigger") {
    const duration = action.durationMs ? ` ${action.durationMs}ms` : "";
    return `gamepad_trigger ${action.trigger} value=${action.value}${duration}`;
  }

  if (action.type === "gamepad_button_down" || action.type === "gamepad_button_up") {
    return `${action.type} ${action.button}`;
  }

  if (action.type === "gamepad_reset") {
    return "gamepad_reset";
  }

  return JSON.stringify(action);
}

module.exports = {
  parseActions,
  describeAction,
  normalizeAction,
  extractPythonCodeBlock,
  parsePythonFunctionCalls,
  stripThinkingBlocks
};

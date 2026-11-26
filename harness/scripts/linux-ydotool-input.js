const { spawn, spawnSync } = require("node:child_process");
const readline = require("node:readline");

const KEY_CODES = {
  escape: 1,
  esc: 1,
  "1": 2,
  "2": 3,
  "3": 4,
  "4": 5,
  "5": 6,
  "6": 7,
  "7": 8,
  "8": 9,
  "9": 10,
  "0": 11,
  minus: 12,
  equal: 13,
  backspace: 14,
  tab: 15,
  q: 16,
  w: 17,
  e: 18,
  r: 19,
  t: 20,
  y: 21,
  u: 22,
  i: 23,
  o: 24,
  p: 25,
  leftbracket: 26,
  bracketleft: 26,
  rightbracket: 27,
  bracketright: 27,
  enter: 28,
  return: 28,
  ctrl: 29,
  control: 29,
  a: 30,
  s: 31,
  d: 32,
  f: 33,
  g: 34,
  h: 35,
  j: 36,
  k: 37,
  l: 38,
  semicolon: 39,
  quote: 40,
  apostrophe: 40,
  backquote: 41,
  grave: 41,
  shift: 42,
  backslash: 43,
  z: 44,
  x: 45,
  c: 46,
  v: 47,
  b: 48,
  n: 49,
  m: 50,
  comma: 51,
  period: 52,
  slash: 53,
  multiply: 55,
  alt: 56,
  space: 57,
  capslock: 58,
  f1: 59,
  f2: 60,
  f3: 61,
  f4: 62,
  f5: 63,
  f6: 64,
  f7: 65,
  f8: 66,
  f9: 67,
  f10: 68,
  numlock: 69,
  scrolllock: 70,
  numpad7: 71,
  numpad8: 72,
  numpad9: 73,
  subtract: 74,
  numpad4: 75,
  numpad5: 76,
  numpad6: 77,
  add: 78,
  numpad1: 79,
  numpad2: 80,
  numpad3: 81,
  numpad0: 82,
  decimal: 83,
  f11: 87,
  f12: 88,
  home: 102,
  up: 103,
  pageup: 104,
  left: 105,
  right: 106,
  end: 107,
  down: 108,
  pagedown: 109,
  insert: 110,
  delete: 111,
  del: 111,
  meta: 125,
  win: 125,
  windows: 125,
  divide: 98
};

const BUTTON_CODES = {
  left: 0,
  right: 1,
  middle: 2
};

let mousemoveSyntax = null;

function writeJson(object) {
  process.stdout.write(`${JSON.stringify(object)}\n`);
}

function failStartup(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function checkCommand(name) {
  const result = spawnSync("sh", ["-lc", `command -v ${name}`], {
    encoding: "utf8",
    timeout: 3000
  });
  return result.status === 0;
}

function ydotoolStatus(args, timeout = 2000) {
  return spawnSync("ydotool", args, {
    encoding: "utf8",
    timeout
  });
}

function commandOutput(result) {
  return `${result.stdout || ""}\n${result.stderr || ""}`;
}

function ydotoolHelpText() {
  return [
    ydotoolStatus(["mousemove", "--help"], 1000),
    ydotoolStatus(["--help"], 1000)
  ].map(commandOutput).join("\n");
}

function failIfLegacyYdotool(helpText) {
  if (/mousemove_relative/.test(helpText)) {
    failStartup("linux-ydotool-input requires ydotool v1.0+ syntax. This ydotool appears to use the older mousemove_relative/key-name command set.");
  }
}

function checkYdotoolDaemon() {
  const flagged = ydotoolStatus(["mousemove", "-x", "0", "-y", "0"]);
  if (flagged.status === 0) {
    mousemoveSyntax = "flags";
    return;
  }
  let lastResult = flagged;

  const helpText = ydotoolHelpText();
  failIfLegacyYdotool(helpText);

  const positional = ydotoolStatus(["mousemove", "--", "0", "0"]);
  if (positional.status === 0) {
    mousemoveSyntax = "positional";
    return;
  }
  lastResult = positional;

  const result = lastResult;
  if (result.status !== 0) {
    const message = result.error?.message || result.stderr?.trim() || result.stdout?.trim() || "ydotool daemon check failed";
    failStartup(`linux-ydotool-input could not talk to ydotoold with ydotool v1 mousemove syntax: ${message}`);
  }
}

function normalizeKey(key) {
  const normalized = String(key || "").trim().toLowerCase().replace(/\s+/g, "");
  if (Object.prototype.hasOwnProperty.call(KEY_CODES, normalized)) {
    return KEY_CODES[normalized];
  }
  throw new Error(`Unsupported key: ${key}`);
}

function buttonCode(button, down) {
  const normalized = String(button || "left").trim().toLowerCase();
  if (!Object.prototype.hasOwnProperty.call(BUTTON_CODES, normalized)) {
    throw new Error(`Unsupported mouse button: ${button}`);
  }
  return `0x${(BUTTON_CODES[normalized] + (down ? 0x40 : 0x80)).toString(16)}`;
}

function runYdotool(args) {
  return new Promise((resolve, reject) => {
    const child = spawn("ydotool", args, {
      stdio: ["ignore", "pipe", "pipe"]
    });
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error(`ydotool command timed out: ${args.join(" ")}`));
    }, 3000);
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.once("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.once("exit", (code) => {
      clearTimeout(timeout);
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(stderr.trim() || `ydotool exited with code ${code}`));
      }
    });
  });
}

async function runMouseMove(dx, dy) {
  const x = String(Math.trunc(Number(dx) || 0));
  const y = String(Math.trunc(Number(dy) || 0));

  if (mousemoveSyntax === "flags") {
    await runYdotool(["mousemove", "-x", x, "-y", y]);
    return;
  }

  await runYdotool(["mousemove", "--", x, y]);
}

async function handle(message) {
  switch (message.type) {
    case "get_cursor":
      throw new Error("ydotool cannot read cursor position on generic Wayland sessions.");
    case "key_down":
      await runYdotool(["key", `${normalizeKey(message.key)}:1`]);
      return {};
    case "key_up":
      await runYdotool(["key", `${normalizeKey(message.key)}:0`]);
      return {};
    case "mouse_move":
      await runMouseMove(message.dx, message.dy);
      return {};
    case "mouse_down":
      await runYdotool(["click", buttonCode(message.button, true)]);
      return {};
    case "mouse_up":
      await runYdotool(["click", buttonCode(message.button, false)]);
      return {};
    case "wheel":
      throw new Error("ydotool wheel events are not implemented by this backend.");
    default:
      throw new Error(`Unsupported input command: ${message.type}`);
  }
}

if (!checkCommand("ydotool")) {
  failStartup("linux-ydotool-input requires ydotool and a running ydotoold daemon.");
}

checkYdotoolDaemon();

writeJson({
  event: "ready",
  backend: "linux-uinput-ydotool",
  warning: "cursor position and wheel events are unavailable in this backend"
});

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", async (line) => {
  if (!line.trim()) {
    return;
  }

  let message = null;
  try {
    message = JSON.parse(line);
    const result = await handle(message);
    writeJson({ id: message.id ?? null, ok: true, ...result });
  } catch (error) {
    writeJson({ id: message?.id ?? null, ok: false, error: error.message });
  }
});

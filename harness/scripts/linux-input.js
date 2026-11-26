const { spawn, spawnSync } = require("node:child_process");
const readline = require("node:readline");

const BUTTONS = {
  left: "1",
  middle: "2",
  right: "3"
};

const KEY_NAMES = {
  enter: "Return",
  return: "Return",
  escape: "Escape",
  esc: "Escape",
  tab: "Tab",
  space: "space",
  backspace: "BackSpace",
  delete: "Delete",
  del: "Delete",
  insert: "Insert",
  home: "Home",
  end: "End",
  pageup: "Page_Up",
  pagedown: "Page_Down",
  left: "Left",
  right: "Right",
  up: "Up",
  down: "Down",
  shift: "Shift_L",
  ctrl: "Control_L",
  control: "Control_L",
  alt: "Alt_L",
  meta: "Super_L",
  win: "Super_L",
  windows: "Super_L",
  pause: "Pause",
  capslock: "Caps_Lock",
  numlock: "Num_Lock",
  scrolllock: "Scroll_Lock",
  minus: "minus",
  equal: "equal",
  comma: "comma",
  period: "period",
  slash: "slash",
  backslash: "backslash",
  semicolon: "semicolon",
  quote: "apostrophe",
  apostrophe: "apostrophe",
  backquote: "grave",
  grave: "grave",
  leftbracket: "bracketleft",
  rightbracket: "bracketright",
  bracketleft: "bracketleft",
  bracketright: "bracketright",
  numpad0: "KP_0",
  numpad1: "KP_1",
  numpad2: "KP_2",
  numpad3: "KP_3",
  numpad4: "KP_4",
  numpad5: "KP_5",
  numpad6: "KP_6",
  numpad7: "KP_7",
  numpad8: "KP_8",
  numpad9: "KP_9",
  add: "KP_Add",
  subtract: "KP_Subtract",
  multiply: "KP_Multiply",
  divide: "KP_Divide",
  decimal: "KP_Decimal"
};

for (let i = 1; i <= 12; i += 1) {
  KEY_NAMES[`f${i}`] = `F${i}`;
}

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

function normalizeKey(key) {
  const normalized = String(key || "").trim().toLowerCase().replace(/\s+/g, "");
  if (/^[a-z0-9]$/.test(normalized)) {
    return normalized;
  }
  if (Object.prototype.hasOwnProperty.call(KEY_NAMES, normalized)) {
    return KEY_NAMES[normalized];
  }
  throw new Error(`Unsupported key: ${key}`);
}

function runXdotool(args) {
  return new Promise((resolve, reject) => {
    const child = spawn("xdotool", args, {
      stdio: ["ignore", "pipe", "pipe"]
    });
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error(`xdotool command timed out: ${args.join(" ")}`));
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
        reject(new Error(stderr.trim() || `xdotool exited with code ${code}`));
      }
    });
  });
}

function parseCursor(output) {
  const values = {};
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(/^([A-Z]+)=(.*)$/);
    if (match) {
      values[match[1]] = match[2];
    }
  }
  return {
    x: Number(values.X),
    y: Number(values.Y)
  };
}

async function handle(message) {
  switch (message.type) {
    case "get_cursor":
      return parseCursor(await runXdotool(["getmouselocation", "--shell"]));
    case "key_down":
      await runXdotool(["keydown", normalizeKey(message.key)]);
      return {};
    case "key_up":
      await runXdotool(["keyup", normalizeKey(message.key)]);
      return {};
    case "mouse_move":
      await runXdotool(["mousemove_relative", "--", String(Math.trunc(Number(message.dx) || 0)), String(Math.trunc(Number(message.dy) || 0))]);
      return {};
    case "mouse_down":
      await runXdotool(["mousedown", BUTTONS[String(message.button || "left").toLowerCase()] || BUTTONS.left]);
      return {};
    case "mouse_up":
      await runXdotool(["mouseup", BUTTONS[String(message.button || "left").toLowerCase()] || BUTTONS.left]);
      return {};
    case "wheel": {
      const amount = Number(message.amount) || 0;
      const button = amount >= 0 ? "4" : "5";
      const clicks = Math.max(1, Math.min(20, Math.round(Math.abs(amount) / 120)));
      for (let index = 0; index < clicks; index += 1) {
        await runXdotool(["click", button]);
      }
      return {};
    }
    default:
      throw new Error(`Unsupported input command: ${message.type}`);
  }
}

if (!process.env.DISPLAY) {
  failStartup("linux-input requires an X11 DISPLAY. Wayland sessions need XWayland access or an alternative input backend.");
}

if (!checkCommand("xdotool")) {
  failStartup("linux-input requires xdotool. Install it with your distro package manager, for example: sudo apt install xdotool");
}

writeJson({ event: "ready", backend: "linux-x11-xdotool" });

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

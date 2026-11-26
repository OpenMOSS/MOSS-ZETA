const { EventEmitter } = require("node:events");
const readline = require("node:readline");
const { spawn } = require("node:child_process");
const { keyToVirtualKey, normalizeKeyName } = require("./keyMap");
const { isWaylandSession } = require("./linuxScreenshotBackend");
const { scriptPath } = require("./projectPaths");
const { performance } = require("node:perf_hooks");

function defaultBackendFor(platform = process.platform) {
  if (platform === "win32") {
    const requestedBackend = String(process.env.AGI_GTA_WINDOWS_INPUT || "gamepad").trim().toLowerCase();
    if (!["gamepad", "x360", "vgamepad", "user32", "keyboard-mouse", "keyboard_mouse"].includes(requestedBackend)) {
      throw new Error(`Unsupported AGI_GTA_WINDOWS_INPUT: ${requestedBackend}. Expected gamepad, x360, vgamepad, user32, keyboard-mouse, or keyboard_mouse.`);
    }
    if (requestedBackend === "gamepad" || requestedBackend === "x360" || requestedBackend === "vgamepad") {
      return {
        name: "windows-vgamepad-x360",
        command: process.env.AGI_GTA_GAMEPAD_PYTHON || "python",
        args: [scriptPath("windows-gamepad-input.py")],
        windowsHide: true
      };
    }

    return {
      name: "windows-powershell-user32",
      command: "powershell.exe",
      args: [
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        scriptPath("windows-input.ps1")
      ],
      windowsHide: true
    };
  }

  if (platform === "darwin") {
    return {
      name: "macos-swift-coregraphics",
      command: "swift",
      args: [scriptPath("macos-input.swift")],
      windowsHide: false
    };
  }

  if (platform === "linux") {
    const requestedBackend = String(process.env.AGI_GTA_LINUX_INPUT || "").trim().toLowerCase();
    if (requestedBackend && !["ydotool", "xdotool", "x11", "python-xtest", "xtest"].includes(requestedBackend)) {
      throw new Error(`Unsupported AGI_GTA_LINUX_INPUT: ${requestedBackend}. Expected ydotool, xdotool, x11, python-xtest, or xtest.`);
    }
    if (requestedBackend === "python-xtest" || requestedBackend === "xtest") {
      return {
        name: "linux-x11-python-xtest",
        command: "python3",
        args: [scriptPath("linux-xtest-input.py")],
        windowsHide: false
      };
    }
    const preferYdotool = requestedBackend === "ydotool" || (!requestedBackend && isWaylandSession());
    if (preferYdotool) {
      return {
        name: "linux-uinput-ydotool",
        command: process.execPath,
        args: [scriptPath("linux-ydotool-input.js")],
        windowsHide: false
      };
    }

    return {
      name: "linux-x11-xdotool",
      command: process.execPath,
      args: [scriptPath("linux-input.js")],
      windowsHide: false
    };
  }

  throw new Error(`Unsupported platform for input control: ${platform}`);
}

function capabilitiesForBackend(backend) {
  const name = typeof backend === "string" ? backend : backend?.name || "";
  const ydotool = name === "linux-uinput-ydotool";
  const gamepad = name === "windows-vgamepad-x360";

  return {
    keyboard: true,
    mouseMove: true,
    mouseButtons: true,
    wheel: !ydotool,
    cursorPosition: !ydotool,
    gamepad
  };
}

class InputController extends EventEmitter {
  constructor(options = {}) {
    super();
    this.platform = options.platform || process.platform;
    this.backend = options.backend || null;
    this.child = null;
    this.readline = null;
    this.ready = false;
    this.startPromise = null;
    this.startCleanup = null;
    this.exitHandled = false;
    this.nextId = 1;
    this.pending = new Map();
    this.realtimePackageCompletionWaiters = new Map();
    this.realtimePackageCompletions = new Map();
  }

  selectedBackend() {
    if (!this.backend) {
      this.backend = defaultBackendFor(this.platform);
    }

    return this.backend;
  }

  capabilities() {
    return capabilitiesForBackend(this.selectedBackend());
  }

  assertCapability(capability, label) {
    const backend = this.selectedBackend();
    const capabilities = capabilitiesForBackend(backend);
    if (capabilities?.[capability] === false) {
      throw new Error(`${label} is unavailable on input backend ${backend.name}.`);
    }
  }

  async ensureStarted() {
    const backend = this.selectedBackend();

    if (this.ready && this.child && !this.child.killed) {
      return;
    }

    if (this.startPromise) {
      return this.startPromise;
    }

    this.exitHandled = false;
    this.startPromise = new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timeout);
        this.off("_ready", onReady);
        this.off("_startError", onStartError);
        this.startCleanup = null;
      };
      const onReady = () => {
        cleanup();
        this.startPromise = null;
        resolve();
      };
      const onStartError = (error) => {
        cleanup();
        this.startPromise = null;
        reject(error);
      };
      const timeout = setTimeout(() => {
        cleanup();
        this.startPromise = null;
        if (this.child && !this.child.killed) {
          this.child.kill();
        }
        reject(new Error("Timed out waiting for input helper readiness."));
      }, 5000);

      this.startCleanup = cleanup;
      this.once("_ready", onReady);
      this.once("_startError", onStartError);
    });

    this.child = spawn(backend.command, backend.args, {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: Boolean(backend.windowsHide)
    });

    this.readline = readline.createInterface({ input: this.child.stdout });
    this.readline.on("line", (line) => this.handleLine(line));

    this.child.stderr.on("data", (chunk) => {
      const message = chunk.toString().trim();
      if (message) {
        this.emit("log", `input helper: ${message}`);
      }
    });

    this.child.once("error", (error) => {
      this.handleExit(error);
    });

    this.child.once("exit", (code, signal) => {
      this.handleExit(new Error(`Input helper exited with code ${code ?? "null"} signal ${signal ?? "null"}.`));
    });

    return this.startPromise;
  }

  handleLine(line) {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }

    let message;
    try {
      message = JSON.parse(trimmed);
    } catch (error) {
      this.emit("log", `input helper output: ${trimmed}`);
      return;
    }

    if (message.event === "ready") {
      this.ready = true;
      this.fixedCadenceSupported = message.fixedCadence === true;
      this.streamCadenceSupported = message.streamCadence === true;
      this.streamHandoffSupported = message.streamHandoff === "atomic-owner-v1";
      this.emit("_ready");
      this.emit("log", `input helper ready (${message.backend || this.backend.name})`);
      if (message.warning) {
        this.emit("log", `input helper warning: ${message.warning}`);
      }
      return;
    }

    if (message.event === "error") {
      this.emit("log", `input helper error: ${message.error || "unknown error"}`);
      return;
    }

    if (
      message.event === "gamepad_package_completed"
      || message.event === "gamepad_package_cancelled"
    ) {
      this.handleRealtimePackageCompletion(message);
      this.emit("event", message);
      return;
    }

    const pending = this.pending.get(String(message.id));
    if (!pending) {
      this.emit("event", message);
      return;
    }

    clearTimeout(pending.timeout);
    this.pending.delete(String(message.id));

    if (message.ok) {
      pending.resolve(message);
    } else {
      pending.reject(new Error(message.error || "Input helper command failed."));
    }
  }

  handleExit(error) {
    if (this.exitHandled) {
      return;
    }

    this.exitHandled = true;
    const wasReady = this.ready;
    this.ready = false;
    this.startPromise = null;
    this.child = null;

    if (this.readline) {
      this.readline.removeAllListeners();
      this.readline.close();
      this.readline = null;
    }

    if (!wasReady) {
      this.emit("_startError", error);
    }

    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
    this.pending.clear();
    for (const waiter of this.realtimePackageCompletionWaiters.values()) {
      clearTimeout(waiter.timeout);
      waiter.reject(error);
    }
    this.realtimePackageCompletionWaiters.clear();
    this.realtimePackageCompletions.clear();

    this.emit("log", error.message);
  }

  handleRealtimePackageCompletion(message) {
    const packageSeq = String(message.packageSeq ?? "");
    if (!packageSeq) {
      this.emit("log", "input helper realtime package completion did not include packageSeq");
      return;
    }

    const waiter = this.realtimePackageCompletionWaiters.get(packageSeq);
    if (waiter) {
      clearTimeout(waiter.timeout);
      this.realtimePackageCompletionWaiters.delete(packageSeq);
      waiter.resolve(message);
      return;
    }

    this.realtimePackageCompletions.set(packageSeq, message);
    if (this.realtimePackageCompletions.size > 32) {
      const oldest = this.realtimePackageCompletions.keys().next().value;
      this.realtimePackageCompletions.delete(oldest);
    }
  }

  async send(type, payload = {}, timeoutMs = 5000) {
    await this.ensureStarted();

    if (!this.child || !this.child.stdin.writable) {
      throw new Error("Input helper is not writable.");
    }

    const id = String(this.nextId++);
    const message = {
      id,
      type,
      ...payload
    };

    const responsePromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Input helper command timed out: ${type}`));
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timeout });
    });

    this.child.stdin.write(`${JSON.stringify(message)}\n`);
    return responsePromise;
  }

  async getCursorPosition() {
    this.assertCapability("cursorPosition", "Cursor position");
    const response = await this.send("get_cursor");
    return {
      x: Number(response.x),
      y: Number(response.y)
    };
  }

  async keyDown(key) {
    this.assertCapability("keyboard", "Keyboard input");
    const normalized = normalizeKeyName(key);
    const payload = { key: normalized };
    if (this.platform === "win32") {
      payload.vk = keyToVirtualKey(normalized);
    }

    await this.send("key_down", payload);
  }

  async keyUp(key) {
    this.assertCapability("keyboard", "Keyboard input");
    const normalized = normalizeKeyName(key);
    const payload = { key: normalized };
    if (this.platform === "win32") {
      payload.vk = keyToVirtualKey(normalized);
    }

    await this.send("key_up", payload);
  }

  async mouseMove(dx, dy) {
    this.assertCapability("mouseMove", "Mouse movement");
    await this.send("mouse_move", { dx, dy });
  }

  async mouseDown(button) {
    this.assertCapability("mouseButtons", "Mouse button input");
    await this.send("mouse_down", { button });
  }

  async mouseUp(button) {
    this.assertCapability("mouseButtons", "Mouse button input");
    await this.send("mouse_up", { button });
  }

  async wheel(amount) {
    this.assertCapability("wheel", "Mouse wheel input");
    await this.send("wheel", { amount });
  }

  async gamepadStick(stick, x, y) {
    this.assertCapability("gamepad", "Gamepad input");
    await this.send("gamepad_stick", { stick, x, y });
  }

  async gamepadTrigger(trigger, value) {
    this.assertCapability("gamepad", "Gamepad input");
    await this.send("gamepad_trigger", { trigger, value });
  }

  async gamepadButtonDown(button) {
    this.assertCapability("gamepad", "Gamepad input");
    await this.send("gamepad_button_down", { button });
  }

  async gamepadButtonUp(button) {
    this.assertCapability("gamepad", "Gamepad input");
    await this.send("gamepad_button_up", { button });
  }

  async gamepadState(state) {
    this.assertCapability("gamepad", "Gamepad input");
    const started = performance.now();
    const response = await this.send("gamepad_state", { state });
    return {
      ipcRoundTripMs: Number((performance.now() - started).toFixed(3)),
      helperStateMs: Number.isFinite(Number(response.gamepadStateMs))
        ? Number(response.gamepadStateMs)
        : null,
      gamepadUpdateMs: Number.isFinite(Number(response.gamepadUpdateMs))
        ? Number(response.gamepadUpdateMs)
        : null
    };
  }

  async gamepadPackage(states, options = {}) {
    this.assertCapability("gamepad", "Gamepad input");
    await this.send("gamepad_package", {
      states,
      chunkDurationMs: options.chunkDurationMs,
      resetAfter: Boolean(options.resetAfter)
    });
  }

  supportsRealtimeGamepadPackage() {
    return this.selectedBackend()?.name === "windows-vgamepad-x360";
  }

  supportsFixedCadenceGamepad() {
    return this.supportsRealtimeGamepadPackage() && this.fixedCadenceSupported === true;
  }

  async calibrateGamepadClock() {
    if (!this.supportsFixedCadenceGamepad()) throw Error('Input helper lacks fixed cadence support; restart it');
    const samples = [];
    for (let i = 0; i < 5; i++) {
      const start = performance.now();
      const reply = await this.send('gamepad_clock');
      const end = performance.now();
      samples.push({ offsetMs: reply.helperPerfMs - (start + end) / 2, roundTripMs: end - start,
        timerResolutionMs: reply.timerResolutionMs ?? null });
    }
    const best = samples.sort((a, b) => a.roundTripMs - b.roundTripMs)[0];
    return { ...best, uncertaintyMs: best.roundTripMs / 2 };
  }

  gamepadPackageCadence(states, options) {
    return this.send('gamepad_package_cadence', { ...options, states });
  }

  cancelGamepadCadence(pendingOnly = false) {
    return this.send('gamepad_cadence_cancel', { pendingOnly });
  }

  async gamepadPackageRealtime(states, options = {}) {
    this.assertCapability("gamepad", "Gamepad input");
    return this.send("gamepad_package_realtime", {
      packageSeq: options.packageSeq ?? null,
      states,
      chunkDurationMs: options.chunkDurationMs,
      resetAfter: Boolean(options.resetAfter),
      idleHoldLeftTrigger: Boolean(options.idleHoldLeftTrigger),
      ...(options.idleState ? { idleState: options.idleState } : {})
    });
  }

  waitForRealtimeGamepadPackageCompletion(packageSeq, timeoutMs = 5000) {
    const key = String(packageSeq ?? "");
    if (!key) {
      return Promise.reject(new Error("Realtime gamepad package completion requires packageSeq."));
    }

    const completed = this.realtimePackageCompletions.get(key);
    if (completed) {
      this.realtimePackageCompletions.delete(key);
      return Promise.resolve(completed);
    }

    if (this.realtimePackageCompletionWaiters.has(key)) {
      return Promise.reject(new Error(`Realtime gamepad package completion is already being awaited: ${key}`));
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.realtimePackageCompletionWaiters.delete(key);
        reject(new Error(`Timed out waiting for realtime gamepad package completion: ${key}`));
      }, timeoutMs);
      this.realtimePackageCompletionWaiters.set(key, { resolve, reject, timeout });
    });
  }

  async cancelRealtimeGamepadPackage(packageSeq = null) {
    this.assertCapability("gamepad", "Gamepad input");
    return this.send("gamepad_package_realtime_cancel", {
      packageSeq
    });
  }

  async gamepadReset() {
    this.assertCapability("gamepad", "Gamepad input");
    const started = performance.now();
    const response = await this.send("gamepad_reset");
    return {
      ipcRoundTripMs: Number((performance.now() - started).toFixed(3)),
      gamepadUpdateMs: Number.isFinite(Number(response.gamepadUpdateMs))
        ? Number(response.gamepadUpdateMs)
        : null
    };
  }

  close() {
    if (this.child && !this.child.killed) {
      this.child.kill();
    }
  }
}

module.exports = {
  InputController,
  capabilitiesForBackend,
  defaultBackendFor
};

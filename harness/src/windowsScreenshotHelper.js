const fs = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const { spawn, execFile } = require("node:child_process");
const readline = require("node:readline");
const { scriptPath } = require("./projectPaths");

const HELPER_SOURCE = scriptPath("windows-screenshot-helper.cs");
const HELPER_MANIFEST = scriptPath("windows-screenshot-helper.manifest");
const HELPER_TIMEOUT_MS = 5000;

let compilePromise = null;
let helper = null;
let startPromise = null;
let nextRequestId = 1;
const pending = new Map();

function isWindows() {
  return process.platform === "win32";
}

function execFileText(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    execFile(command, args, {
      encoding: "utf8",
      timeout: options.timeout || 10000,
      windowsHide: true
    }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(stderr?.trim() || error.message));
        return;
      }
      resolve(stdout);
    });
  });
}

async function findCsc() {
  const windowsRoot = process.env.WINDIR || process.env.SystemRoot;
  const roots = windowsRoot
    ? ["Framework64", "Framework"].map((framework) => path.join(windowsRoot, "Microsoft.NET", framework))
    : [];
  const candidates = [];
  for (const root of roots) {
    let entries = [];
    try {
      entries = await fs.readdir(root, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (!entry.isDirectory()) {
        continue;
      }
      const candidate = path.join(root, entry.name, "csc.exe");
      try {
        await fs.access(candidate);
        candidates.push(candidate);
      } catch {
        // Try the next framework version.
      }
    }
  }
  candidates.sort((left, right) => right.localeCompare(left, undefined, { numeric: true }));
  if (candidates.length === 0) {
    throw new Error("Windows screenshot helper requires csc.exe from the .NET Framework.");
  }
  return candidates[0];
}

async function compileHelper() {
  if (!isWindows()) {
    throw new Error("The direct Windows screenshot helper is only available on Windows.");
  }

  const sourceStat = await fs.stat(HELPER_SOURCE);
  const manifestStat = await fs.stat(HELPER_MANIFEST);
  const stamp = `${Math.trunc(sourceStat.mtimeMs)}-${Math.trunc(manifestStat.mtimeMs)}`;
  const output = path.join(os.tmpdir(), `agi-gta-windows-screenshot-helper-${stamp}.exe`);
  try {
    await fs.access(output);
    return output;
  } catch {
    // Compile below.
  }

  const csc = await findCsc();
  await execFileText(csc, [
    "/nologo",
    "/target:exe",
    `/out:${output}`,
    `/win32manifest:${HELPER_MANIFEST}`,
    "/r:System.Drawing.dll",
    HELPER_SOURCE
  ], { timeout: 30000 });
  return output;
}

function rejectPending(error) {
  for (const request of pending.values()) {
    clearTimeout(request.timeout);
    request.reject(error);
  }
  pending.clear();
}

function closeHelper(error = new Error("Windows screenshot helper stopped.")) {
  const child = helper;
  helper = null;
  startPromise = null;
  if (child && !child.killed) {
    child.kill();
  }
  rejectPending(error);
}

function numericTiming(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Number(number.toFixed(3)) : null;
}

function helperTimingFromResponse(response, request, receivedAtMs) {
  const helperTotalMs = numericTiming(response.helperTotalMs);
  const ipcRoundTripMs = numericTiming(receivedAtMs - request.enqueuedAtMs);
  const queueOrTransportMs = helperTotalMs != null && ipcRoundTripMs != null
    ? numericTiming(Math.max(0, ipcRoundTripMs - helperTotalMs))
    : null;

  return {
    helper_total_ms: helperTotalMs,
    helper_monitor_ms: numericTiming(response.helperMonitorMs),
    helper_gdi_ms: numericTiming(response.helperGdiMs),
    helper_jpeg_encode_ms: numericTiming(response.helperJpegEncodeMs),
    helper_base64_ms: numericTiming(response.helperBase64Ms),
    ipc_round_trip_ms: ipcRoundTripMs,
    queue_or_transport_ms: queueOrTransportMs,
    pending_at_send: request.pendingAtSend,
    pending_at_receive: pending.size
  };
}

async function ensureHelper() {
  if (!isWindows()) {
    return null;
  }
  if (helper && helper.stdin.writable) {
    return helper;
  }
  if (startPromise) {
    return startPromise;
  }

  startPromise = (async () => {
    const executable = await compilePromise;
    const child = spawn(executable, [], {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true
    });
    helper = child;
    const lines = readline.createInterface({ input: child.stdout });
    lines.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return;
      }
      let message;
      try {
        message = JSON.parse(trimmed);
      } catch {
        return;
      }
      if (message.event === "ready") {
        return;
      }
      const request = pending.get(String(message.id || ""));
      if (!request) {
        return;
      }
      pending.delete(String(message.id));
      clearTimeout(request.timeout);
      if (message.ok) {
        request.resolve({
          ...message,
          timing: helperTimingFromResponse(message, request, performance.now())
        });
      } else {
        request.reject(new Error(message.error || "Windows screenshot helper capture failed."));
      }
    });
    child.stderr.on("data", () => {});
    child.once("error", (error) => closeHelper(error));
    child.once("exit", (code, signal) => {
      if (helper === child) {
        closeHelper(new Error(`Windows screenshot helper exited with code ${code ?? "null"} signal ${signal ?? "null"}.`));
      }
    });
    return child;
  })().finally(() => {
    startPromise = null;
  });

  return startPromise;
}

async function captureWindowsScreenshot(options = {}) {
  if (process.env.AGI_GTA_WINDOWS_CAPTURE_BACKEND === 'dxgi') {
    return require('./windowsDxgiScreenshotHelper').captureWindowsDxgiScreenshot(options);
  }
  if (!isWindows()) {
    throw new Error("Direct Windows screenshot capture is unavailable on this platform.");
  }
  if (!compilePromise) {
    compilePromise = compileHelper();
  }
  const child = await ensureHelper();
  if (!child || !child.stdin.writable) {
    throw new Error("Windows screenshot helper is not writable.");
  }

  const id = String(nextRequestId++);
  const request = {
    id,
    type: "capture",
    displayId: String(options.displayId || ""),
    width: Math.max(1, Math.trunc(Number(options.width) || 1280)),
    height: Math.max(1, Math.trunc(Number(options.height) || 720)),
    quality: Math.max(1, Math.min(100, Math.trunc(Number(options.quality) || 85)))
  };

  const responsePromise = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`Windows screenshot helper timed out after ${HELPER_TIMEOUT_MS}ms.`));
      closeHelper(new Error("Windows screenshot helper request timed out."));
    }, HELPER_TIMEOUT_MS);
    pending.set(id, {
      resolve,
      reject,
      timeout,
      enqueuedAtMs: performance.now(),
      pendingAtSend: pending.size + 1
    });
  });
  child.stdin.write(`${JSON.stringify({ id, ...request })}\n`);
  const response = await responsePromise;
  if (!response.data) {
    throw new Error("Windows screenshot helper returned no image data.");
  }
  return {
    buffer: Buffer.from(response.data, "base64"),
    mimeType: String(response.mimeType || "image/jpeg"),
    width: Number(response.width) || request.width,
    height: Number(response.height) || request.height,
    display: {
      id: String(options.displayId || ""),
      sourceId: String(options.displayId || "")
    },
    contentBounds: {
      left: Number(response.contentLeft) || 0,
      top: Number(response.contentTop) || 0,
      width: Number(response.contentWidth) || request.width,
      height: Number(response.contentHeight) || request.height
    },
    cursorIncluded: Boolean(response.cursorIncluded),
    cursor: Number(response.cursorX) >= 0 && Number(response.cursorY) >= 0
      ? { x: Number(response.cursorX), y: Number(response.cursorY) }
      : null,
    timing: response.timing || null
  };
}

async function closeWindowsScreenshotHelper() {
  require('./windowsDxgiScreenshotHelper').closeWindowsDxgiScreenshotHelper();
  if (helper) {
    closeHelper(new Error("Windows screenshot helper closed."));
  }
  if (compilePromise) {
    await compilePromise.catch(() => {});
  }
}

process.once("exit", () => {
  if (helper && !helper.killed) {
    helper.kill();
  }
});

module.exports = {
  captureWindowsScreenshot,
  closeWindowsScreenshotHelper
};

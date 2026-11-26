const screenshot = require("screenshot-desktop");
const sharp = require("sharp");
const path = require("node:path");
const os = require("node:os");
const fs = require("node:fs/promises");
const { execFile } = require("node:child_process");
const { resolutionSize } = require("./config");
const { saveCapture } = require("./captureStore");
const { captureWindowsScreenshot } = require("./windowsScreenshotHelper");
const { assertLinuxScreenshotBackendAvailable, commandExists, isWaylandScreenshotBackend, selectLinuxScreenshotBackend } = require("./linuxScreenshotBackend");
const { scriptPath } = require("./projectPaths");

const WAYLAND_FALLBACK_DISPLAY_ID = "wayland-default";
let macosCaptureCommandPromise = null;
let displayCache = null;

function linuxScreenshotBackend() {
  return selectLinuxScreenshotBackend();
}

function shouldUseWaylandScreenshot() {
  return process.platform === "linux" && isWaylandScreenshotBackend(linuxScreenshotBackend());
}

function firstFinite(...values) {
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number)) {
      return number;
    }
  }

  return 0;
}

function normalizeDisplay(display, index) {
  const left = firstFinite(display.left, display.x, display.offsetX, display.bounds?.left, display.bounds?.x, 0);
  const top = firstFinite(display.top, display.y, display.offsetY, display.bounds?.top, display.bounds?.y, 0);
  const width = firstFinite(display.width, display.bounds?.width, Number(display.right) - left, 0);
  const height = firstFinite(display.height, display.bounds?.height, Number(display.bottom) - top, 0);
  const id = String(display.id ?? display.displayId ?? display.name ?? index);
  const sourceId = Object.prototype.hasOwnProperty.call(display, "sourceId") ? display.sourceId : display.id;

  return {
    id,
    sourceId,
    index,
    name: display.name || `Display ${index + 1}`,
    left,
    top,
    width,
    height,
    right: width ? left + width : firstFinite(display.right, 0),
    bottom: height ? top + height : firstFinite(display.bottom, 0)
  };
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

function execFileBuffer(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    execFile(command, args, {
      encoding: "buffer",
      maxBuffer: options.maxBuffer || 64 * 1024 * 1024,
      timeout: options.timeout || 10000,
      windowsHide: true
    }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(stderr?.toString().trim() || error.message));
        return;
      }
      resolve(stdout);
    });
  });
}

async function captureSpectacle() {
  const filename = path.join(os.tmpdir(), `agi-gta-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.png`);
  const attempts = [
    ["--background", "--nonotify", "--fullscreen", "--pointer", "--output", filename],
    ["-b", "-n", "-f", "-p", "-o", filename]
  ];
  let lastError = null;

  try {
    for (const args of attempts) {
      try {
        await fs.unlink(filename).catch(() => {});
        await execFileText("spectacle", args, { timeout: 15000 });
        return await fs.readFile(filename);
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error("spectacle screenshot failed.");
  } finally {
    await fs.unlink(filename).catch(() => {});
  }
}

async function captureGnomeScreenshot() {
  const filename = path.join(os.tmpdir(), `agi-gta-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.png`);
  const attempts = [
    ["--include-pointer", `--file=${filename}`],
    ["--include-pointer", "--file", filename],
    ["-p", "-f", filename]
  ];
  let lastError = null;

  try {
    for (const args of attempts) {
      try {
        await fs.unlink(filename).catch(() => {});
        await execFileText("gnome-screenshot", args, { timeout: 15000 });
        return await fs.readFile(filename);
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error("gnome-screenshot failed.");
  } finally {
    await fs.unlink(filename).catch(() => {});
  }
}

async function capturePythonPillowDisplay(display) {
  const left = Math.trunc(firstFinite(display.left, 0));
  const top = Math.trunc(firstFinite(display.top, 0));
  const right = Math.trunc(firstFinite(display.right, left + display.width, 0));
  const bottom = Math.trunc(firstFinite(display.bottom, top + display.height, 0));
  const script = [
    "from PIL import ImageGrab",
    "import io, sys",
    "left, top, right, bottom = [int(float(value)) for value in sys.argv[1:5]]",
    "bbox = (left, top, right, bottom) if right > left and bottom > top else None",
    "image = ImageGrab.grab(bbox=bbox).convert('RGB')",
    "buffer = io.BytesIO()",
    "image.save(buffer, format='PNG')",
    "sys.stdout.buffer.write(buffer.getvalue())"
  ].join("\n");

  try {
    return await execFileBuffer("python3", ["-c", script, String(left), String(top), String(right), String(bottom)], {
      timeout: 15000,
      maxBuffer: 128 * 1024 * 1024
    });
  } catch (error) {
    throw new Error(`Python Pillow screenshot failed: ${error.message}`);
  }
}

async function macosDisplayBounds() {
  const stdout = await execFileText("swift", [scriptPath("macos-displays.swift")]);
  const parsed = JSON.parse(stdout);
  return Array.isArray(parsed) ? parsed : [];
}

async function macosDisplays() {
  return macosDisplayBounds();
}

async function macosCaptureCommand() {
  const sourcePath = scriptPath("macos-capture.swift");
  if (String(process.env.AGI_GTA_MACOS_CAPTURE_MODE || "").trim().toLowerCase() === "swift") {
    return {
      command: "swift",
      argsPrefix: [sourcePath],
      compiled: false
    };
  }

  if (!macosCaptureCommandPromise) {
    macosCaptureCommandPromise = (async () => {
      try {
        const stats = await fs.stat(sourcePath);
        const binaryPath = path.join(os.tmpdir(), `agi-gta-macos-capture-${process.arch}-${Math.trunc(stats.mtimeMs)}`);
        try {
          await fs.access(binaryPath);
        } catch (error) {
          await execFileText("swiftc", [sourcePath, "-o", binaryPath], { timeout: 30000 });
        }
        return {
          command: binaryPath,
          argsPrefix: [],
          compiled: true
        };
      } catch (error) {
        return {
          command: "swift",
          argsPrefix: [sourcePath],
          compiled: false
        };
      }
    })();
  }

  return macosCaptureCommandPromise;
}

async function captureMacosDisplay(sourceId) {
  if (sourceId === undefined || sourceId === null || String(sourceId) === "") {
    throw new Error("macOS capture requires a CGDirectDisplayID.");
  }

  const filename = path.join(os.tmpdir(), `agi-gta-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.png`);
  const selected = await macosCaptureCommand();
  const fallback = {
    command: "swift",
    argsPrefix: [scriptPath("macos-capture.swift")]
  };
  try {
    try {
      await execFileText(selected.command, [
        ...selected.argsPrefix,
        String(sourceId),
        filename
      ], { timeout: 15000 });
    } catch (error) {
      if (!selected.compiled) {
        throw error;
      }
      await execFileText(fallback.command, [
        ...fallback.argsPrefix,
        String(sourceId),
        filename
      ], { timeout: 15000 });
    }
    return await fs.readFile(filename);
  } finally {
    await fs.unlink(filename).catch(() => {});
  }
}

async function enrichMacosDisplays(displays) {
  if (process.platform !== "darwin") {
    return displays;
  }

  try {
    const bounds = await macosDisplayBounds();
    return displays.map((display, index) => ({
      ...display,
      ...(bounds[index] || {})
    }));
  } catch (error) {
    return displays;
  }
}

function modeSize(output) {
  if (output.current_mode && typeof output.current_mode === "object") {
    return output.current_mode;
  }

  if (Number.isInteger(output.current_mode) && Array.isArray(output.modes)) {
    return output.modes[output.current_mode] || {};
  }

  if (Array.isArray(output.modes)) {
    return output.modes.find((mode) => mode.current) || output.modes.find((mode) => mode.preferred) || output.modes[0] || {};
  }

  return {};
}

function parseWlrRandrDisplays(stdout) {
  let outputs;
  try {
    outputs = JSON.parse(stdout);
  } catch (error) {
    return [];
  }

  if (!Array.isArray(outputs)) {
    return [];
  }

  return outputs
    .filter((output) => output && output.enabled !== false)
    .map((output, index) => {
      const mode = modeSize(output);
      const position = output.position && typeof output.position === "object" ? output.position : {};
      const name = String(output.name || output.description || `Wayland Display ${index + 1}`);
      return {
        id: name,
        sourceId: name,
        name,
        left: firstFinite(position.x, output.x, 0),
        top: firstFinite(position.y, output.y, 0),
        width: firstFinite(output.logical_width, output.width, mode.width, 0),
        height: firstFinite(output.logical_height, output.height, mode.height, 0)
      };
    })
    .filter((display) => display.width > 0 && display.height > 0);
}

async function waylandDisplays() {
  if (commandExists("wlr-randr")) {
    try {
      const displays = parseWlrRandrDisplays(await execFileText("wlr-randr", ["--json"]));
      if (displays.length > 0) {
        return displays;
      }
    } catch (error) {
      // Fall through to a synthetic full-compositor display.
    }
  }

  return syntheticWaylandDisplays();
}

function syntheticWaylandDisplays() {
  return [
    {
      id: WAYLAND_FALLBACK_DISPLAY_ID,
      sourceId: null,
      name: "Wayland compositor",
      left: 0,
      top: 0,
      width: 0,
      height: 0
    }
  ];
}

function roundStat(value) {
  return Math.round(Number(value || 0) * 1000) / 1000;
}

async function captureVisualStats(raw) {
  const stats = await sharp(raw)
    .resize({ width: 64, height: 36, fit: "fill" })
    .removeAlpha()
    .stats();
  const channels = (stats.channels || []).slice(0, 3).map((channel) => ({
    min: roundStat(channel.min),
    max: roundStat(channel.max),
    mean: roundStat(channel.mean),
    stdev: roundStat(channel.stdev)
  }));
  const maxRange = channels.reduce((max, channel) => Math.max(max, channel.max - channel.min), 0);
  const maxStdev = channels.reduce((max, channel) => Math.max(max, channel.stdev), 0);

  return {
    sampleWidth: 64,
    sampleHeight: 36,
    maxRange: roundStat(maxRange),
    maxStdev: roundStat(maxStdev),
    isFlat: maxRange < 5 && maxStdev < 1,
    channels
  };
}

async function listDisplays(options = {}) {
  if (displayCache && !options.refresh) {
    return displayCache.map((display) => ({ ...display }));
  }

  let displays;
  if (process.platform === "darwin") {
    displays = (await macosDisplays()).map(normalizeDisplay);
  } else if (shouldUseWaylandScreenshot()) {
    const backend = linuxScreenshotBackend();
    assertLinuxScreenshotBackendAvailable(backend);
    if (backend === "grim") {
      displays = (await waylandDisplays()).map(normalizeDisplay);
    } else {
      displays = syntheticWaylandDisplays().map(normalizeDisplay);
    }
  } else {
    displays = (await enrichMacosDisplays(await screenshot.listDisplays())).map(normalizeDisplay);
  }

  displayCache = displays;
  return displays.map((display) => ({ ...display }));
}

function clearDisplayCache() {
  displayCache = null;
}

async function prepareScreenshotCapture() {
  if (process.platform !== 'win32' || process.env.AGI_GTA_WINDOWS_CAPTURE_BACKEND !== 'dxgi') return null;
  const warm = await require('./windowsDxgiScreenshotHelper').warmWindowsDxgiScreenshotHelper();
  await listDisplays();
  return warm;
}

function chooseDisplay(displays, displayId) {
  if (displayId !== undefined && displayId !== null && String(displayId) !== "") {
    const id = String(displayId);
    const found = displays.find((display) => display.id === id || String(display.index) === id);
    if (found) {
      return found;
    }
  }

  if (!displays.length) {
    throw new Error("No displays were reported by the screenshot backend.");
  }

  return displays[0];
}

function createCursorSvg() {
  return Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="30" height="42" viewBox="0 0 30 42">
      <path d="M3 2 L3 34 L11 27 L16 40 L22 37 L17 25 L28 25 Z" fill="white" stroke="black" stroke-width="2" stroke-linejoin="round"/>
    </svg>`
  );
}

function containTransform(source, target) {
  const sourceWidth = Math.max(1, Number(source.width) || target.width);
  const sourceHeight = Math.max(1, Number(source.height) || target.height);
  const scale = Math.min(target.width / sourceWidth, target.height / sourceHeight);
  const contentWidth = sourceWidth * scale;
  const contentHeight = sourceHeight * scale;
  const offsetX = (target.width - contentWidth) / 2;
  const offsetY = (target.height - contentHeight) / 2;

  return {
    scale,
    offsetX,
    offsetY,
    contentWidth,
    contentHeight
  };
}

function roundedContentBounds(transform) {
  return {
    left: Math.round(transform.offsetX),
    top: Math.round(transform.offsetY),
    width: Math.round(transform.contentWidth),
    height: Math.round(transform.contentHeight)
  };
}

function mapCursor(cursor, display, metadata, target) {
  if (!cursor || !Number.isFinite(cursor.x) || !Number.isFinite(cursor.y)) {
    return null;
  }

  const coordinateWidth = display.width || metadata.width || target.width;
  const coordinateHeight = display.height || metadata.height || target.height;
  const imageWidth = metadata.width || coordinateWidth;
  const imageHeight = metadata.height || coordinateHeight;
  const left = display.left || 0;
  const top = display.top || 0;
  const relX = cursor.x - left;
  const relY = cursor.y - top;

  if (coordinateWidth <= 0 || coordinateHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    return null;
  }

  if (relX < 0 || relY < 0 || relX > coordinateWidth || relY > coordinateHeight) {
    return null;
  }

  const imageX = (relX / coordinateWidth) * imageWidth;
  const imageY = (relY / coordinateHeight) * imageHeight;
  const transform = containTransform({ width: imageWidth, height: imageHeight }, target);

  return {
    x: Math.round(transform.offsetX + imageX * transform.scale),
    y: Math.round(transform.offsetY + imageY * transform.scale)
  };
}

async function captureObserve(options = {}) {
  const captureStarted = performance.now();
  const displays = await listDisplays();
  const display = chooseDisplay(displays, options.displayId);
  const target = resolutionSize(options.resolution || "1280x720");
  const outputFormat = String(options.captureFormat || "jpeg").trim().toLowerCase() === "png" ? "png" : "jpeg";
  const screenshotOptions = { format: "png" };
  let raw;
  let nativeCursorIncluded = false;
  let directCapture = null;
  let directCaptureError = null;

  const linuxBackend = process.platform === "linux" ? linuxScreenshotBackend() : "screenshot-desktop";
  const backendStarted = performance.now();
  if (process.platform === "win32" && options.directCapture && outputFormat === "jpeg") {
    try {
      directCapture = await captureWindowsScreenshot({
        displayId: display.sourceId,
        width: target.width,
        height: target.height,
        quality: options.captureQuality
      });
      raw = directCapture.buffer;
      nativeCursorIncluded = Boolean(directCapture.cursorIncluded);
    } catch (error) {
      if (error.nonFatalCaptureSkip) throw error;
      if (options.directCaptureFallback === false) {
        throw error;
      }
      directCaptureError = error.message || String(error);
      directCapture = null;
    }
  }

  if (!directCapture) {
    assertLinuxScreenshotBackendAvailable(linuxBackend);

    if (process.platform === "darwin") {
      raw = await captureMacosDisplay(display.sourceId);
    } else if (linuxBackend === "grim") {
      const args = ["-t", "png", "-c"];
      if (display.sourceId !== undefined && display.sourceId !== null) {
        args.push("-o", String(display.sourceId));
      }
      args.push("-");
      raw = await execFileBuffer("grim", args);
      nativeCursorIncluded = true;
    } else if (linuxBackend === "gnome-screenshot") {
      raw = await captureGnomeScreenshot();
      nativeCursorIncluded = true;
    } else if (linuxBackend === "spectacle") {
      raw = await captureSpectacle();
      nativeCursorIncluded = true;
    } else if (linuxBackend === "python-pillow") {
      raw = await capturePythonPillowDisplay(display);
    } else if (display.sourceId !== undefined && display.sourceId !== null) {
      screenshotOptions.screen = display.sourceId;
      raw = await screenshot(screenshotOptions);
    } else {
      raw = await screenshot(screenshotOptions);
    }
  }

  const screenshotBackendMs = performance.now() - backendStarted;
  const metadataStarted = performance.now();
  const metadata = directCapture
    ? { width: directCapture.width, height: directCapture.height, format: "jpeg" }
    : await sharp(raw).metadata();
  const metadataMs = performance.now() - metadataStarted;
  const visualStatsStarted = performance.now();
  const visualStats = options.includeVisualStats ? await captureVisualStats(raw).catch(() => null) : null;
  const visualStatsMs = performance.now() - visualStatsStarted;
  const contentBounds = directCapture?.contentBounds || roundedContentBounds(containTransform(metadata, target));
  const letterboxed = contentBounds.left > 0 || contentBounds.top > 0 || contentBounds.width !== target.width || contentBounds.height !== target.height;
  let cursor = null;

  const cursorStarted = performance.now();
  if (directCapture?.cursor) {
    cursor = directCapture.cursor;
  } else if (options.inputController) {
    try {
      cursor = await options.inputController.getCursorPosition();
    } catch (error) {
      cursor = null;
    }
  }
  const cursorQueryMs = performance.now() - cursorStarted;

  const mappedCursor = directCapture ? null : mapCursor(cursor, display, metadata, target);
  const pipeline = directCapture ? null : sharp(raw).resize({
      width: target.width,
      height: target.height,
      fit: "contain",
      background: { r: 0, g: 0, b: 0, alpha: 1 }
    });

  if (pipeline && mappedCursor && !nativeCursorIncluded) {
    const cursorWidth = 30;
    const cursorHeight = 42;
    pipeline.composite([
      {
        input: createCursorSvg(),
        left: Math.max(0, Math.min(target.width - cursorWidth, mappedCursor.x)),
        top: Math.max(0, Math.min(target.height - cursorHeight, mappedCursor.y))
      }
    ]);
  }

  const renderStarted = performance.now();
  const buffer = directCapture
    ? raw
    : outputFormat === "jpeg"
      ? await pipeline.jpeg({ quality: Math.max(1, Math.min(100, Math.trunc(Number(options.captureQuality) || 90))) }).toBuffer()
      : await pipeline.png().toBuffer();
  const renderAndPngMs = performance.now() - renderStarted;
  const outputMimeType = directCapture?.mimeType || (outputFormat === "jpeg" ? "image/jpeg" : "image/png");
  const saved = options.save === false ? null : await saveCapture(buffer, {
    sessionId: options.sessionId || "manual",
    label: options.label || "observe",
    mimeType: outputMimeType,
    captureRoot: options.captureRoot
  });

  if (directCapture?.backend === 'windows-dxgi') {
    const elapsed = performance.now() - captureStarted;
    if (elapsed > directCapture.timing.capture_deadline_ms) {
      throw Object.assign(new Error('Full DXGI observation exceeded capture deadline'), {
        nonFatalCaptureSkip: true, captureBackend: 'windows-dxgi', elapsedMs: elapsed
      });
    }
    directCapture.timing.frame_age_at_observation_ready_ms = performance.now() - directCapture.timing.frame_perf_ms;
  }

  return {
    buffer,
    imagePath: saved?.filePath || null,
    imageUrl: saved?.publicUrl || null,
    mimeType: saved?.mimeType || outputMimeType,
    width: target.width,
    height: target.height,
    display,
    cursor,
    cursorVisible: Boolean(mappedCursor || nativeCursorIncluded || directCapture?.cursor),
    contentBounds,
    letterboxed,
    visualStats,
    captureTiming: {
      total_ms: Number((performance.now() - captureStarted).toFixed(3)),
      screenshot_backend_ms: Number(screenshotBackendMs.toFixed(3)),
      screenshot_backend: directCapture?.backend || (directCapture ? "windows-helper" : linuxBackend),
      windows_helper_fallback_error: directCaptureError,
      metadata_ms: Number(metadataMs.toFixed(3)),
      visual_stats_ms: Number(visualStatsMs.toFixed(3)),
      cursor_query_ms: Number(cursorQueryMs.toFixed(3)),
      render_and_png_ms: Number(renderAndPngMs.toFixed(3)),
      windows_helper_total_ms: directCapture?.timing?.helper_total_ms ?? null,
      windows_helper_monitor_ms: directCapture?.timing?.helper_monitor_ms ?? null,
      windows_helper_gdi_ms: directCapture?.timing?.helper_gdi_ms ?? null,
      windows_helper_jpeg_encode_ms: directCapture?.timing?.helper_jpeg_encode_ms ?? null,
      windows_helper_base64_ms: directCapture?.timing?.helper_base64_ms ?? null,
      windows_helper_ipc_round_trip_ms: directCapture?.timing?.ipc_round_trip_ms ?? null,
      windows_helper_queue_or_transport_ms:
        directCapture?.timing?.queue_or_transport_ms ?? null,
      windows_helper_pending_at_send: directCapture?.timing?.pending_at_send ?? null,
      windows_helper_pending_at_receive:
        directCapture?.timing?.pending_at_receive ?? null,
      ...(directCapture?.backend === 'windows-dxgi' ? directCapture.timing : {})
    },
    capturedAt: new Date().toISOString()
  };
}

module.exports = {
  prepareScreenshotCapture,
  listDisplays,
  captureObserve,
  containTransform,
  normalizeDisplay,
  mapCursor,
  enrichMacosDisplays,
  macosDisplays,
  parseWlrRandrDisplays,
  linuxScreenshotBackend,
  shouldUseWaylandScreenshot,
  clearDisplayCache
};

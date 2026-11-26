const fs = require("node:fs/promises");
const path = require("node:path");
const { configuredPath } = require("./projectPaths");

const CAPTURE_ROOT = configuredPath(process.env.CAPTURE_DIR, "captures");
// A pending evidence file already has complete pixels in memory. Consumers
// must not wait for, or race, its asynchronous write.
const pendingImages = new Map();
function registerPendingImage(filePath, buffer) {
  const key = path.resolve(filePath);
  if (pendingImages.has(key)) throw Error('Duplicate pending capture path');
  pendingImages.set(key, buffer);
  return () => pendingImages.delete(key);
}

function safeSegment(value, fallback = "capture") {
  const cleaned = String(value || "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return cleaned || fallback;
}

function timestampSegment(date = new Date()) {
  return date.toISOString().replace(/[:.]/g, "-");
}

function createSessionId(date = new Date()) {
  const random = Math.random().toString(36).slice(2, 8);
  return `session-${timestampSegment(date)}-${random}`;
}

function captureDirectory(sessionId, captureRoot = CAPTURE_ROOT) {
  return path.join(captureRoot, safeSegment(sessionId, "manual"));
}

function publicUrlForCapture(filePath) {
  const relative = path.relative(CAPTURE_ROOT, filePath);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`Capture path is outside capture root: ${filePath}`);
  }

  return `/captures/${relative
    .split(path.sep)
    .map((segment) => encodeURIComponent(segment))
    .join("/")}`;
}

async function saveCapture(buffer, options = {}) {
  const captureRoot = path.resolve(options.captureRoot || CAPTURE_ROOT);
  const directory = captureDirectory(options.sessionId, captureRoot);
  await fs.mkdir(directory, { recursive: true });

  const label = safeSegment(options.label, "observe");
  const random = Math.random().toString(36).slice(2, 8);
  const mimeType = String(options.mimeType || "image/jpeg").toLowerCase() === "image/png" ? "image/png" : "image/jpeg";
  const extension = mimeType === "image/jpeg" ? "jpg" : "png";
  const filename = `${timestampSegment()}-${random}-${label}.${extension}`;
  const filePath = path.join(directory, filename);
  await fs.writeFile(filePath, buffer);

  return {
    filePath,
    publicUrl: publicUrlForCapture(filePath),
    mimeType
  };
}

async function imagePathToDataUrl(filePath, mimeType = "image/png") {
  const buffer = await imagePathToBuffer(filePath);
  return `data:${mimeType};base64,${buffer.toString("base64")}`;
}

async function imagePathToBuffer(filePath) {
  return pendingImages.get(path.resolve(filePath)) || fs.readFile(filePath);
}

module.exports = {
  imagePathToBuffer,
  CAPTURE_ROOT,
  createSessionId,
  saveCapture,
  imagePathToDataUrl,
  registerPendingImage,
  publicUrlForCapture
};

const path = require("node:path");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const OCR_ARTIFACT_ROOT = path.join(PROJECT_ROOT, "artifacts", "ocr");

function projectPath(...segments) {
  return path.join(PROJECT_ROOT, ...segments);
}

function scriptPath(fileName) {
  return projectPath("scripts", fileName);
}

function ocrArtifactPath(...segments) {
  return path.join(OCR_ARTIFACT_ROOT, ...segments);
}

function configuredPath(value, ...fallbackSegments) {
  if (value) {
    return path.resolve(path.isAbsolute(value) ? value : projectPath(value));
  }

  return projectPath(...fallbackSegments);
}

function normalizeRelativeSubpath(value = "") {
  const raw = String(value ?? "").trim();
  if (!raw) {
    return "";
  }
  if (path.isAbsolute(raw) || path.win32.isAbsolute(raw) || raw.includes("\0")) {
    throw new Error("Session output folder must be a relative path.");
  }

  const segments = raw.split(/[\\/]+/).filter(Boolean);
  if (segments.some((segment) => segment === "." || segment === ".." || /[<>:"|?*]/.test(segment))) {
    throw new Error("Session output folder contains an unsafe path segment.");
  }

  const normalized = segments.join(path.sep);
  return normalized;
}

function sessionOutputRoots(captureRoot, historyRoot, subpath = "") {
  const safeSubpath = normalizeRelativeSubpath(subpath);
  const resolvedCaptureRoot = path.resolve(captureRoot);
  const resolvedHistoryRoot = path.resolve(historyRoot);
  const captureSessionRoot = path.resolve(resolvedCaptureRoot, safeSubpath);
  const historySessionRoot = path.resolve(resolvedHistoryRoot, safeSubpath);
  const isInside = (root, candidate) => {
    const relative = path.relative(root, candidate);
    return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
  };
  if (!isInside(resolvedCaptureRoot, captureSessionRoot) || !isInside(resolvedHistoryRoot, historySessionRoot)) {
    throw new Error("Session output folder resolves outside the configured output roots.");
  }
  return {
    captureRoot: captureSessionRoot,
    sessionHistoryRoot: historySessionRoot,
    sessionOutputSubpath: safeSubpath.split(path.sep).join("/")
  };
}

module.exports = {
  PROJECT_ROOT,
  OCR_ARTIFACT_ROOT,
  configuredPath,
  ocrArtifactPath,
  projectPath,
  normalizeRelativeSubpath,
  sessionOutputRoots,
  scriptPath
};

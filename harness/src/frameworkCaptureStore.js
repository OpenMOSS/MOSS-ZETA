const path = require('node:path');
const { HighFrequencyCaptureStore } = require('./highFrequencyCaptureStore');

function normalizedPath(value) {
  const resolved = path.resolve(value);
  return process.platform === 'win32' ? resolved.toLowerCase() : resolved;
}

// Accept window/timeline/OCR objects or an explicit collection of paths.
function collectCapturePaths(value, result = new Set(), seen = new WeakSet()) {
  if (typeof value === 'string') {
    if (path.isAbsolute(value)) result.add(normalizedPath(value));
    return result;
  }
  if (!value || typeof value !== 'object' || Buffer.isBuffer(value) || seen.has(value)) return result;
  seen.add(value);
  if (Array.isArray(value) || value instanceof Set) {
    for (const item of value) collectCapturePaths(item, result, seen);
  } else {
    for (const [key, item] of Object.entries(value)) {
      if (typeof item === 'object' || /(?:imagePath|filePath|sourceImagePath|ocrImagePath|Path)$/.test(key)) {
        collectCapturePaths(item, result, seen);
      }
    }
  }
  return result;
}

class FrameworkCaptureStore {
  constructor(options = {}) {
    this.captureObserve = options.captureObserve;
    this.getPinnedPaths = options.getPinnedPaths || (() => []);
    this.pins = new Map();
    this.latestCapture = null;
    this.store = new HighFrequencyCaptureStore({ ...options,
      isPinned: (filePath) => this.isPinned(filePath),
      createPinnedFileCheck: () => this.createPinnedFileCheck() });
  }
  configure(options = {}) {
    if (options.getPinnedPaths) this.getPinnedPaths = options.getPinnedPaths;
    this.store.configure(options);
  }
  isPinned(filePath) {
    const key = normalizedPath(filePath);
    return this.pins.has(key) || normalizedPath(this.latestCapture?.imagePath || '.') === key
      || collectCapturePaths(this.getPinnedPaths()).has(key);
  }
  createPinnedFileCheck() {
    // The caller uses this snapshot only until its next asynchronous boundary.
    // Walking the complete Brain evidence separately for every file is quadratic.
    const retained = collectCapturePaths(this.getPinnedPaths());
    return (filePath) => {
      const key = normalizedPath(filePath);
      return this.pins.has(key) || normalizedPath(this.latestCapture?.imagePath || '.') === key
        || retained.has(key);
    };
  }
  pin(filePath) {
    if (!filePath) return () => {};
    const key = normalizedPath(filePath);
    this.pins.set(key, (this.pins.get(key) || 0) + 1);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      const count = (this.pins.get(key) || 0) - 1;
      if (count > 0) this.pins.set(key, count); else this.pins.delete(key);
    };
  }
  async startSession(sessionId) {
    this.latestCapture = null;
    await this.store.startSession(sessionId);
  }
  async capture(options = {}) {
    const observation = await this.captureObserve({ ...options, memoryOnly: true, save: false });
    // Test fixtures and externally persisted observations already have a real path.
    if (!Buffer.isBuffer(observation.buffer)) {
      if (!observation.imagePath) throw new Error('Framework capture requires an image buffer or a readable imagePath.');
      this.latestCapture = observation;
      return observation;
    }
    const persisted = this.store.enqueueTick(observation.buffer, {
      sessionId: options.sessionId || this.store.sessionId, capturedAt: observation.capturedAt,
      label: options.label, mimeType: observation.mimeType, writeScratch: true,
      archiveAsync: true, archiveInBackground: true
    });
    const result = { ...observation,
      imagePath: persisted.scratch.filePath,
      imageUrl: persisted.scratch.publicUrl,
      archiveImagePath: persisted.archive?.filePath || null,
      archiveImageUrl: persisted.archive?.publicUrl || null,
      capturePersistence: persisted,
      captureTiming: { ...observation.captureTiming, ...persisted.timing }
    };
    this.latestCapture = result;
    return result;
  }
  persistEvent(options = {}) { return this.store.persistEvent(options); }
  getStatus() { return { ...this.store.getStatus(), explicitPinnedFiles: this.pins.size }; }
  async finishSession() {
    this.latestCapture = null;
    // Outstanding users retain pins. Finishing must not remove their inputs.
    await this.store.finishSession();
  }
}
module.exports = { FrameworkCaptureStore, collectCapturePaths };

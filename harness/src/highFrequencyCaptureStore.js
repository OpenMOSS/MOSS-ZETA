const fs = require("node:fs/promises");
const path = require("node:path");
const sharp = require("sharp");
const { CAPTURE_ROOT, registerPendingImage, imagePathToBuffer } = require("./captureStore");

const GIB = 1024 * 1024 * 1024;
const SCRATCH_CLEANUP_INTERVAL_MS = 10000;
const QUOTA_CLEANUP_INTERVAL_MS = 60000;
const PIN_CHECK_BATCH_SIZE = 64;

function safeSegment(value, fallback = "capture") {
  const cleaned = String(value || "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return cleaned || fallback;
}

function timestampSegment(value = new Date()) {
  const date = value instanceof Date ? value : new Date(value);
  const safeDate = Number.isFinite(date.getTime()) ? date : new Date();
  return safeDate.toISOString().replace(/[:.]/g, "-");
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function boundedNumber(value, fallback, min, max) {
  return Math.max(min, Math.min(max, finiteNumber(value, fallback)));
}

function archiveFormat(value) {
  const normalized = String(value || "jpeg").trim().toLowerCase();
  if (normalized === "jpg") {
    return "jpeg";
  }
  return ["jpeg", "webp", "png"].includes(normalized) ? normalized : "jpeg";
}

function archiveExtension(format) {
  return format === "jpeg" ? "jpg" : format;
}

function archiveMimeType(format) {
  return format === "jpeg" ? "image/jpeg" : `image/${format}`;
}

function imageFormatForMimeType(mimeType, fallback = "png") {
  const normalized = String(mimeType || "").trim().toLowerCase();
  if (normalized === "image/jpeg" || normalized === "image/jpg") return "jpeg";
  if (normalized === "image/webp") return "webp";
  if (normalized === "image/png") return "png";
  return fallback;
}

function isInside(root, candidate) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

async function listFiles(root) {
  const files = [];
  const pending = [root];

  while (pending.length > 0) {
    const directory = pending.pop();
    let entries;
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch (error) {
      if (error.code === "ENOENT") {
        continue;
      }
      throw error;
    }

    for (const entry of entries) {
      const fullPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        pending.push(fullPath);
      } else if (entry.isFile()) {
        const stats = await fs.stat(fullPath);
        files.push({ filePath: fullPath, size: stats.size, mtimeMs: stats.mtimeMs });
      }
    }
  }

  return files;
}

async function removeEmptyDirectories(root) {
  let entries;
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") {
      return true;
    }
    throw error;
  }

  let empty = true;
  for (const entry of entries) {
    if (!entry.isDirectory()) {
      empty = false;
      continue;
    }
    const child = path.join(root, entry.name);
    const childEmpty = await removeEmptyDirectories(child);
    if (childEmpty) {
      await fs.rmdir(child).catch(() => {});
    } else {
      empty = false;
    }
  }
  return empty;
}

class HighFrequencyCaptureStore {
  constructor(options = {}) {
    this.captureRoot = path.resolve(options.captureRoot || CAPTURE_ROOT);
    this.publicRoot = path.resolve(options.publicRoot || this.captureRoot);
    this.archiveRoot = path.join(this.captureRoot, "high-frequency");
    this.scratchRoot = path.join(this.captureRoot, "_scratch", "high-frequency");
    this.now = typeof options.now === "function" ? options.now : () => Date.now();
    this.isPinned = typeof options.isPinned === "function" ? options.isPinned : () => false;
    this.createPinnedFileCheck = typeof options.createPinnedFileCheck === "function"
      ? options.createPinnedFileCheck : () => (filePath) => this.isPinned(filePath);
    this.onLog = typeof options.onLog === "function" ? options.onLog : () => {};
    this.onPersistence = typeof options.onPersistence === 'function' ? options.onPersistence : () => {};
    this.maxBytesOverride = Number.isFinite(Number(options.maxBytes)) ? Math.max(1, Number(options.maxBytes)) : null;
    this.maintenancePromise = null;
    this.lastScratchCleanupMs = 0;
    this.lastQuotaCleanupMs = 0;
    this.archiveQueue = Promise.resolve();
    this.scratchQueue = Promise.resolve();
    this.pendingScratchBytes = 0;
    this.pendingScratchCount = 0;
    this.scratchError = null;
    this.maxPendingScratchBytes = options.maxPendingScratchBytes ?? 64 * 1024 * 1024;
    this.sessionId = "";
    this.createdDirectories = new Set();
    this.lastArchiveAtMs = null;
    this.stats = {
      scratchFilesWritten: 0,
      archivedCaptures: 0,
      eventCaptures: 0,
      scratchFilesDeleted: 0,
      archiveFilesDeleted: 0,
      archiveBytesDeleted: 0,
      archiveBytes: 0,
      lastQuotaCheckedAt: null,
      lastArchivePath: null,
      lastEventArchivePath: null,
      lastMaintenanceError: ""
    };
    this.configure(options);
  }

  configure(options = {}) {
    if (options.captureRoot) {
      this.captureRoot = path.resolve(options.captureRoot);
      this.archiveRoot = path.join(this.captureRoot, "high-frequency");
      this.scratchRoot = path.join(this.captureRoot, "_scratch", "high-frequency");
    }
    this.archiveHz = boundedNumber(options.hfCaptureArchiveHz ?? options.archiveHz, 1, 0, 30);
    this.format = archiveFormat(options.hfCaptureArchiveFormat ?? options.format);
    this.quality = Math.trunc(boundedNumber(options.hfCaptureArchiveQuality ?? options.quality, 90, 1, 100));
    this.scratchSeconds = Math.trunc(boundedNumber(options.hfCaptureScratchSeconds ?? options.scratchSeconds, 60, 1, 3600));
    this.maxGb = boundedNumber(options.hfCaptureMaxGb ?? options.maxGb, 50, 0.1, 100000);
    this.saveOnEvent = options.hfCaptureSaveOnEvent ?? options.saveOnEvent ?? true;
  }

  getStatus() {
    return {
      archiveHz: this.archiveHz,
      archiveFormat: this.format,
      archiveQuality: this.quality,
      scratchSeconds: this.scratchSeconds,
      maxGb: this.maxGb,
      saveOnEvent: Boolean(this.saveOnEvent),
      archiveRoot: this.archiveRoot,
      scratchRoot: this.scratchRoot,
      sessionId: this.sessionId || null,
      pendingScratchBytes: this.pendingScratchBytes,
      pendingScratchCount: this.pendingScratchCount,
      scratchError: this.scratchError?.message || null,
      ...this.stats
    };
  }

  publicUrl(filePath) {
    if (!isInside(this.publicRoot, filePath)) {
      throw new Error(`Capture path is outside capture root: ${filePath}`);
    }
    const relative = path.relative(this.publicRoot, filePath);
    return `/captures/${relative.split(path.sep).map(encodeURIComponent).join("/")}`;
  }

  sessionArchiveDirectory(sessionId = this.sessionId) {
    return path.join(this.archiveRoot, safeSegment(sessionId, "session"));
  }

  sessionScratchDirectory(sessionId = this.sessionId) {
    return path.join(this.scratchRoot, safeSegment(sessionId, "session"));
  }

  async startSession(sessionId) {
    await this.scratchQueue;
    if (this.scratchError) throw this.scratchError;
    this.sessionId = safeSegment(sessionId, "session");
    this.lastArchiveAtMs = null;
    this.archiveQueue = Promise.resolve();
    this.createdDirectories.clear();
    Object.assign(this.stats, {
      scratchFilesWritten: 0,
      archivedCaptures: 0,
      eventCaptures: 0,
      scratchFilesDeleted: 0,
      archiveFilesDeleted: 0,
      archiveBytesDeleted: 0,
      lastArchivePath: null,
      lastEventArchivePath: null,
      lastMaintenanceError: ""
    });
    await this.ensureDirectory(this.sessionArchiveDirectory());
    await this.ensureDirectory(this.sessionScratchDirectory());
    await this.cleanupScratch(this.now() - this.scratchSeconds * 1000, {
      directory: this.scratchRoot
    });
    await this.enforceArchiveQuota();
    this.lastScratchCleanupMs = this.now();
    this.lastQuotaCleanupMs = this.now();
  }

  async ensureDirectory(directory) {
    if (this.createdDirectories.has(directory)) {
      return;
    }
    await fs.mkdir(directory, { recursive: true });
    this.createdDirectories.add(directory);
  }

  async encodeArchive(buffer) {
    if (this.format === "png") {
      return sharp(buffer).png({ compressionLevel: 9 }).toBuffer();
    }
    if (this.format === "webp") {
      return sharp(buffer).webp({ quality: this.quality }).toBuffer();
    }
    return sharp(buffer).jpeg({ quality: this.quality }).toBuffer();
  }

  shouldArchive(capturedAtMs) {
    if (this.archiveHz <= 0) {
      return false;
    }
    if (this.lastArchiveAtMs === null) {
      return true;
    }
    return capturedAtMs - this.lastArchiveAtMs >= 1000 / this.archiveHz;
  }

  reserveScratch(metadata = {}) {
    const directory = this.sessionScratchDirectory(metadata.sessionId);
    const label = safeSegment(metadata.label, "observe");
    const format = imageFormatForMimeType(metadata.mimeType, "png");
    const extension = format === "jpeg" ? "jpg" : format;
    const filename = `${timestampSegment(metadata.capturedAt)}-${Math.random().toString(36).slice(2, 8)}-${label}.${extension}`;
    const filePath = path.join(directory, filename);
    return { filePath, publicUrl: this.publicUrl(filePath), mimeType: archiveMimeType(format) };
  }

  async writeScratch(buffer, metadata = {}) {
    const started = performance.now();
    const target = metadata.scratchTarget || this.reserveScratch(metadata);
    const { filePath } = target;
    await this.ensureDirectory(path.dirname(filePath));
    await fs.writeFile(filePath, buffer);
    this.stats.scratchFilesWritten += 1;
    return {
      filePath,
      publicUrl: this.publicUrl(filePath),
      mimeType: target.mimeType,
      timing: {
        scratch_write_ms: Number((performance.now() - started).toFixed(3))
      }
    };
  }

  async writeArchive(buffer, metadata = {}) {
    const started = performance.now();
    const directory = this.sessionArchiveDirectory(metadata.sessionId);
    await this.ensureDirectory(directory);
    const label = safeSegment(metadata.label, metadata.event ? "event" : "observe");
    const extension = archiveExtension(this.format);
    const filename = `${timestampSegment(metadata.capturedAt)}-${Math.random().toString(36).slice(2, 8)}-${label}.${extension}`;
    const filePath = path.join(directory, filename);
    const encodeStarted = performance.now();
    const inputFormat = imageFormatForMimeType(metadata.mimeType, "png");
    const encoded = inputFormat === this.format ? buffer : await this.encodeArchive(buffer);
    const encodeMs = performance.now() - encodeStarted;
    const writeStarted = performance.now();
    await fs.writeFile(filePath, encoded);
    const writeMs = performance.now() - writeStarted;
    this.stats.archivedCaptures += 1;
    this.stats.archiveBytes += encoded.length;
    this.stats.lastArchivePath = filePath;
    if (metadata.event) {
      this.stats.eventCaptures += 1;
      this.stats.lastEventArchivePath = filePath;
    }
    this.scheduleMaintenance();
    return {
      filePath,
      publicUrl: this.publicUrl(filePath),
      mimeType: archiveMimeType(this.format),
      size: encoded.length,
      timing: {
        archive_total_ms: Number((performance.now() - started).toFixed(3)),
        archive_encode_ms: Number(encodeMs.toFixed(3)),
        archive_write_ms: Number(writeMs.toFixed(3))
      }
    };
  }

  async persistTick(buffer, metadata = {}) {
    if (!Buffer.isBuffer(buffer)) {
      throw new Error("High-frequency capture persistence requires an image buffer.");
    }
    const capturedAtMs = Date.parse(metadata.capturedAt || "");
    const timestampMs = Number.isFinite(capturedAtMs) ? capturedAtMs : this.now();
    const scratch = metadata.writeScratch === false
      ? null
      : await this.writeScratch(buffer, metadata);
    let archive = null;
    if (this.shouldArchive(timestampMs)) {
      this.lastArchiveAtMs = timestampMs;
      if (metadata.archiveAsync === true) {
        this.enqueueArchive(buffer, metadata);
      } else {
        archive = await this.writeArchive(buffer, metadata);
      }
    } else {
      this.scheduleMaintenance();
    }
    return {
      scratch,
      archive,
      timing: {
        scratch_write_ms: scratch?.timing?.scratch_write_ms ?? null,
        archive_total_ms: archive?.timing?.archive_total_ms ?? null,
        archive_encode_ms: archive?.timing?.archive_encode_ms ?? null,
        archive_write_ms: archive?.timing?.archive_write_ms ?? null
      }
    };
  }

  enqueueTick(buffer, metadata = {}) {
    if (this.scratchError) throw this.scratchError;
    if (!Buffer.isBuffer(buffer)) throw Error('Capture requires image bytes');
    if (this.pendingScratchBytes + buffer.length > this.maxPendingScratchBytes)
      throw Error('Capture evidence queue exceeded its memory budget');
    const scratch = this.reserveScratch(metadata);
    const release = registerPendingImage(scratch.filePath, buffer);
    const state = { scratch, archive: null, status: 'pending', timing: {} };
    const queuedAt = performance.now();
    this.pendingScratchBytes += buffer.length;
    this.pendingScratchCount++;
    this.scratchQueue = this.scratchQueue.then(async () => {
      const queueWaitMs = performance.now() - queuedAt;
      try {
        const result = await this.persistTick(buffer, { ...metadata, scratchTarget: scratch });
        Object.assign(state, result, { status: 'saved' });
      } catch (error) {
        this.scratchError ||= error;
        Object.assign(state, { status: 'failed', error: error.message });
        this.onLog(`capture evidence write failed: ${error.message}`);
      } finally {
        release();
        this.pendingScratchBytes -= buffer.length;
        this.pendingScratchCount--;
        try {
          this.onPersistence({ imagePath: scratch.filePath, capturedAt: metadata.capturedAt,
            status: state.status, error: state.error || null, ...state.timing, queue_wait_ms: queueWaitMs });
        } catch (error) {
          // Keep draining queued images even if the evidence callback fails.
          this.scratchError ||= error;
        }
      }
    });
    return state;
  }

  enqueueArchive(buffer, metadata = {}) {
    this.archiveQueue = this.archiveQueue
      .catch(() => {})
      .then(() => this.writeArchive(buffer, metadata))
      .catch((error) => {
        this.stats.lastMaintenanceError = error.message;
        this.onLog(`high-frequency archive write failed: ${error.message}`);
      });
  }

  async persistEvent(options = {}) {
    if (!this.saveOnEvent) {
      return null;
    }
    let buffer = options.buffer;
    if (!Buffer.isBuffer(buffer) && options.imagePath) {
      buffer = await imagePathToBuffer(options.imagePath);
    }
    if (!Buffer.isBuffer(buffer)) {
      return null;
    }
    return this.writeArchive(buffer, {
      sessionId: options.sessionId || this.sessionId,
      capturedAt: options.capturedAt,
      label: options.label || "event",
      event: true,
      mimeType: options.mimeType || "image/png"
    });
  }

  scheduleMaintenance() {
    if (this.maintenancePromise) {
      return;
    }
    const now = this.now();
    const cleanScratch = now - this.lastScratchCleanupMs >= SCRATCH_CLEANUP_INTERVAL_MS;
    const enforceQuota = now - this.lastQuotaCleanupMs >= QUOTA_CLEANUP_INTERVAL_MS;
    if (!cleanScratch && !enforceQuota) {
      return;
    }

    this.maintenancePromise = (async () => {
      if (cleanScratch) {
        await this.cleanupScratch(now - this.scratchSeconds * 1000, {
          directory: this.sessionScratchDirectory()
        });
        this.lastScratchCleanupMs = now;
      }
      if (enforceQuota) {
        await this.enforceArchiveQuota();
        this.lastQuotaCleanupMs = now;
      }
      this.stats.lastMaintenanceError = "";
    })().catch((error) => {
      this.stats.lastMaintenanceError = error.message;
      this.onLog(`high-frequency capture maintenance failed: ${error.message}`);
    }).finally(() => {
      this.maintenancePromise = null;
    });
  }

  async cleanupScratch(cutoffMs = this.now() - this.scratchSeconds * 1000, options = {}) {
    const directory = options.directory || this.sessionScratchDirectory();
    const files = await listFiles(directory);
    let deleted = 0;
    let checkPinned = null;
    let checked = 0;
    for (const file of files) {
      if (file.mtimeMs >= cutoffMs) continue;
      if (checked >= PIN_CHECK_BATCH_SIZE) {
        await new Promise(setImmediate);
        checkPinned = null;
        checked = 0;
      }
      checkPinned ||= this.createPinnedFileCheck();
      checked += 1;
      if (checkPinned(file.filePath)) continue;
      try {
        await fs.unlink(file.filePath);
        deleted += 1;
      } catch (error) {
        if (error.code === "ENOENT") {
          deleted += 1;
        }
      } finally {
        // Pins and retained context can change while an unlink is in flight.
        checkPinned = null;
        checked = 0;
      }
    }
    if (deleted > 0) {
      this.stats.scratchFilesDeleted += deleted;
      await removeEmptyDirectories(directory).catch(() => {});
    }
    return deleted;
  }

  async enforceArchiveQuota() {
    const files = await listFiles(this.archiveRoot);
    const maxBytes = this.maxBytesOverride ?? this.maxGb * GIB;
    let totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    if (totalBytes <= maxBytes) {
      this.stats.archiveBytes = totalBytes;
      this.stats.lastQuotaCheckedAt = new Date(this.now()).toISOString();
      return { totalBytes, deletedFiles: 0, deletedBytes: 0 };
    }

    let deletedFiles = 0;
    let deletedBytes = 0;
    let checkPinned = null;
    let checked = 0;
    for (const file of files.sort((left, right) => left.mtimeMs - right.mtimeMs)) {
      if (totalBytes <= maxBytes) {
        break;
      }
      if (checked >= PIN_CHECK_BATCH_SIZE) {
        await new Promise(setImmediate);
        checkPinned = null;
        checked = 0;
      }
      checkPinned ||= this.createPinnedFileCheck();
      checked += 1;
      if (checkPinned(file.filePath)) continue;
      try {
        await fs.unlink(file.filePath);
      } catch (error) {
        if (error.code !== "ENOENT") {
          continue;
        }
      } finally {
        checkPinned = null;
        checked = 0;
      }
      totalBytes -= file.size;
      deletedFiles += 1;
      deletedBytes += file.size;
    }
    this.stats.archiveFilesDeleted += deletedFiles;
    this.stats.archiveBytesDeleted += deletedBytes;
    this.stats.archiveBytes = totalBytes;
    this.stats.lastQuotaCheckedAt = new Date(this.now()).toISOString();
    if (deletedFiles > 0) {
      await removeEmptyDirectories(this.archiveRoot).catch(() => {});
      this.onLog(`high-frequency archive quota removed ${deletedFiles} old file(s), ${Math.round(deletedBytes / 1024 / 1024)} MiB`);
    }
    return { totalBytes, deletedFiles, deletedBytes };
  }

  async finishSession() {
    await this.scratchQueue;
    if (this.scratchError) throw this.scratchError;
    await this.archiveQueue;
    if (this.maintenancePromise) {
      await this.maintenancePromise;
    }
    const directory = this.sessionScratchDirectory();
    if (isInside(this.scratchRoot, directory)) {
      await this.cleanupScratch(Infinity, { directory });
      await fs.rmdir(directory).catch((error) => {
        if (!["ENOENT", "ENOTEMPTY", "EEXIST"].includes(error.code)) throw error;
      });
    }
    await removeEmptyDirectories(this.scratchRoot).catch(() => {});
  }
}

module.exports = {
  HighFrequencyCaptureStore,
  GIB,
  archiveFormat,
  archiveMimeType
};

const fs = require("node:fs/promises");
const { createReadStream } = require("node:fs");
const path = require("node:path");
const { isDeepStrictEqual } = require("node:util");
const { Worker } = require("node:worker_threads");
const { configuredPath } = require("./projectPaths");

const SESSION_HISTORY_ROOT = configuredPath(process.env.SESSION_HISTORY_DIR, "logs", "session-history");

function safeSegment(value, fallback = "session") {
  const cleaned = String(value || "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return cleaned || fallback;
}

function nowIso() {
  return new Date().toISOString();
}

function cloneJson(value) {
  if (value === undefined) {
    return null;
  }

  return JSON.parse(JSON.stringify(value));
}

function sessionHistoryDirectory(sessionId) {
  return path.join(SESSION_HISTORY_ROOT, safeSegment(sessionId));
}

function sessionHistoryPath(sessionId) {
  return path.join(sessionHistoryDirectory(sessionId), "history.json");
}

function sessionHistoryJournalPath(sessionId) {
  return path.join(sessionHistoryDirectory(sessionId), "history.ndjson");
}

function patchChanges(previous, next, keys, changes) {
  if (isDeepStrictEqual(previous, next)) return;
  if (Array.isArray(previous) && Array.isArray(next)
    && next.length >= previous.length
    && previous.every((value, index) => isDeepStrictEqual(value, next[index]))) {
    changes.push({ path: keys, append: next.slice(previous.length) });
  } else if (previous && next && typeof previous === "object" && typeof next === "object"
    && !Array.isArray(previous) && !Array.isArray(next)) {
    for (const key of Object.keys(previous)) {
      if (!Object.hasOwn(next, key)) changes.push({ path: [...keys, key], remove: true });
    }
    for (const key of Object.keys(next)) patchChanges(previous[key], next[key], [...keys, key], changes);
  } else {
    changes.push({ path: keys, value: next });
  }
}

function applyPatchChanges(history, changes = []) {
  for (const change of changes) {
    const keys = change.path;
    if (!Array.isArray(keys) || !keys.length
      || keys.some((key) => typeof key !== "string" || ["__proto__", "constructor", "prototype"].includes(key))) {
      throw new Error("Invalid history delta path.");
    }
    let target = history;
    for (const key of keys.slice(0, -1)) {
      if (!Object.hasOwn(target, key) || !target[key] || typeof target[key] !== "object") target[key] = {};
      target = target[key];
    }
    const key = keys.at(-1);
    if (change.remove) delete target[key];
    else if (Array.isArray(change.append)) {
      if (!Array.isArray(target[key])) target[key] = [];
      target[key].push(...change.append);
    }
    else target[key] = change.value;
  }
}

function applyJournalOperation(history, operation) {
  if (!operation || typeof operation !== "object") {
    return;
  }

  if (operation.kind === "update") {
    Object.assign(history, operation.patch || {});
    applyPatchChanges(history, operation.patchChanges);
    return;
  }

  if (operation.kind === "append") {
    if (operation.collection) {
      if (!Array.isArray(history[operation.collection])) {
        history[operation.collection] = [];
      }
      history[operation.collection].push(operation.record);
    }
    if (operation.event) {
      if (!Array.isArray(history.events)) {
        history.events = [];
      }
      history.events.push(operation.event);
    }
    Object.assign(history, operation.patch || {});
    applyPatchChanges(history, operation.patchChanges);
  }
}

async function* readJournalOperations(filePath) {
  let pending = Buffer.alloc(0);
  try {
    // Bound the read buffer by one operation, not the total journal size.
    for await (const block of createReadStream(filePath, { highWaterMark: 65536 })) {
      pending = pending.length ? Buffer.concat([pending, block]) : block;
      let start = 0;
      let end;
      while ((end = pending.indexOf(10, start)) >= 0) {
        const line = pending.subarray(start, end).toString("utf8").replace(/\r$/, "");
        start = end + 1;
        if (line) yield JSON.parse(line); // A corrupt committed line must fail.
      }
      pending = pending.subarray(start);
    }
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }
  if (pending.length) {
    // Preserve the previous contract: a partial final write is tolerated, while
    // a fully parseable final record without a newline is still replayed.
    const line = pending.toString("utf8").replace(/\r$/, "");
    try { if (line) yield JSON.parse(line); }
    catch (error) { if (!(error instanceof SyntaxError)) throw error; }
  }
}

class SessionHistoryRecorder {
  constructor(initial = {}, options = {}) {
    if (!initial.sessionId) {
      throw new Error("Session history requires a sessionId.");
    }

    this.sessionHistoryRoot = path.resolve(initial.sessionHistoryRoot || SESSION_HISTORY_ROOT);
    this.filePath = path.join(this.sessionHistoryRoot, safeSegment(initial.sessionId), "history.json");
    this.journalPath = path.join(this.sessionHistoryRoot, safeSegment(initial.sessionId), "history.ndjson");
    this.memoryRecordLimit=options.memoryRecordLimit||0;
    this.appendOnly = Boolean(options.appendOnly);
    this.asyncWrites = this.appendOnly && options.asyncWrites === true;
    this.retainHistoryInMemory = options.retainHistoryInMemory !== false;
    if (!this.retainHistoryInMemory && !this.asyncWrites) {
      throw new Error("Disabling the history cache requires the asynchronous append-only writer.");
    }
    this.writerOptions = options;
    this.writer = null;
    this.writerError = null;
    this.writerRequests = new Map();
    this.writerRequestSequence = 0;
    this.writerClosed = false;
    this.writerFlush = null;
    this.deltaPatches = this.appendOnly && options.deltaPatches === true;
    this.writeQueue = Promise.resolve();
    this.journalHandle = null;
    this.writeDebounceMs = Math.max(0, Math.trunc(Number(options.writeDebounceMs) || 0));
    this.writeTimer = null;
    this.coalescedWriteActive = false;
    this.pendingWriteWaiters = [];
    this.flushWaiters = [];
    this.flushRequested = false;
    this.lastWriteTiming = null;
    this.nextJournalSequence = Number(initial.journalSequence || 0);
    this.history = {
      schemaVersion: 1,
      journalSequence: this.nextJournalSequence,
      sessionId: initial.sessionId,
      historyPath: this.filePath,
      historyRoot: this.sessionHistoryRoot,
      captureRoot: initial.captureRoot || null,
      sessionOutputSubpath: initial.sessionOutputSubpath || initial.config?.sessionOutputSubpath || "",
      agentMode: initial.agentMode || initial.config?.agentMode || "",
      status: initial.status || "running",
      startedAt: initial.startedAt || nowIso(),
      stoppedAt: null,
      updatedAt: null,
      config: cloneJson(initial.config || {}),
      systemPrompt: initial.systemPrompt || "",
      taskInstruction: initial.taskInstruction || "",
      modelName: initial.modelName || "",
      actionCount: Number(initial.actionCount || 0),
      responseCount: Number(initial.responseCount || 0),
      planCount: Number(initial.planCount || 0),
      lastPlan: cloneJson(initial.lastPlan || null),
      maxSteps: initial.maxSteps ?? initial.config?.maxSteps ?? initial.maxActions ?? initial.config?.maxActions ?? null,
      maxActions: initial.maxActions ?? initial.config?.maxActions ?? initial.maxSteps ?? initial.config?.maxSteps ?? null,
      messages: cloneJson(initial.messages || []),
      observations: cloneJson(initial.observations || []),
      modelResponses: cloneJson(initial.modelResponses || []),
      plans: cloneJson(initial.plans || []),
      actions: cloneJson(initial.actions || []),
      policyActionCandidates: cloneJson(initial.policyActionCandidates || []),
      executedActions: cloneJson(initial.executedActions || []),
      brainRevisions: cloneJson(initial.brainRevisions || []),
      taskResults: cloneJson(initial.taskResults || []),
      ocrResults: cloneJson(initial.ocrResults || []),
      errors: cloneJson(initial.errors || []),
      events: cloneJson(initial.events || []),
      sharedDocuments: cloneJson(initial.sharedDocuments || null),
      framework: cloneJson(initial.framework || null),
      memory: cloneJson(initial.memory || null),
      sessionHandoff: cloneJson(initial.sessionHandoff || null)
    };
    this.journalPatchState = this.deltaPatches ? { ...this.history } : null;
  }

  async startWriter() {
    const worker = new Worker(path.join(__dirname, "sessionHistoryWorker.js"), {
      workerData: { history: this.history, filePath: this.filePath, journalPath: this.journalPath,
        ioDelayMs: Number(this.writerOptions.ioDelayMs || 0),
        failAfterSequence: this.writerOptions.failAfterSequence ?? null }
    });
    this.writer = worker;
    await new Promise((resolve, reject) => {
      const fail = (error) => {
        this.writerError ||= error;
        reject(error);
        for (const request of this.writerRequests.values()) request.reject(error);
        this.writerRequests.clear();
        this.writerOptions.onError?.(error);
      };
      worker.on("error", fail);
      worker.on("exit", (code) => {
        if (!this.writerClosed) fail(new Error(`History writer exited before flush (code ${code}).`));
      });
      worker.on("message", (message) => {
        if (message.type === "ready") { worker.unref(); resolve(); return; }
        if (message.type === "failure") {
          const error = new Error(`History writer failed: ${message.error}`);
          error.stack = message.stack || error.stack;
          fail(error);
          return;
        }
        if (message.timing) this.lastWriteTiming = message.timing;
        if (message.type === "complete") {
          const request = this.writerRequests.get(message.id);
          this.writerRequests.delete(message.id);
          if (!this.writerRequests.size) worker.unref();
          request?.resolve();
        }
      });
    });
  }

  writerCommand(command) {
    if (this.writerError) return Promise.reject(this.writerError);
    if (this.writerClosed) return Promise.reject(new Error("History writer is already closed."));
    const id = ++this.writerRequestSequence;
    return new Promise((resolve, reject) => {
      this.writerRequests.set(id, { resolve, reject });
      this.writer.ref();
      try { this.writer.postMessage({ command, id }); }
      catch (error) { this.writerRequests.delete(id); reject(error); }
    });
  }

  enqueueOperation(operation) {
    if (this.writerError) throw this.writerError;
    if (this.writerClosed || this.writerFlush) throw new Error("Cannot record after history flush began.");
    // Clone only this event and incremental state. No complete timeline is
    // copied, compared or serialized on the control thread. postMessage takes
    // an immutable structured-clone snapshot before returning to its caller.
    const cloned = structuredClone(operation);
    const explicitChanges = cloned.patch?.__historyPatchChanges || [];
    if (cloned.patch) delete cloned.patch.__historyPatchChanges;
    cloned.patchChanges = explicitChanges;
    cloned.sequence = ++this.nextJournalSequence;
    // The worker and journal retain every operation. Native high-frequency
    // control already owns its full Brain evidence in SessionState; keeping a
    // second cumulative copy here makes an unlimited run grow without bound.
    // Opted-out callers use readAll() for evidence, not this initial cache.
    if (this.retainHistoryInMemory) applyJournalOperation(this.history, cloned);
    this.history.journalSequence = cloned.sequence;
    this.history.updatedAt = cloned.event?.at || nowIso();
    cloned.patch = { ...cloned.patch, updatedAt: this.history.updatedAt };
    this.writer.postMessage({ operation: cloned });
    return Promise.resolve();
  }

  async update(patch = {}) {
    if (this.asyncWrites) return this.enqueueOperation({ kind: "update", patch });
    const clonedPatch = cloneJson(patch);
    Object.assign(this.history, clonedPatch);
    return this.persistOperation({
      kind: "update",
      patch: clonedPatch
    });
  }

  async appendRecord(collection, record, eventType, eventPayload = {}, patch = {}) {
    if (this.asyncWrites) return this.enqueueOperation({ kind: "append", collection, record,
      event: { type: eventType || collection, at: nowIso(), ...eventPayload }, patch });
    if (!Array.isArray(this.history[collection])) {
      throw new Error(`Session history collection is not appendable: ${collection}`);
    }

    const clonedRecord = cloneJson(record);
    const clonedPatch = cloneJson(patch);
    const event = {
      type: eventType || collection,
      at: nowIso(),
      ...cloneJson(eventPayload)
    };
    this.history[collection].push(clonedRecord);
    this.history.events.push(event);
    if (clonedPatch && Object.keys(clonedPatch).length > 0) {
      Object.assign(this.history, clonedPatch);
    }
    this.history.updatedAt = event.at;
    return this.persistOperation({
      kind: "append",
      collection,
      record: clonedRecord,
      event,
      patch: clonedPatch
    });
  }

  async appendEvent(type, payload = {}, patch = {}) {
    if (this.asyncWrites) return this.enqueueOperation({ kind: "append",
      event: { type, at: nowIso(), ...payload }, patch });
    const event = {
      type,
      at: nowIso(),
      ...cloneJson(payload)
    };
    const clonedPatch = cloneJson(patch);
    this.history.events.push(event);

    if (clonedPatch && Object.keys(clonedPatch).length > 0) {
      Object.assign(this.history, clonedPatch);
    }
    this.history.updatedAt = event.at;

    return this.persistOperation({
      kind: "append",
      event,
      patch: clonedPatch
    });
  }

  persistOperation(operation) {
    if (operation.patch?.__historyPatchChanges) {
      operation.patchChanges = operation.patch.__historyPatchChanges;
      delete operation.patch.__historyPatchChanges;
      delete this.history.__historyPatchChanges;
      applyPatchChanges(this.history, operation.patchChanges);
    }
    if (this.appendOnly) {
      if (this.deltaPatches) {
        operation.patchChanges ||= [];
        const patch = { ...operation.patch, updatedAt: this.history.updatedAt };
        for (const [key, value] of Object.entries(patch)) {
          patchChanges(this.journalPatchState[key], value, [key], operation.patchChanges);
          this.journalPatchState[key] = value;
        }
        delete operation.patch;
      }
      this.nextJournalSequence += 1;
      operation.sequence = this.nextJournalSequence;
      this.history.journalSequence = this.nextJournalSequence;
      // A failed journal write poisons the queue; never silently resume after a gap.
      this.writeQueue = this.writeQueue
        .then(() => this.appendJournalOperation(operation));
      if(this.memoryRecordLimit){
        for(const value of Object.values(this.history))if(Array.isArray(value)&&value.length>this.memoryRecordLimit)value.splice(0,value.length-this.memoryRecordLimit);
      }
      return this.writeQueue.then(()=>undefined);
    }
    return this.write();
  }

  async write() {
    if (this.writeDebounceMs > 0) {
      return this.writeCoalesced();
    }

    this.writeQueue = this.writeQueue.catch(() => {}).then(() => this.writeNow());
    return this.writeQueue;
  }

  writeCoalesced() {
    const promise = new Promise((resolve, reject) => {
      this.pendingWriteWaiters.push({ resolve, reject });
    });
    this.scheduleCoalescedWrite(this.flushRequested ? 0 : this.writeDebounceMs);
    return promise;
  }

  scheduleCoalescedWrite(delayMs) {
    if (this.coalescedWriteActive || this.writeTimer) {
      return;
    }

    this.writeTimer = setTimeout(() => {
      this.writeTimer = null;
      this.startCoalescedWrite();
    }, Math.max(0, delayMs));
    this.writeTimer.unref?.();
  }

  startCoalescedWrite() {
    if (this.coalescedWriteActive || this.pendingWriteWaiters.length === 0) {
      this.resolveFlushWaitersIfIdle();
      return;
    }

    this.coalescedWriteActive = true;
    const waiters = this.pendingWriteWaiters.splice(0);
    this.writeQueue = this.writeQueue.catch(() => {}).then(() => this.writeNow());
    void this.writeQueue.then(
      () => {
        for (const waiter of waiters) {
          waiter.resolve(this.history);
        }
      },
      (error) => {
        for (const waiter of waiters) {
          waiter.reject(error);
        }
      }
    ).finally(() => {
      this.coalescedWriteActive = false;
      if (this.pendingWriteWaiters.length > 0) {
        this.scheduleCoalescedWrite(this.flushRequested ? 0 : this.writeDebounceMs);
      } else {
        this.resolveFlushWaitersIfIdle();
      }
    });
  }

  resolveFlushWaitersIfIdle() {
    if (this.writeTimer || this.coalescedWriteActive || this.pendingWriteWaiters.length > 0) {
      return;
    }

    this.flushRequested = false;
    const waiters = this.flushWaiters.splice(0);
    for (const resolve of waiters) {
      resolve();
    }
  }

  async writeNow() {
    if (this.asyncWrites && this.writer) return this.writerCommand("snapshot");
    const writeStarted = performance.now();
    this.history.updatedAt = nowIso();
    const serializeStarted = performance.now();
    const serialized = `${JSON.stringify(this.history, null, 2)}\n`;
    const serializeMs = performance.now() - serializeStarted;
    const filesystemStarted = performance.now();
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    const temporaryPath = `${this.filePath}.${process.pid}.tmp`;
    await fs.writeFile(temporaryPath, serialized, "utf8");
    await fs.rename(temporaryPath, this.filePath);
    const filesystemMs = performance.now() - filesystemStarted;
    this.lastWriteTiming = {
      total_ms: Number((performance.now() - writeStarted).toFixed(3)),
      serialize_ms: Number(serializeMs.toFixed(3)),
      filesystem_ms: Number(filesystemMs.toFixed(3)),
      bytes: Buffer.byteLength(serialized)
    };
    return this.history;
  }

  async appendJournalOperation(operation) {
    const writeStarted = performance.now();
    const serializeStarted = performance.now();
    const serialized = `${JSON.stringify(operation)}\n`;
    const serializeMs = performance.now() - serializeStarted;
    const filesystemStarted = performance.now();
    // Keep one ordered append handle for the session. Opening and closing the
    // file for every observation/candidate/event adds several Windows I/O waits
    // to each 200ms control period. Completion still waits for the actual write.
    if (!this.journalHandle) {
      await fs.mkdir(path.dirname(this.journalPath), { recursive: true });
      this.journalHandle = await fs.open(this.journalPath, "a");
    }
    await this.journalHandle.writeFile(serialized, "utf8");
    const filesystemMs = performance.now() - filesystemStarted;
    this.lastWriteTiming = {
      mode: "append",
      total_ms: Number((performance.now() - writeStarted).toFixed(3)),
      serialize_ms: Number(serializeMs.toFixed(3)),
      filesystem_ms: Number(filesystemMs.toFixed(3)),
      bytes: Buffer.byteLength(serialized)
    };
    return this.history;
  }

  async readAll() {
    if (this.asyncWrites && !this.writerClosed) await this.writerCommand("sync");
    else await this.writeQueue;
    if (this.writerError) throw this.writerError;
    return readSessionHistoryFile(this.filePath);
  }

  getWriteTiming() {
    return this.lastWriteTiming ? { ...this.lastWriteTiming } : null;
  }

  async flush() {
    if (this.asyncWrites) {
      if (!this.writerFlush) this.writerFlush = (async () => {
        try { await this.writerCommand("flush"); }
        finally { this.writerClosed = true; await this.writer.terminate(); }
      })();
      return this.writerFlush;
    }
    if (this.appendOnly) {
      const boundary = this.writeQueue;
      this.writeQueue = boundary.finally(async () => {
        const handle = this.journalHandle;
        this.journalHandle = null;
        if (handle) await handle.close();
      });
      await this.writeQueue;
      if(this.memoryRecordLimit)return; // Keep initial snapshot + complete journal, never compact a truncated cache.
      await this.writeNow();
      await fs.rm(this.journalPath, { force: true });
      return;
    }

    if (this.writeDebounceMs > 0) {
      this.flushRequested = true;
      if (this.writeTimer) {
        clearTimeout(this.writeTimer);
        this.writeTimer = null;
      }
      if (!this.coalescedWriteActive && this.pendingWriteWaiters.length > 0) {
        this.startCoalescedWrite();
      }

      if (!this.coalescedWriteActive && this.pendingWriteWaiters.length === 0) {
        this.resolveFlushWaitersIfIdle();
        return;
      }

      await new Promise((resolve) => {
        this.flushWaiters.push(resolve);
      });
      return;
    }

    return this.writeQueue;
  }
}

async function createSessionHistory(initial = {}, options = {}) {
  const recorder = new SessionHistoryRecorder(initial, options);
  if (recorder.asyncWrites) {
    try { await recorder.startWriter(); return recorder; }
    catch (error) { recorder.writerClosed = true; await recorder.writer?.terminate(); throw error; }
  }
  await recorder.writeNow();
  if (recorder.appendOnly) {
    await fs.rm(recorder.journalPath, { force: true });
    recorder.journalHandle = await fs.open(recorder.journalPath, "a");
  }
  return recorder;
}

async function locateSessionHistory(sessionId) {
  const directPath = sessionHistoryPath(sessionId);
  try {
    await fs.access(directPath);return directPath;
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }

  const targetName = safeSegment(sessionId);
  const pending = [SESSION_HISTORY_ROOT];
  while (pending.length > 0) {
    const directory = pending.pop();
    let entries;
    try {
      entries = await fs.opendir(directory);
    } catch (error) {
      if (error.code === "ENOENT") continue;
      throw error;
    }
    for await (const entry of entries) {
      if (!entry.isDirectory() || entry.name === "history-page-index") continue;
      const child = path.join(directory, entry.name);
      if (entry.name === targetName) {
        const file=path.join(child,"history.json");await fs.access(file);return file;
      }
      pending.push(child);
    }
  }
  const error = new Error(`Session history not found: ${sessionId}`);
  error.code = "ENOENT";
  throw error;
}

async function readSessionHistory(sessionId) {
  return readSessionHistoryFile(await locateSessionHistory(sessionId));
}

async function readSessionHistoryFile(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  const history = JSON.parse(raw);
  const journalPath = path.join(path.dirname(filePath), "history.ndjson");
  const compactedThrough = Number(history.journalSequence || 0);
  for await (const operation of readJournalOperations(journalPath)) {
    if (Number(operation.sequence || 0) <= compactedThrough) {
      continue;
    }
    applyJournalOperation(history, operation);
    history.journalSequence = Math.max(Number(history.journalSequence || 0), Number(operation.sequence || 0));
  }
  if(history.framework?.timeline?.records){
    try{const archived=(await fs.readFile(path.join(path.dirname(filePath),'timeline-archive.ndjson'),'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);history.framework.timeline.records=archived.concat(history.framework.timeline.records);}catch(e){if(e.code!=='ENOENT')throw e;}
  }
  return history;
}

module.exports = {
  SESSION_HISTORY_ROOT,
  SessionHistoryRecorder,
  createSessionHistory,
  readSessionHistory,
  locateSessionHistory,
  readSessionHistoryFile,
  sessionHistoryDirectory,
  sessionHistoryJournalPath,
  sessionHistoryPath,
  applyJournalOperation
};

// Evidence persistence runs away from the input/deadline event loop. The parent
// sends immutable, ordered event deltas; only this worker serializes and writes.
const fs = require("node:fs/promises");
const path = require("node:path");
const { parentPort, workerData } = require("node:worker_threads");
const { applyJournalOperation } = require("./sessionHistoryStore");

let history = workerData.history;
workerData.history = null;
let snapshotSequence = Number(history.journalSequence || 0);
let persistedSequence = snapshotSequence;
let journalBytes = 0;
// This bounds optional compaction work, not records or evidence retention.
const COMPACTION_JOURNAL_BYTES = 64 * 1024 * 1024;
const pending = [];
let handle;
let scheduled = false;
let active = false;
let failed = false;
let timing = null;

async function snapshot() {
  if (!history) return false;
  const temporaryPath = `${workerData.filePath}.${process.pid}.worker.tmp`;
  await fs.writeFile(temporaryPath, `${JSON.stringify(history, null, 2)}\n`, "utf8");
  await fs.rename(temporaryPath, workerData.filePath);
  snapshotSequence = persistedSequence;
  return true;
}

function schedule() {
  if (scheduled || active || failed) return;
  scheduled = true;
  setImmediate(() => { scheduled = false; void drain(); });
}

async function drain() {
  if (active || failed) return;
  active = true;
  try {
    while (pending.length) {
      if (pending[0].operation) {
        const started = performance.now();
        const lines = [];
        let characters = 0;
        // Batch I/O while retaining every ordered operation and its own sequence.
        // Never coalesce away event-time state or OCR corrections.
        while (pending.length && pending[0].operation && lines.length < 256 && characters < 1024 * 1024) {
          const operation = pending.shift().operation;
          if (workerData.failAfterSequence !== null && operation.sequence >= workerData.failAfterSequence) {
            throw new Error(`Injected history I/O failure at sequence ${operation.sequence}`);
          }
          if (history) {
            applyJournalOperation(history, operation);
            history.journalSequence = operation.sequence;
          }
          const line = JSON.stringify(operation);
          lines.push(line);
          characters += line.length + 1;
          persistedSequence = operation.sequence;
        }
        const serialized = `${lines.join("\n")}\n`;
        const serializeMs = performance.now() - started;
        const filesystemStarted = performance.now();
        if (workerData.ioDelayMs) await new Promise(resolve => setTimeout(resolve, workerData.ioDelayMs));
        await handle.writeFile(serialized, "utf8");
        const batchBytes = Buffer.byteLength(serialized);
        journalBytes += batchBytes;
        // Large sessions keep their initial/last compacted snapshot plus every
        // ordered journal operation. Drop only the worker's duplicate cache.
        if (journalBytes > COMPACTION_JOURNAL_BYTES) history = null;
        timing = { mode: "worker_append", total_ms: performance.now() - started,
          serialize_ms: serializeMs, filesystem_ms: performance.now() - filesystemStarted,
          bytes: batchBytes, operations: lines.length,
          persistedSequence };
        parentPort.postMessage({ type: "persisted", timing });
      } else {
        const { command, id } = pending.shift();
        await handle.sync();
        let compacted = false;
        if (command === "snapshot" || command === "flush") {
          try { compacted = await snapshot(); }
          catch (error) {
            // The previous snapshot and synced journal are still intact.
            if (!(error instanceof RangeError) || !/Invalid string length/.test(error.message)) throw error;
            history = null;
          }
        }
        if (command === "flush") {
          const persistedBytes = (await handle.stat()).size;
          await handle.close(); handle = null;
          if (compacted) await fs.rm(workerData.journalPath, { force: true });
          else {
            const receipt = path.join(path.dirname(workerData.filePath), "history-flush.json");
            const temporary = `${receipt}.${process.pid}.tmp`;
            await fs.writeFile(temporary, JSON.stringify({ schemaVersion: 1,
              format: "snapshot-and-append-journal-v1", writerClosed: true,
              snapshotSequence, journalSequence: persistedSequence, journalBytes: persistedBytes,
              flushedAt: new Date().toISOString() }) + "\n", "utf8");
            await fs.rename(temporary, receipt);
          }
        }
        parentPort.postMessage({ type: "complete", id, timing });
      }
    }
  } catch (error) {
    failed = true;
    await handle?.close().catch(() => {});
    handle = null;
    parentPort.postMessage({ type: "failure", error: error.message, stack: error.stack });
  } finally {
    active = false;
    if (pending.length && !failed) schedule();
  }
}

(async () => {
  await fs.mkdir(path.dirname(workerData.filePath), { recursive: true });
  await snapshot();
  await fs.rm(workerData.journalPath, { force: true });
  handle = await fs.open(workerData.journalPath, "a");
  parentPort.on("message", message => { pending.push(message); schedule(); });
  parentPort.postMessage({ type: "ready" });
})().catch(error => {
  parentPort.postMessage({ type: "failure", error: error.message, stack: error.stack });
});

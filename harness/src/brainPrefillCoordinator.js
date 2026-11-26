const { randomUUID } = require('node:crypto');
const { freezeSnapshot } = require('./brainPrefillMessages');

class BrainPrefillCoordinator {
  constructor({ enabled = false, intervalMs = 1000, prepare, run, isCurrent, recordEvent = async () => {}, now = () => Date.now() }) {
    Object.assign(this, { enabled, intervalMs, prepare, run, isCurrent, recordEvent, now });
    this.generation = 0;
    this.lastOfferedAt = -Infinity;
    this.inFlight = null;
    this.completed = null;
    this.formalActive = false;
    this.stopped = false;
    this.stats = { offered: 0, completed: 0, rejected: 0, failed: 0, cancelled: 0 };
  }

  canOffer() {
    return this.enabled && !this.stopped && !this.formalActive && !this.inFlight
      && this.now() - this.lastOfferedAt >= this.intervalMs;
  }

  offer(snapshotOrFactory) {
    if (!this.canOffer()) { this.stats.rejected++; return false; }
    const job = { jobId: `brain-prefill-${randomUUID()}`, generation: this.generation,
      controller: new AbortController(), offeredAt: this.now(), snapshot: null };
    this.inFlight = job;
    this.lastOfferedAt = job.offeredAt;
    this.stats.offered++;
    // The lazy factory freezes a single coherent runtime view off the capture
    // callback. There is never a queue of observation snapshots or image jobs.
    job.promise = new Promise(resolve => setImmediate(resolve)).then(async () => {
      job.controller.signal.throwIfAborted();
      const preparationStarted = performance.now();
      const snapshot = typeof snapshotOrFactory === 'function' ? snapshotOrFactory() : snapshotOrFactory;
      if (snapshot == null) {
        job.controller.abort(Error('Brain prefill became ineligible before snapshot'));
        job.controller.signal.throwIfAborted();
      }
      job.snapshot = freezeSnapshot(snapshot);
      const frozenAt = performance.now();
      if (!this.isCurrent(job.snapshot)) throw Error('Brain prefill snapshot belongs to an obsolete state');
      const prefix = await this.prepare(job.snapshot, job.controller.signal);
      job.controller.signal.throwIfAborted();
      await this.recordEvent('brain_prefill_started', { jobId: job.jobId, sessionId: job.snapshot.sessionId,
        brainRevision: job.snapshot.brainRevision, cutoffActionSeq: job.snapshot.cutoffActionSeq,
        prefixHash: prefix.prefixHash, systemSha256: prefix.systemSha256,
        snapshotPreparationMs: job.snapshot.snapshotPreparationMs, snapshotFactoryAndFreezeMs: frozenAt - preparationStarted,
        prefixPreparationMs: performance.now() - frozenAt,
        inputEvidence: prefix.inputEvidence, images: prefix.images, messages: prefix.referenceMessages });
      const response = await this.run(prefix, { signal: job.controller.signal, jobId: job.jobId });
      job.controller.signal.throwIfAborted();
      if (this.stopped || job.generation !== this.generation || !this.isCurrent(job.snapshot)) return;
      const finishedAt = this.now();
      this.completed = { ...prefix, jobId: job.jobId, prefillResponse: response,
        completedAt: finishedAt, wallLatencyMs: finishedAt - job.offeredAt };
      this.stats.completed++;
      await this.recordEvent('brain_prefill_completed', { jobId: job.jobId, prefixHash: prefix.prefixHash,
        sessionId: job.snapshot.sessionId, brainRevision: job.snapshot.brainRevision,
        cutoffActionSeq: job.snapshot.cutoffActionSeq, wallLatencyMs: finishedAt - job.offeredAt,
        ...response.audit, actualPromptTokenIds: response.promptTokenIds || null });
    }).catch(async error => {
      const cancelled = job.controller.signal.aborted || job.generation !== this.generation;
      this.stats[cancelled ? 'cancelled' : 'failed']++;
      this.lastError = cancelled ? null : error.message;
      try { await this.recordEvent(cancelled ? 'brain_prefill_cancelled' : 'brain_prefill_error', {
        jobId: job.jobId, error: error.message, cancellationReason: job.controller.signal.reason?.message || null }); }
      catch (historyError) { this.lastError = historyError.message; }
    }).finally(() => { if (this.inFlight === job) this.inFlight = null; });
    return true;
  }

  beginInference() {
    this.formalActive = true;
    this.inFlight?.controller.abort(Error('Formal Brain request takes priority over unfinished prefill'));
    return this.completed;
  }

  endInference() { this.formalActive = false; }

  cancelInFlight(reason = 'Brain request superseded') {
    this.inFlight?.controller.abort(Error(reason));
  }

  invalidate(reason = 'Brain state changed') {
    this.generation++;
    this.completed = null;
    this.lastOfferedAt = -Infinity;
    this.cancelInFlight(reason);
  }

  stop(reason = 'session stopped') { this.stopped = true; this.invalidate(reason); }

  getPinnedPaths() {
    return [...new Set([this.inFlight?.snapshot, this.completed?.snapshot].filter(Boolean)
      .flatMap(snapshot => snapshot.trajectory.flatMap(record => [record.imagePath, record.navigation?.minimap?.imagePath]).filter(Boolean)))];
  }

  getStatus() {
    return { enabled: this.enabled, intervalMs: this.intervalMs, inFlight: this.inFlight?.jobId || null,
      formalActive: this.formalActive, lastError: this.lastError || null, ...this.stats,
      ready: this.completed ? { jobId: this.completed.jobId, prefixHash: this.completed.prefixHash,
        completedAt: this.completed.completedAt, cutoffActionSeq: this.completed.snapshot.cutoffActionSeq,
        brainRevision: this.completed.snapshot.brainRevision, audit: this.completed.prefillResponse.audit } : null };
  }
}

module.exports = { BrainPrefillCoordinator };

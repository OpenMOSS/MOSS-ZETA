// Portable integration adapter for the production clock. A streamed state must
// never release the full-response dependency between two Policy requests.
class GenerationAdmission {
  constructor({ generate, validate, commit = async () => {}, onError = () => {}, onSkip = () => {}, now = () => performance.now() }) {
    if (typeof generate !== 'function' || typeof validate !== 'function') throw Error('Generation and full-response validation callbacks are required');
    Object.assign(this, { generate, validate, commit, onError, onSkip, now });
    this.pending = null; this.active = null; this.stopped = false;
  }
  offer({ observation, gate }) {
    if (this.stopped) { gate.cancel('Admission stopped'); return; }
    if (this.pending) {
      this.pending.gate.cancel('Superseded by a fresh observation');
      this.onSkip({ slot: this.pending.gate.slot, reason: 'superseded_observation' });
    }
    this.pending = { observation, gate };
    this.pump();
  }
  pump() {
    if (this.stopped || this.active || !this.pending) return;
    const item = this.pending; this.pending = null;
    if (item.gate.closed || this.now() > item.gate.expiresAt) {
      item.gate.cancel('Expired observation'); this.onSkip({ slot: item.gate.slot, reason: 'expired_slot' }); return;
    }
    const controller = new AbortController();
    const active = { controller, gate: item.gate }; this.active = active;
    active.promise = (async () => {
      try {
        const admission = await item.gate.wait(controller.signal);
        controller.signal.throwIfAborted();
        const result = await this.generate({ observation: item.observation, gate: item.gate,
          admission, scheduledStartPerfMs: item.gate.executeAt, signal: controller.signal });
        controller.signal.throwIfAborted();
        if (await this.validate(result) !== true) throw Error('Previous Policy response is not complete, valid and committed');
        await this.commit(result, item);
      } catch (error) {
        if (error.nonFatalSlotSkip) this.onSkip({ slot: item.gate.slot, reason: error.message });
        else if (!this.stopped) {
          this.stop(error);
          this.onError(error);
        }
      } finally {
        if (this.active === active) this.active = null;
        this.pump();
      }
    })();
  }
  stop(reason = Error('Admission stopped')) {
    this.stopped = true;
    this.pending?.gate.cancel('Admission stopped'); this.pending = null;
    this.active?.controller.abort(reason);
  }
  async waitForIdle() {
    while (this.active) await this.active.promise;
  }
}
module.exports = { GenerationAdmission };

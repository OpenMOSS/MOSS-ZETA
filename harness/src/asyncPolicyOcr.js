const ROI_FIELDS = { tutorial_prompt: 'P', top_prompt: 'P', mission_objective: 'B', subtitle: 'B', subtitle_tight: 'B', central_result: 'C', key_prompt: 'K' };
const block = (text = '') => `<screen_ocr>${text ? '\n' + text + '\n' : ''}</screen_ocr>`;
const emptyScreen = () => ({ P: '', B: '', C: '', K: '' });
const normalize = text => String(text || '').split(/\r?\n/).map(line => line.normalize('NFKC')
  .replace(/\s+/g, ' ').trim().replace(/\.{2,}/g, '…').replace(/([!?。])\1+/g, '$1')
  .replace(/(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fffA-Za-z0-9])/g, '')
  .replace(/(?<=[A-Za-z0-9]) +(?=[\u3400-\u9fff])/g, '')
  .replace(/ +(?=[,;:!?。…])/g, '').replace(/(?<=[,;:!?。…]) +(?=[\u3400-\u9fff])/g, '')
  .replace(/\\/g, '\\\\').replace(/\|/g, '\\|')).filter(Boolean).join('|');
const identity = o => o?.frameworkObservationId || o?.observationId;

// OCR never gates inference. Delivery is fixed to the preceding actual request,
// not the preceding capture. A missed deadline permanently drops that result.
class AsyncPolicyOcr {
  constructor() { this.inFlight = false; this.pending = null; this.reset(); }
  reset() {
    this.epoch = (this.epoch || 0) + 1;
    this.sequence = 0; this.latest = null; this.results = [];
    this.previousRequest = null; this.closed = new Set(); this.reservations = new Map();
    this.pending?.resolve(); this.pending = null; this.beginSegment();
  }
  beginSegment({ preserveScreen = false } = {}) {
    this.segment = (this.segment || 0) + 1;
    this.screen = preserveScreen && this.screen ? this.screen : emptyScreen(); this.hasSnapshot = false; this.reservations?.clear();
  }
  recoverCommittedState(screen) {
    // Backend history owns the exact S/D replay baseline. Discard OCR work from
    // cancelled requests and pause-menu captures; never synthesize an S reset.
    this.reset();
    if (screen !== null) {
      this.screen = { ...screen };
      this.hasSnapshot = true;
    }
  }
  schedule(observation, recognize, onResult = () => {}) {
    return new Promise(resolve => {
      this.pending?.resolve();
      this.pending = { observation, recognize, onResult, resolve, epoch: this.epoch, sequence: ++this.sequence };
      void this.pump();
    });
  }
  async pump() {
    if (this.inFlight || !this.pending) return;
    const item = this.pending; this.pending = null; this.inFlight = true;
    this.activeObservation = item.observation;
    try {
      const result = await item.recognize(item.observation);
      if (item.epoch !== this.epoch || !result?.ok || result.skipped) return;
      this.latest = { sequence: item.sequence, result, observationId: identity(item.observation),
        capturedAt: item.observation.capturedAt, imagePath: item.observation.imagePath };
      if (!this.closed.has(this.latest.observationId)) this.results.push(this.latest);
      this.results = this.results.slice(-8);
      // Brain evidence is independent of the Policy OCR deadline.
      await item.onResult(result);
    } catch (_) {
      // Recognition failure is not a field-clear event.
    } finally {
      this.inFlight = false; this.activeObservation = null; item.resolve(); void this.pump();
    }
  }
  checkpointDelivery() {
    return { previousRequest: this.previousRequest, closed: new Set(this.closed),
      results: this.results.slice(), reservations: new Map(this.reservations), segment: this.segment };
  }
  restoreDelivery(saved) {
    if (saved.segment !== this.segment) return;
    this.previousRequest = saved.previousRequest; this.closed = saved.closed;
    this.reservations = saved.reservations;
    const merged = [...saved.results, ...this.results, ...(this.latest ? [this.latest] : [])];
    this.results = [...new Map(merged.filter(r => !this.closed.has(r.observationId)).map(r => [r.observationId, r])).values()].slice(-8);
  }
  reserve(observation) {
    const current = identity(observation);
    const token = `${this.segment}:${current}`;
    if (this.reservations.has(token)) return this.reservations.get(token);
    const item = this.results.find(r => r.observationId === this.previousRequest);
    if (this.previousRequest != null) this.closed.add(this.previousRequest);
    this.previousRequest = current;
    this.results = this.results.filter(r => r.observationId === current);
    this.closed = new Set([...this.closed].slice(-32));
    const fields = { ...item?.result.policyFields };
    if (!item?.result.policyFields) for (const region of item?.result.results || []) {
      if (ROI_FIELDS[region.roi]) fields[ROI_FIELDS[region.roi]] = normalize(Array.isArray(region.lines) && region.lines.length ? region.lines.map(line => line.text).join('\n') : region.text);
    }
    const snapshot = { ...this.screen, ...fields };
    const changes = Object.entries(fields).filter(([k,v]) => this.screen[k] !== v);
    const values = !this.hasSnapshot ? Object.entries(snapshot).filter(([,v]) => v) : changes;
    const lines = values.sort(([a],[b]) => 'PBCK'.indexOf(a)-'PBCK'.indexOf(b)).map(([k,v]) => `${k}=${v}`);
    const text = !this.hasSnapshot ? ['S', ...lines].join('\n') : lines.length ? ['D', ...lines].join('\n') : '';
    const records = text ? [{ id: token, observationId: item?.observationId ?? null,
      capturedAt: item?.capturedAt ?? null, text }] : [];
    const reservation = { token, records, text: block(text), snapshot };
    this.reservations.set(token, reservation);
    return reservation;
  }
  acknowledge(tokens) {
    for (const token of tokens) {
      const reservation = this.reservations.get(token);
      if (!reservation) continue;
      this.screen = reservation.snapshot; this.hasSnapshot = true;
      this.reservations.delete(token);
    }
  }
  consume(observation) { return this.reserve(observation).text; }
}
module.exports = { AsyncPolicyOcr, ocrBlock: block };

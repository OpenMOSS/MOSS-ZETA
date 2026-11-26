const { spawn } = require('node:child_process');
const path = require('node:path');
const readline = require('node:readline');
const { performance } = require('node:perf_hooks');

function missedSlot(reason) {
  return Object.assign(new Error(reason), { nonFatalSlotSkip: true });
}

class SubmissionGate {
  constructor(slot, submitAt, executeAt, allowLateTimer = false, maxLatenessMs = 0, onDecision = null) {
    if (!Number.isFinite(maxLatenessMs) || maxLatenessMs < 0 || maxLatenessMs > 100) throw Error('Submission grace must be in [0,100]ms');
    Object.assign(this, { slot, submitAt, executeAt, allowLateTimer, maxLatenessMs, onDecision,
      expiresAt: submitAt + maxLatenessMs, armed: false, closed: false, dispatched: false, due: false });
  }
  wait(signal) {
    if (this.closed || this.armed || performance.now() > this.expiresAt || signal?.aborted) {
      return Promise.reject(missedSlot('HTTP payload missed its submission slot'));
    }
    this.armed = true;
    this.readyAt = performance.now();
    return new Promise((resolve, reject) => {
      this.resolve = resolve; this.reject = reject;
      this.abort = () => this.cancel('Submission cancelled');
      this.signal = signal;
      signal?.addEventListener('abort', this.abort, { once: true });
      if (this.due || performance.now() >= this.submitAt) this.release();
    });
  }
  release() {
    if (this.closed) return;
    this.due = true;
    const now = performance.now();
    if (!this.armed && this.maxLatenessMs && now < this.expiresAt) {
      if (!this.expiryTimer) this.expiryTimer = setTimeout(() => {
        this.expiryTimer = null;
        this.release();
      }, Math.max(1, this.expiresAt - now));
      return;
    }
    this.closed = true;
    clearTimeout(this.expiryTimer);
    this.signal?.removeEventListener('abort', this.abort);
    // A blocked event loop must not turn a stale tick into a late request.
    if (!this.armed || (this.maxLatenessMs ? now > this.expiresAt : this.allowLateTimer ? now >= this.executeAt + 200 : now - this.submitAt > 10)) {
      this.reason = this.armed ? (this.allowLateTimer ? 'Original action window expired' : 'Submission timer exceeded 10ms scheduling tolerance') : 'No prepared request at submission slot';
      this.reject?.(missedSlot(this.reason));
      this.onDecision?.(this);
      return;
    }
    this.dispatched = true;
    this.resolve({ submission_slot: this.slot, scheduled_submit_perf_ms: this.submitAt,
      payload_ready_perf_ms: this.readyAt, submission_release_perf_ms: now,
      submission_lateness_ms: now - this.submitAt, submission_wait_ms: now - this.readyAt,
      scheduled_action_perf_ms: this.executeAt, submission_max_lateness_ms: this.maxLatenessMs });
    this.onDecision?.(this);
  }
  cancel(reason) {
    if (this.closed) return;
    this.closed = true; this.reason = reason;
    clearTimeout(this.expiryTimer);
    this.signal?.removeEventListener('abort', this.abort);
    this.reject?.(missedSlot(reason));
    this.onDecision?.(this);
  }
}

class FixedSubmissionClock {
  constructor({ offsetMs, actionDelayMs = 200, allowLateTimer = false, maxLatenessMs = 0, clockOffsetMs, onCapture, onSubmit, onError, python }) {
    Object.assign(this, { offsetMs, actionDelayMs, allowLateTimer, maxLatenessMs, clockOffsetMs, onCapture, onSubmit, onError, python });
    this.gates = new Map();
  }
  async start() {
    this.stopped = false;
    const proc = spawn(this.python || process.env.AGI_GTA_GAMEPAD_PYTHON || 'python',
      ['-u', path.join(__dirname, '..', 'scripts', 'observation-clock.py')],
      { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    this.child = proc;
    await new Promise((resolve, reject) => {
      let ready = false, stderr = '';
      const timer = setTimeout(() => { this.stop(); reject(Error('Observation clock startup timeout')); }, 10000);
      proc.stderr.on('data', data => { stderr = (stderr + data).slice(-2000); });
      readline.createInterface({ input: proc.stdout }).on('line', line => {
        let event;
        try { event = JSON.parse(line); } catch { return; }
        if (event.event === 'ready') { ready = true; clearTimeout(timer); resolve(); return; }
        if (this.stopped) return;
        const scheduled = event.scheduledMs - this.clockOffsetMs;
        if (event.event === 'capture') {
          for (const [slot, gate] of this.gates) if (slot < event.slot && performance.now() >= gate.expiresAt) {
            gate.cancel('Capture slot expired'); this.gates.delete(slot);
          }
          const gate = new SubmissionGate(event.slot, scheduled + this.offsetMs, scheduled + this.offsetMs + this.actionDelayMs,
            this.allowLateTimer, this.maxLatenessMs, gate => { if (!this.stopped) this.onSubmit?.(gate); });
          this.gates.set(event.slot, gate);
          this.onCapture({ slot: event.slot, scheduledPerfMs: scheduled, gate });
        } else if (event.event === 'submit') {
          const gate = this.gates.get(event.slot);
          if (gate) gate.release();
        }
      });
      proc.once('error', error => { clearTimeout(timer); reject(error); if (ready && !this.stopped) this.onError(error); });
      proc.once('exit', () => {
        clearTimeout(timer);
        const error = Error(`Observation clock exited: ${stderr}`);
        if (!ready) reject(error); else if (!this.stopped) this.onError(error);
      });
    });
    this.epochPerfMs = performance.now() + 100;
    proc.stdin.write(JSON.stringify({ epochMs: this.epochPerfMs + this.clockOffsetMs, periodMs: 200, offsetMs: this.offsetMs }) + '\n');
  }
  stop() {
    this.stopped = true;
    for (const gate of this.gates.values()) gate.cancel('Observation clock stopped');
    this.gates.clear();
    this.child?.kill();
  }
}
module.exports = { FixedSubmissionClock, SubmissionGate, missedSlot };

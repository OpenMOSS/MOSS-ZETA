const test = require('node:test');
const assert = require('node:assert/strict');
const { SubmissionGate, FixedSubmissionClock } = require('../src/fixedSubmissionClock');

test('prepared payload waits for the scheduled submission and preserves action start', async () => {
  const now = performance.now();
  const gate = new SubmissionGate(0, now + 25, now + 225, true, 100);
  let sent = false;
  const pending = gate.wait().then(receipt => { sent = true; return receipt; });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(sent, false);
  await new Promise(resolve => setTimeout(resolve, Math.max(0, gate.submitAt - performance.now()) + 2));
  gate.release();
  const receipt = await pending;
  assert(receipt.submission_release_perf_ms >= gate.submitAt);
  assert.equal(receipt.scheduled_action_perf_ms, now + 225);
  assert.equal(gate.dispatched, true);
});

test('expired and cancelled observations never catch up in another slot', async () => {
  const now = performance.now(), decisions = [];
  const late = new SubmissionGate(1, now - 120, now + 80, true, 100, g => decisions.push(g.dispatched));
  late.release();
  await assert.rejects(late.wait(), error => error.nonFatalSlotSkip);
  late.release(); late.cancel('stopped');
  assert.deepEqual(decisions, [false]);
  const cancelled = new SubmissionGate(2, now + 1000, now + 1200, true, 100);
  const pending = cancelled.wait();
  cancelled.cancel('manual pause');
  await assert.rejects(pending, error => error.nonFatalSlotSkip);
  assert.equal(cancelled.dispatched, false);
});

test('the independent process emits absolute 200ms capture and submission slots', async () => {
  const captures = [], submissions = [], errors = [];
  let done;
  const finished = new Promise(resolve => { done = resolve; });
  const clock = new FixedSubmissionClock({ offsetMs: 80, actionDelayMs: 200, maxLatenessMs: 100,
    // Only relative planned timestamps are checked here; no native-device calibration is claimed.
    clockOffsetMs: 0,
    onCapture: event => { captures.push(event); },
    onSubmit: gate => { submissions.push(gate); if (submissions.length === 3) done(); },
    onError: error => { errors.push(error); done(); } });
  // Node performance.now and Python perf_counter use different origins. Calibrate
  // using an isolated one-shot read; production uses the native helper RTT samples.
  const { execFileSync } = require('node:child_process');
  const before = performance.now();
  const helper = Number(execFileSync(process.env.AGI_GTA_GAMEPAD_PYTHON || 'python', ['-c', 'import time;print(time.perf_counter()*1000)'], { encoding: 'utf8' }));
  clock.clockOffsetMs = helper - (before + performance.now()) / 2;
  let timeout;
  try {
    await clock.start();
    await Promise.race([finished, new Promise((_, reject) => { timeout = setTimeout(() => reject(Error('clock timeout')), 5000); })]);
    assert.deepEqual(errors, []);
    assert(captures.length >= 3);
    for (let i = 1; i < captures.length; i++) {
      const slotDelta = captures[i].slot - captures[i - 1].slot;
      assert(slotDelta >= 1);
      assert(Math.abs(captures[i].scheduledPerfMs - captures[i - 1].scheduledPerfMs - slotDelta * 200) < 0.001);
    }
    assert(submissions.every(g => !g.dispatched));
    assert(captures.every(e => Math.abs(e.gate.executeAt - e.gate.submitAt - 200) < 0.001));
  } finally { clearTimeout(timeout); clock.stop(); }
});

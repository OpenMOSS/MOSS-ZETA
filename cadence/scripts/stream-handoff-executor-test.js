const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { StreamCadenceExecutor } = require('../src/streamCadenceExecutor');

function state(rt = 1) { return { leftStick: { x: 0, y: 0 }, rightStick: { x: 0, y: 0 }, leftTrigger: 0, rightTrigger: rt, buttons: [] }; }
async function fixture() {
  const input = new EventEmitter(), sent = [], resets = [];
  Object.assign(input, { streamCadenceSupported: true, streamHandoffSupported: true,
    calibrateGamepadClock: async () => ({ offsetMs: 0 }),
    send: async (command, payload) => { sent.push({ command, payload }); return { accepted: true, ...payload }; } });
  const executor = new StreamCadenceExecutor(input, { actionRuntime: { reset: async reason => resets.push(reason) } });
  executor.start(); await executor.arm();
  const prepare = (changes = {}) => {
    const now = performance.now();
    return { expectedOwner: 'A', nextOwner: 'B', updateVersion: 1, ownershipVersion: 1,
      observationPerfMs: now - 150, maxObservationAgeMs: 400, firstExecutablePerfMs: now - 10,
      pkg: { requestId: 'B-request', states: Array.from({ length: 6 }, () => state()), scheduledStartPerfMs: now + 30 },
      commit: () => {}, ...changes };
  };
  return { executor, input, sent, resets, prepare };
}

test('handoff sends one atomic native command, changes routing in synchronous commit and never resets', async () => {
  const f = await fixture(); let committed = 0;
  const beforeEpoch = f.executor.epoch;
  const result = await f.executor.handoff(f.prepare({ commit: () => {
    committed++; assert.equal(f.executor.ownerId, 'B'); assert.notEqual(f.executor.epoch, beforeEpoch);
  } }));
  assert(result.accepted); assert.equal(committed, 1);
  assert.deepEqual(f.resets, ['stream_cadence_start']);
  assert.deepEqual(f.sent.map(x => x.command), ['gamepad_stream_arm', 'gamepad_stream_handoff']);
  assert.equal(result.package.policyOwnershipVersion, 1);
  assert.equal(result.package.finalized, true);
  await f.executor.stop();
});

test('old-owner and unlabeled streamed results cannot enter the new epoch', async () => {
  const f = await fixture();
  await f.executor.handoff(f.prepare());
  const n = f.sent.length;
  const event = { index: 0, state: state(0), receivedPerfMs: performance.now() };
  const metadata = { requestId: 'late-A', scheduledStartPerfMs: performance.now(),
    policyOwnerId: 'A', policyOwnershipVersion: 0, policyUpdateVersion: 0 };
  f.executor.streamState(event, metadata);
  f.executor.streamState(event, { requestId: 'unlabeled', scheduledStartPerfMs: performance.now() });
  await f.executor.submissionTail;
  assert.equal(f.sent.length, n);
  f.executor.streamState(event, { ...metadata, requestId: 'B-next', policyOwnerId: 'B', policyOwnershipVersion: 1, policyUpdateVersion: 1 });
  await f.executor.submissionTail;
  assert.equal(f.sent.at(-1).command, 'gamepad_stream_state');
  assert.equal(f.sent.at(-1).payload.ownerId, 'B');
  assert.equal(f.sent.at(-1).payload.epoch, f.executor.epoch);
  await f.executor.stop();
});

test('rejected standby leaves active ownership and input untouched', async () => {
  const f = await fixture();
  const send = f.input.send;
  f.input.send = (cmd, payload) => cmd === 'gamepad_stream_handoff'
    ? Promise.resolve({ accepted: false, reason: 'stale_observation' }) : send(cmd, payload);
  const epoch = f.executor.epoch;
  const ack = await f.executor.handoff(f.prepare({ commit: () => assert.fail('no commit on rejection') }));
  assert.equal(ack.reason, 'stale_observation'); assert.equal(f.executor.epoch, epoch);
  assert.equal(f.executor.ownerId, 'A'); assert.equal(f.executor.armed, true);
  assert.equal(f.executor.accepting, true); assert.equal(f.executor.items.size, 0);
  assert.deepEqual(f.resets, ['stream_cadence_start']);
  await f.executor.stop();
});

test('version superseded while ordered pipe is busy cannot reach native handoff', async () => {
  const f = await fixture(); let release, current = true;
  f.executor.submissionTail = new Promise(resolve => { release = resolve; });
  const pending = f.executor.handoff(f.prepare({ isCurrent: () => current, commit: () => assert.fail('superseded') }));
  current = false; release();
  assert.equal((await pending).reason, 'superseded_update');
  assert.equal(f.sent.filter(x => x.command === 'gamepad_stream_handoff').length, 0);
  assert.equal(f.executor.items.size, 0); assert.equal(f.executor.ownerId, 'A');
  await f.executor.stop();
});

test('no HTTP status or partial/invalid packet is treated as handoff readiness', async () => {
  const f = await fixture();
  for (const states of [[], [state()], [state(), state(), state(), state(), state(), {}]]) {
    const ack = await f.executor.handoff(f.prepare({ pkg: { requestId: 'invalid', states, scheduledStartPerfMs: performance.now() + 10 } }));
    assert.equal(ack.reason, 'incomplete_or_invalid_action');
  }
  const stale = f.prepare({ observationPerfMs: performance.now() - 401 });
  assert.equal((await f.executor.handoff(stale)).reason, 'stale_observation');
  assert.equal(f.sent.filter(x => x.command === 'gamepad_stream_handoff').length, 0);
  await f.executor.stop();
});

test('valid empty states remain empty and repeated B-to-A swap increments execution ownership', async () => {
  const f = await fixture();
  const empty = f.prepare(); empty.pkg.states = Array.from({ length: 6 }, () => state(0));
  await f.executor.handoff(empty);
  assert(f.sent.at(-1).payload.states.every(s => s.rightTrigger === 0));
  const next = f.prepare({ expectedOwner: 'B', nextOwner: 'A', ownershipVersion: 2, updateVersion: 2 });
  next.pkg.requestId = 'A-next-context';
  assert((await f.executor.handoff(next)).accepted);
  assert.equal(f.executor.ownerId, 'A'); assert.equal(f.executor.ownershipVersion, 2);
  assert.deepEqual(f.resets, ['stream_cadence_start']);
  await f.executor.stop();
});

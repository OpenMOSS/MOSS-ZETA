const test = require('node:test');
const assert = require('node:assert/strict');
const { GenerationAdmission } = require('../src/generationAdmission');
const immediate = () => new Promise(resolve => setImmediate(resolve));
const gate = slot => ({ slot, executeAt: slot * 200 + 280, expiresAt: Infinity, closed: false,
  wait: async function () { this.closed = true; return { submission_slot: slot }; },
  cancel: function () { this.closed = true; } });

test('first streamed state does not admit another request before complete validated response', async () => {
  let finish; const started = [], commits = [], errors = [];
  const admission = new GenerationAdmission({
    generate: async ({ observation }) => {
      started.push(observation);
      if (observation === 'first') return new Promise(resolve => { finish = resolve; });
      return { committed: true };
    }, validate: result => result.committed === true, commit: result => commits.push(result), onError: e => errors.push(e) });
  admission.offer({ observation: 'first', gate: gate(0) });
  await immediate();
  // A first executable state may already have reached the independent executor.
  admission.offer({ observation: 'second', gate: gate(1) });
  await immediate(); assert.deepEqual(started, ['first']);
  finish({ committed: true });
  await admission.waitForIdle();
  assert.deepEqual(started, ['first', 'second']); assert.equal(commits.length, 2); assert.deepEqual(errors, []);
});

test('missed observations are replaced, never replayed as a backlog', async () => {
  let finish; const started = [], skipped = [];
  const admission = new GenerationAdmission({ generate: async ({ observation }) => {
    started.push(observation); if (observation === 0) return new Promise(resolve => { finish = resolve; });
    return true;
  }, validate: result => result === true, onSkip: event => skipped.push(event) });
  admission.offer({ observation: 0, gate: gate(0) }); await immediate();
  const stale = gate(1); admission.offer({ observation: 1, gate: stale });
  admission.offer({ observation: 2, gate: gate(2) });
  assert(stale.closed); finish(true); await admission.waitForIdle();
  assert.deepEqual(started, [0, 2]); assert.equal(skipped[0].reason, 'superseded_observation');
});

test('invalid completion closes pending admission and cannot commit history', async () => {
  let finish; const errors = [], commits = [];
  const admission = new GenerationAdmission({ generate: () => new Promise(resolve => { finish = resolve; }),
    validate: result => result.committed === true, commit: result => commits.push(result), onError: error => errors.push(error) });
  admission.offer({ observation: 0, gate: gate(0) }); await immediate();
  const next = gate(1); admission.offer({ observation: 1, gate: next });
  finish({ committed: false }); await admission.waitForIdle();
  assert(admission.stopped); assert(next.closed); assert.equal(commits.length, 0); assert.equal(errors.length, 1);
});

test('expired slot is discarded before invoking the model callback', async () => {
  let generated = 0; const skipped = [];
  const admission = new GenerationAdmission({ generate: async () => { generated++; }, validate: () => true,
    now: () => 1000, onSkip: event => skipped.push(event) });
  admission.offer({ observation: 0, gate: { ...gate(0), expiresAt: 900 } });
  await admission.waitForIdle();
  assert.equal(generated, 0); assert.equal(skipped[0].reason, 'expired_slot');
});

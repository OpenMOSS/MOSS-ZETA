const { createHash } = require('node:crypto');

const VERSION = 'committed-generated-history-v1';
const sha256 = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');

function stateFromSystem(system) {
  const match = typeof system === 'string' && system.match(/\n\n当前有效目标：([\s\S]*?)\n\nMemory：([\s\S]*?)\n\n模型思考：([\s\S]*)$/);
  if (!match) throw Error('History recovery requires the current canonical Policy state');
  return { instruction: match[1], memory: match[2], model_thought: match[3] };
}

function validateRecovery(receipt, { sessionId, recoveryId, state, window }) {
  const hash = /^[a-f0-9]{64}$/;
  const stateHash = sha256(Object.fromEntries(Object.entries(state).sort(([a], [b]) => a.localeCompare(b))));
  if (receipt?.recovery_version !== VERSION || receipt.ready !== true
    || receipt.session_id !== sessionId || receipt.recovery_id !== recoveryId
    || receipt.state_sha256 !== stateHash || receipt.window_max !== window.max || receipt.window_keep !== window.keep
    || !Number.isSafeInteger(receipt.history_turns) || receipt.history_turns < 0 || receipt.history_turns > window.max
    || !Number.isSafeInteger(receipt.turn) || receipt.turn < receipt.history_turns
    || !hash.test(receipt.history_sha256 || '') || !hash.test(receipt.prefix_sha256 || '')
    || receipt.history_sha256 !== receipt.prefix_sha256
    || !Number.isSafeInteger(receipt.image_count) || receipt.image_count < 0
    || receipt.image_count > 2 * receipt.history_turns
    || (receipt.history_turns > 0 && receipt.image_count === 0)
    || !Array.isArray(receipt.routes) || receipt.routes.length !== 2
    || ![0, 1].every(route => receipt.routes.some(r => r.route === route && r.prepared === true
      && r.prefix_sha256 === receipt.prefix_sha256 && r.image_count === receipt.image_count))) {
    throw Error('Deployment did not acknowledge the retained generated history and both prepared routes');
  }
  const ocr = receipt.ocr_state;
  if (!(ocr === null || (ocr && typeof ocr === 'object' && !Array.isArray(ocr)
    && Object.keys(ocr).length === 4 && ['P', 'B', 'C', 'K'].every(k => typeof ocr[k] === 'string'))))
    throw Error('History recovery returned an invalid committed OCR state');
  if (receipt.history_turns > 0 && ocr === null) throw Error('History recovery lost committed OCR state');
  return receipt;
}

module.exports = { VERSION, stateFromSystem, validateRecovery };

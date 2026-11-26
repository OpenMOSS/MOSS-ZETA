const { migrateMemory, outputKeys } = require('./brainProtocol');
function emptyMemory() { return ''; }
function text(value, name, required = false) {
  if (typeof value !== 'string' || value.length > 12000 || (required && !value.trim())) throw new Error(`Invalid Brain ${name}`);
  return value.trim();
}
function validateMemory(value) { return text(value, 'memory').replace(/\s*\n\s*/g, ' '); }
function validateBrainOutput(value, agentMode, type = 'full', scope = 'memory_think') {
  const keys = outputKeys(type, scope);
  if (!value || Array.isArray(value) || Object.keys(value).length !== keys.length || keys.some(k => !Object.hasOwn(value, k))) throw new Error(`Brain output (${type}) requires exactly: ${keys.join(', ')}`);
  return type === 'small' ? { memory: validateMemory(value.memory) } : {
    ...(keys.includes('instruction') ? { instruction: text(value.instruction, 'instruction', true) } : {}),
    memory: validateMemory(value.memory), think: text(value.think, 'think')
  };
}
function migrateBrainState(value = {}) {
  const result = validateBrainOutput({instruction: value.instruction, memory: migrateMemory(value.memory), think: value.think ?? value.model_thought ?? ''});
  return { ...result, plan: String(value.plan || ''), phaseBoundary: String(value.phaseBoundary || '') };
}
function brainView(state) {
  return { instruction: state.currentPolicyInstruction?.text || '', memory: state.brain.memory, think: state.brain.think || '' };
}
module.exports = { emptyMemory, validateMemory, validateBrainOutput, brainView, migrateBrainState };

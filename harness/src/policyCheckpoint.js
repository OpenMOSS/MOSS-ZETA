function validatePolicyCheckpoint(value) {
  if (!value || value.schemaVersion !== 2 || !/^[a-z0-9][a-z0-9-]{0,79}$/.test(value.id || "")) {
    throw new Error("Checkpoint requires schemaVersion=2 and a lowercase id.");
  }
  const text = (value, field, required = false) => {
    if (value !== undefined && typeof value !== "string") throw new Error(`Checkpoint ${field} must be text.`);
    const result = String(value || "").trim();
    if ((required && !result) || result.length > 4000) throw new Error(`Invalid checkpoint ${field}.`);
    return result;
  };
  const brain = require('./brainSchema').migrateBrainState(value.brain);
  return {
    schemaVersion: 2,
    id: value.id,
    synthetic: true,
    taskInstruction: text(value.taskInstruction, "taskInstruction", true),
    setup: text(value.setup, "setup"),
    sourceSessionId: text(value.sourceSessionId, "sourceSessionId"),
    sourceObservationId: text(value.sourceObservationId, "sourceObservationId"),
    brain
  };
}

function policyCheckpointConfig(config) {
  return {
    ...config,
    continuationHandoffPath: "",
    sessionOutputSubpath: "agent-framework/policy-checkpoint"
  };
}

module.exports = { validatePolicyCheckpoint, policyCheckpointConfig };

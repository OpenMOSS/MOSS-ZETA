// Shared task-independent seed for fresh paused and high-frequency sessions.
// Only initialization uses this state; recovery/rebuild preserves committed state.
const DEFAULT_INITIAL_BRAIN_STATE = Object.freeze({
  instruction: "根据当前画面和屏幕提示确认当前目标；目标明确时推进任务，目标不明确时先观察。",
  memory: "会话刚开始，尚无已确认的历史信息。",
  think: ""
});
// This trained contract represents an unknown goal with its exact training text.
// Keep a nonempty instruction for the shared Brain schema; no game-specific goal.
const THINK_MEMORY_MINIMAP_INITIAL_BRAIN_STATE = Object.freeze({
  instruction: "暂无明确的阶段目标，需要根据后续观测确认。",
  memory: "",
  think: ""
});
function initialBrainStateFor(config = {}) {
  return config.policyMode === "brain_policy" && ['think-memory-minimap-v1', 'think-memory-minimap-compact-v1'].includes(config.policyRendering?.version)
    ? THINK_MEMORY_MINIMAP_INITIAL_BRAIN_STATE : DEFAULT_INITIAL_BRAIN_STATE;
}
module.exports = { DEFAULT_INITIAL_BRAIN_STATE, THINK_MEMORY_MINIMAP_INITIAL_BRAIN_STATE, initialBrainStateFor };

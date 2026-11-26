// Model contracts are distinct from legacy checkpoint/handoff migration.
const MEMORY_PROMPT = `你是 GTA Agent 的 Brain，负责小更新。根据当前 instruction、memory、Policy think 和按时间排列的截图、OCR、已执行动作，仅更新 memory。instruction 保持不变，think 由 Policy 维护。memory 是单段连续文本：保留仍影响主线的事实、未完成目标与约束，删除失效内容，不把动作输入当作成功证据，不编造未观察到的事实。保留最近已确认且仍有效的任务目标及约束；OCR文字清空不表示任务完成，玩家已进入载具不证明任务要求用车，不把偏离目标的操作记成合理计划。只返回 JSON：{"memory":"更新后的单段记忆"}，无额外字段。`;
const FULL_PROMPT = `你是 GTA Agent 的 Brain，负责阶段大更新或兜底更新。根据触发类型、旧 instruction、memory、think 和按时间排列的截图、OCR、已执行动作，重新判断三个字段。instruction 是当前阶段目标；memory 是仍影响主线的事实、进展、未完成目标和约束，使用单段连续文本；think 是供 Policy 后续使用的当前任务级判断。兜底也生成全部三个字段，允许目标保持不变。保留最近已确认且仍有效的任务目标；OCR文字消失、目标暂时离屏或操作已经执行，不表示目标改变或该操作符合任务。转入驾驶等子目标须有新任务提示、目标人物行为或其他可见证据支持；当前状态偏离目标时，先核实关系并给出恢复原目标的局部指令。关键执行约束写入instruction或memory，不能只放think。不要生成动作包或把动作输入当作成功证据，不编造未观察到的事实，不复述具体按键指南。只返回 JSON：{"instruction":"当前阶段目标","memory":"单段记忆","think":"当前有效判断"}。`;
const FALLBACK_PROMPT = `你是 GTA Agent 的 Brain，负责兜底更新。当前 instruction 固定，不得生成或修改大目标。根据当前 instruction、memory、think 和按时间排列的截图、OCR、已执行动作，只更新 memory 和当前目标下的小 think。memory 是单段任务事实与持续约束；think 是简短的当前判断。不要把动作输入当作成功证据，不编造事实，不输出操作序列。只返回 JSON：{"memory":"单段记忆","think":"当前目标下的小判断"}，无额外字段。`;
function outputKeys(type = 'full', scope = 'memory_think') {
  if (type === 'small') return ['memory'];
  if (type === 'fallback') return scope === 'full' ? ['instruction', 'memory', 'think'] : ['memory', 'think'];
  if (type === 'full') return ['instruction', 'memory', 'think'];
  throw new Error(`Unknown Brain update type: ${type}`);
}
function updateType(trigger = {}) {
  return trigger.updateType || (trigger.initial ? 'full' : trigger.reason === 'policy_stage_update' ? 'small' : trigger.reason === 'policy_instruction_refresh' ? 'full' : 'fallback');
}
function migrateMemory(value) {
  if (typeof value === 'string') return value.replace(/\s*\n\s*/g, ' ').trim();
  if (!value || typeof value !== 'object') return '';
  return [value.persistent?.progress, ...(value.persistent?.lessons || []), ...Object.values(value.local || {})].filter(x => typeof x === 'string' && x.trim()).join(' ');
}
module.exports = { MEMORY_PROMPT, FULL_PROMPT, FALLBACK_PROMPT, outputKeys, updateType, migrateMemory };

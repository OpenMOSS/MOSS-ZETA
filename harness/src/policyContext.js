// Exact dynamic fields shared by both execution modes.
function buildPolicySystemMessage(host) {
  if (host.agentMode === 'high_frequency' && require('./policyHandoffContext').enabled(host.defaultConfig))
    return require('./policyHandoffContext').systemMessage(host);
  const brain = host.currentSession?.brain || {};
  const instruction = host.currentSession?.currentPolicyInstruction?.text || host.taskInstruction || '';
  const config = host.defaultConfig || {};
  const prompt = String(host.systemPrompt || '').replace(/图片和动作历史每满\d+轮保留最近\d+轮/g,
    `图片和动作历史每满${config.policyWindowMaxRounds || 16}轮保留最近${config.policyWindowKeepRounds || 8}轮`);
  return { role: 'system', content: [prompt,
    ...(config.policyIncludeMinimap ? ['每个观察轮依次提供完整主图和从该主图裁出的同帧小地图；两张图属于同一时刻，不是两个时间点。'] : []),
    `当前有效目标：${instruction || '暂无明确的阶段目标，需要根据后续观测确认。'}`,
    `Memory：${brain.memory || '暂无。'}`, `模型思考：${brain.think || ''}`].join('\n\n') };
}
module.exports = { buildPolicySystemMessage };

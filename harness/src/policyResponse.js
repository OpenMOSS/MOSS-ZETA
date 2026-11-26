const THINKING_TAG_PATTERN = /<(thinking|thought|thoughts|reasoning)\b[^>]*>([\s\S]*?)<\/\1>/gi;
const THINKING_OPEN_TAG_PATTERN = /<(thinking|thought|thoughts|reasoning)\b[^>]*>/i;
const THINKING_CLOSE_TAG_PATTERN = /<\/(thinking|thought|thoughts|reasoning)>/i;
// Legacy <t>/<i> remain accepted; only a nonempty judgment or call marker requests Brain.
const POLICY_TRIGGER_TAG = /<\/?(?:t|think)\b[^>]*>/gi;
function extractPolicyTrigger(raw) {
  const tags = [...raw.matchAll(POLICY_TRIGGER_TAG)];
  let block = "", inner = "";
  if (tags.length) {
    const opening = tags[0][0].match(/^<(t|think)(?:\s[^<>]*)?>$/i);
    if (tags.length !== 2 || !opening
      || tags[1][0].toLowerCase() !== `</${opening[1].toLowerCase()}>`) {
      throw new Error("Policy response requires at most one complete <t> or <think> Brain trigger block.");
    }
    const start = tags[0].index, end = tags[1].index + tags[1][0].length;
    block = raw.slice(start, end);
    inner = raw.slice(start + tags[0][0].length, tags[1].index);
  }
  const instructionUpdateRequested = /<i\s*\/?>/i.test(inner) || inner.includes('<|call_brain|>');
  if (raw.replace(block, '').includes('<|call_brain|>')) throw new Error('call_brain must be inside think');
  if (/<i\b/i.test(inner) && !instructionUpdateRequested) {
    throw new Error("Paused Policy <i> marker is malformed.");
  }
  const actionContent = raw.replace(block, "").trim();
  if (/<\/?i\b/i.test(actionContent)) {
    throw new Error("Paused Policy <i> marker must be nested inside <t> or <think>.");
  }
  return {
    content: actionContent,
    triggerThinking: inner.replace(/<\/?i\s*\/?>/gi, "").replaceAll('<|call_brain|>', '').trim(),
    brainRequested: instructionUpdateRequested || Boolean(inner.trim()),
    instructionUpdateRequested
  };
}

function extractPolicyResponse(content, independentThinking = "", renderingVersion = "") {
  if (['think-memory-minimap-v1', 'think-memory-minimap-compact-v1'].includes(renderingVersion)) {
    const raw=String(content ?? '');
    const match=raw.match(/^<think>([\s\S]*?)<\/think>\s*(<a>[^<>]*<\/a>)$/);
    if (!match) throw Error('think-memory-minimap-v1 requires complete think and action blocks');
    if (renderingVersion === 'think-memory-minimap-compact-v1'
      && (/[\r\n]/.test(match[1]) || !raw.includes('</think><a>')))
      throw Error('Compact think requires consecutive tags and no literal newlines');
    const parts=match[1].split('<|call_brain|>');
    if(parts.length>2) throw Error('Repeated call_brain marker');
    let policyProposal=null;
    if(parts.length===2) {
      policyProposal=JSON.parse(parts[1].trim());
      if(!policyProposal || Array.isArray(policyProposal) ||
        Object.keys(policyProposal).sort().join(',')!=='instruction,small_think' ||
        typeof policyProposal.instruction!=='string' || !policyProposal.instruction.trim() ||
        typeof policyProposal.small_think!=='string') throw Error('Invalid instruction/small_think proposal');
    }
    const triggerThinking=parts[0].trim();
    return {content:match[2],thinking:[independentThinking,triggerThinking].filter(Boolean).join('\n'),
      triggerThinking,policyProposal,brainRequested:Boolean(triggerThinking)||Boolean(policyProposal),
      instructionUpdateRequested:Boolean(policyProposal)};
  }
  const raw = String(content ?? "");
  const matches = [...raw.matchAll(THINKING_TAG_PATTERN)];
  const hasThinkingOpen = THINKING_OPEN_TAG_PATTERN.test(raw);
  const hasThinkingClose = THINKING_CLOSE_TAG_PATTERN.test(raw);
  if (matches.length > 1) {
    throw new Error("Paused Policy response may contain at most one thinking tag block.");
  }
  if (hasThinkingOpen !== hasThinkingClose || (hasThinkingOpen && matches.length !== 1)) {
    throw new Error("Paused Policy thinking tag is not properly closed.");
  }

  const taggedThinking = matches[0]?.[2]?.trim() || "";
  const thinking = [String(independentThinking || "").trim(), taggedThinking]
    .filter(Boolean)
    .join("\n");
  const withoutThinkingTags = raw.replace(THINKING_TAG_PATTERN, "").trim();
  const trigger = extractPolicyTrigger(withoutThinkingTags);
  return {
    content: trigger.content,
    thinking: [thinking, trigger.triggerThinking].filter(Boolean).join("\n"),
    triggerThinking: trigger.triggerThinking,
    brainRequested: trigger.brainRequested,
    instructionUpdateRequested: trigger.instructionUpdateRequested
  };
}

module.exports = {
  extractPolicyResponse
};

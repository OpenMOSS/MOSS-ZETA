function usesGlmAnthropic(baseUrl, model) {
  return /(?:^|\/)glm-[\w.-]+$/i.test(String(model || "").trim())
    && /\/anthropic(?:\/v1)?(?:\/messages)?\/?$/i.test(String(baseUrl || "").trim());
}

function glmAnthropicEndpoint(baseUrl) {
  const base = String(baseUrl).trim().replace(/\/+$/, "");
  if (/\/messages$/i.test(base)) return base;
  return `${base}${/\/v1$/i.test(base) ? "" : "/v1"}/messages`;
}

function contentBlocks(content) {
  if (typeof content === "string") return content ? [{ type: "text", text: content }] : [];
  if (!Array.isArray(content)) throw new Error("GLM Messages requires text or image content.");
  return content.map((part) => {
    if (part.type === "text" && typeof part.text === "string") return { type: "text", text: part.text };
    if (part.type === "image_url") {
      const url = part.image_url?.url;
      const match = /^data:(image\/(?:jpeg|png|gif|webp));base64,([A-Za-z0-9+/=\r\n]+)$/.exec(url || "");
      if (match) return { type: "image", source: { type: "base64", media_type: match[1], data: match[2] } };
      if (/^https?:\/\//i.test(url || "")) return { type: "image", source: { type: "url", url } };
      throw new Error("GLM Messages image must be a supported base64 data URL or HTTP(S) URL.");
    }
    throw new Error(`Unsupported GLM Messages content block: ${part.type}`);
  });
}

function glmAnthropicRequest(config, model, messages) {
  const system = [], conversation = [];
  for (const message of messages) {
    if (!["system", "user", "assistant"].includes(message.role) || message.tool_calls) {
      throw new Error("GLM Messages adapter supports system/user/assistant text and images only.");
    }
    const content = contentBlocks(message.content);
    if (message.role === "system") {
      if (conversation.length || content.some((part) => part.type !== "text")) {
        throw new Error("GLM Messages system content must be leading text only.");
      }
      system.push(...content);
    } else if (content.length) {
      if (message.role === "assistant" && content.some((part) => part.type !== "text")) {
        throw new Error("GLM Messages assistant history must contain text only.");
      }
      const previous = conversation.at(-1);
      if (previous?.role === message.role) previous.content.push(...content);
      else conversation.push({ role: message.role, content });
    }
  }
  if (conversation[0]?.role !== "user" || conversation.at(-1)?.role !== "user") {
    throw new Error("GLM Messages conversation must begin and end with a user observation.");
  }
  const maxTokens = config.maxTokens ?? 4096;
  // GLM-5 is always-thinking; use native Messages budgets, not OpenAI fields.
  const thinking = /(?:^|\/)glm-5(?:[.-]|$)/i.test(model) || config.allowThinking === true;
  if (thinking && (!Number.isInteger(maxTokens) || maxTokens <= 1024)) {
    throw new Error("GLM Messages thinking requires maxTokens > 1024 to leave room for the answer.");
  }
  const requestedBudget = config.reasoningEffort === "max" ? maxTokens - 1024
    : config.reasoningEffort === "high" ? Math.floor(maxTokens / 2) : 1024;
  return {
    model, max_tokens: maxTokens, stream: false,
    ...(system.length ? { system } : {}), messages: conversation,
    thinking: thinking ? { type: "enabled", budget_tokens: Math.min(maxTokens - 1, Math.max(1024, requestedBudget)) }
      : { type: "disabled" },
    ...(!thinking ? { temperature: config.temperature ?? 0.2 } : {})
  };
}

function glmAnthropicResponse(body) {
  if (body?.error || body?.success === false || (body?.code && !body?.content)) {
    const error = new Error(`GLM Messages provider error: code=${body.error?.code ?? body.error?.type ?? body.code ?? "unknown"}; message=${String(body.error?.message ?? body.msg ?? body.message ?? "unknown").slice(0,500)}`);
    error.body = body;
    throw error;
  }
  if (body?.type !== "message" || body?.role !== "assistant" || !Array.isArray(body.content)) {
    throw new Error("GLM Messages response is not an assistant message with content blocks.");
  }
  if (body.content.some((part) => !["text", "thinking", "redacted_thinking"].includes(part.type))) {
    throw new Error("GLM Messages returned unsupported non-text output; no action accepted.");
  }
  const content = body.content.filter((part) => part.type === "text").map((part) => part.text || "").join("");
  const thinking = body.content.filter((part) => part.type === "thinking").map((part) => part.thinking || "").join("\n");
  const inputTokens = Number(body.usage?.input_tokens || 0) + Number(body.usage?.cache_read_input_tokens || 0) + Number(body.usage?.cache_creation_input_tokens || 0);
  const outputTokens = Number(body.usage?.output_tokens || 0);
  return {
    choices: [{ message: { role: "assistant", content, reasoning_content: thinking },
      finish_reason: ({ end_turn: "stop", stop_sequence: "stop", max_tokens: "length" })[body.stop_reason] || body.stop_reason }],
    usage: { ...body.usage, prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    __client_timing: body.__client_timing
  };
}

module.exports = { usesGlmAnthropic, glmAnthropicEndpoint, glmAnthropicRequest, glmAnthropicResponse };

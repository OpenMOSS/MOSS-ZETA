// Explicit native Responses transport. No model/profile selection or game policy.
function usesResponses(config) {
  return config.apiFormat === "responses";
}

function responsesEndpoint(baseUrl) {
  const base = String(baseUrl || "").trim().replace(/\/+$/, "");
  if (!base || /\/chat\/completions$/.test(base)) throw new Error("Responses requires its own base URL, not Chat Completions.");
  return /\/responses$/.test(base) ? base : `${base}/responses`;
}

function responsesRequest(config, model, messages) {
  return {
    model,
    input: messages.map(({ role, content }) => ({ role, content: Array.isArray(content)
      ? content.map(part => {
        if (part.type === "text") return { type: role === "assistant" ? "output_text" : "input_text", text: part.text };
        if (part.type === "image_url" && role !== "assistant") return {
          type: "input_image", image_url: part.image_url.url,
          ...(part.image_url.detail ? { detail: part.image_url.detail } : {})
        };
        throw new Error(`Unsupported Responses input block: ${role}/${part.type}`);
      }) : content })),
    reasoning: { effort: config.reasoningEffort },
    max_output_tokens: config.maxTokens ?? 16384,
    store: false,
    stream: false
  };
}

function terminalResponse(body) {
  if (typeof body?.raw !== "string") return body;
  let response;
  for (const line of body.raw.split(/\r?\n/)) {
    if (!line.startsWith("data:")) continue;
    const text = line.slice(5).trim();
    if (!text || text === "[DONE]") continue;
    const event = JSON.parse(text);
    if (event.type === "error") throw new Error(`Responses stream error: ${event.message || event.error?.message || "unknown"}`);
    if (["response.completed", "response.incomplete", "response.failed"].includes(event.type)) response = event.response;
  }
  if (!response) throw new Error("Responses stream ended without a terminal response; partial deltas are not actions.");
  return response;
}

function responsesResult(body, requestedModel) {
  const raw = terminalResponse(body);
  const fail = (message, code) => { const error = new Error(message); error.body = raw; if (code) error.code = code; throw error; };
  if (raw?.error || raw?.status === "failed") fail(`Responses provider error: ${raw?.error?.message || raw?.status}`);
  if (!["completed", "incomplete"].includes(raw?.status)) fail("Responses response missing completed/incomplete status.");
  if (raw.model && raw.model !== requestedModel) fail(`Responses model mismatch: requested ${requestedModel}, received ${raw.model}; no fallback accepted.`);
  if ((raw.output || []).some(item => !["message", "reasoning"].includes(item.type))) fail("Unexpected tool output in no-tools Responses run.");
  const messages = (raw.output || []).filter(item => item.type === "message");
  // A Responses turn can contain commentary and a separate final answer. They
  // are distinct messages, not fragments of one executable assistant reply.
  const phased = messages.some(item => item.phase != null);
  const finalMessages = phased ? messages.filter(item => item.phase === "final_answer") : messages;
  if (!finalMessages.length) fail(phased
    ? "Responses final_answer message missing from phase-tagged output."
    : "Responses assistant message missing.", "RESPONSES_FINAL_MESSAGE_MISSING");
  if (finalMessages.length !== 1) fail(phased
    ? "Responses output is ambiguous: multiple final_answer messages."
    : "Responses output is ambiguous: multiple messages without phase.", "RESPONSES_FINAL_MESSAGE_AMBIGUOUS");
  const content = (finalMessages[0].content || []).filter(part => part.type === "output_text")
    .map(part => part.text || "").join("");
  return {
    raw, content,
    finishReason: raw.status === "incomplete" ? "length" : "stop",
    usage: raw.usage ? { ...raw.usage, prompt_tokens: raw.usage.input_tokens, completion_tokens: raw.usage.output_tokens } : null
  };
}

module.exports = { usesResponses, responsesEndpoint, responsesRequest, terminalResponse, responsesResult };

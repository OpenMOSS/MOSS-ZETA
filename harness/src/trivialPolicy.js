const fs = require("node:fs/promises");
const path = require("node:path");

function isTrivialPolicy(config = {}) {
  return config.agentMode === "high_frequency" && config.policyMode === "trivial";
}

function loadTrivialPolicyPrompt() {
  return fs.readFile(path.join(__dirname, "../prompts/trivial-policy-system-prompt.txt"), "utf8");
}

function buildTrivialSystemMessage(host) {
  // Byte-identical (after trim) to the checkpoint's training system. No dynamic state.
  return { role: "system", content: host.systemPrompt.trim() };
}

function extractTrivialPolicyResponse(content, independentThinking = "") {
  const raw = String(content ?? "").trim();
  const action = raw.match(/^<a>([^<>]+)<\/a>$/)?.[1] ?? raw;
  if (String(independentThinking || "").trim()
      || !/^(?:\+0|[+-][3-9]){4}[A-Za-z0-9;]*$/.test(action)
      || action.split(";").length !== 6) {
    throw new Error("Trivial Policy requires only one six-chunk compact action package.");
  }
  const chunks = action.slice(8).split(";");
  if (chunks[0] !== chunks[1] || chunks[1] !== chunks[2] || chunks[3] !== chunks[4] || chunks[4] !== chunks[5]) {
    throw new Error("Trivial Policy requires two groups of three equal button chunks.");
  }
  return { content: raw, thinking: "", triggerThinking: "",
    brainRequested: false, instructionUpdateRequested: false };
}

function trivialOcrSnapshot(snapshot) {
  const lines = ["S", ...["P", "B", "C", "K"].filter(key => snapshot[key]).map(key => `${key}=${snapshot[key]}`)];
  return require("./asyncPolicyOcr").ocrBlock(lines.join("\n"));
}

function compactTrivialOcrHistory(messages, reservation, intervalMs, now = performance.now()) {
  if (!intervalMs) return reservation.records;
  const aggregate = messages.find(message => message.ocrAggregate);
  if (!aggregate) throw Error('Trivial Policy requires an OCR aggregate.');
  // The timer lives in the same transactional message snapshot as the archive.
  // A fixed-submission skip restores both; it cannot consume a reset boundary.
  if (aggregate.ocrHistoryEpochStartedAt == null) aggregate.ocrHistoryEpochStartedAt = now;
  if (now - aggregate.ocrHistoryEpochStartedAt < intervalMs) return reservation.records;
  aggregate.content = '<screen_ocr></screen_ocr>';
  aggregate.ocrRecords = [];
  aggregate.ocrHistoryEpochStartedAt = now;
  aggregate.ocrHistoryEpoch = (aggregate.ocrHistoryEpoch || 0) + 1;
  // Retained visual/action turns survive, but their old OCR events must not
  // reappear in the archive when those turns are evicted later.
  for (const message of messages) if (Array.isArray(message.ocrRecords)) message.ocrRecords = [];
  const text = trivialOcrSnapshot(reservation.snapshot).slice('<screen_ocr>\n'.length, -'\n</screen_ocr>'.length);
  return [{ id: `ocr-reset:${reservation.token}`, observationId: null, text,
    source: 'periodic_delivered_state_snapshot', epoch: aggregate.ocrHistoryEpoch }];
}

module.exports = { compactTrivialOcrHistory, trivialOcrSnapshot, isTrivialPolicy, loadTrivialPolicyPrompt, buildTrivialSystemMessage, extractTrivialPolicyResponse };

const fs = require("node:fs/promises");
const path = require("node:path");
const { configuredPath } = require("./projectPaths");

const BEHAVIOR_GUIDE_PATH = configuredPath(
  process.env.BEHAVIOR_GUIDE_PATH,
  "logs",
  "behavior-guide.json"
);

const BRAIN_SYSTEM_PROMPT_PATH = configuredPath(
  process.env.BRAIN_SYSTEM_PROMPT_PATH,
  "prompts",
  "brain-system-prompt.txt"
);

const BRAIN_MEMORY_SYSTEM_PROMPT_PATH = configuredPath(process.env.BRAIN_MEMORY_SYSTEM_PROMPT_PATH,
  'prompts', 'brain-memory-system-prompt.txt');
async function loadBrainMemorySystemPrompt() {
  try { return await fs.readFile(BRAIN_MEMORY_SYSTEM_PROMPT_PATH, 'utf8'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; return require('./brainProtocol').MEMORY_PROMPT; }
}
async function saveBrainMemorySystemPrompt(prompt) {
  if (typeof prompt !== 'string' || !prompt.trim()) throw new Error('Brain memory prompt cannot be empty.');
  await fs.mkdir(path.dirname(BRAIN_MEMORY_SYSTEM_PROMPT_PATH), { recursive: true });
  await fs.writeFile(BRAIN_MEMORY_SYSTEM_PROMPT_PATH, prompt, 'utf8');
  return { prompt, path: BRAIN_MEMORY_SYSTEM_PROMPT_PATH };
}

const POLICY_SYSTEM_PROMPT_PATH = configuredPath(
  process.env.POLICY_SYSTEM_PROMPT_PATH || process.env.PAUSED_PACKAGE_POLICY_SYSTEM_PROMPT_PATH,
  "prompts", "policy-system-prompt.txt"
);

const DEFAULT_BRAIN_SYSTEM_PROMPT = require('./brainProtocol').FULL_PROMPT;

async function ensureBrainPrompt() {
  await fs.mkdir(path.dirname(BRAIN_SYSTEM_PROMPT_PATH), { recursive: true });
  try {
    await fs.access(BRAIN_SYSTEM_PROMPT_PATH);
  } catch {
    await fs.writeFile(BRAIN_SYSTEM_PROMPT_PATH, DEFAULT_BRAIN_SYSTEM_PROMPT, "utf8");
  }
}

async function loadBrainSystemPrompt() {
  await ensureBrainPrompt();
  return fs.readFile(BRAIN_SYSTEM_PROMPT_PATH, "utf8");
}

async function saveBrainSystemPrompt(prompt) {
  if (typeof prompt !== "string" || !prompt.trim()) {
    throw new Error("Brain prompt cannot be empty.");
  }
  await fs.mkdir(path.dirname(BRAIN_SYSTEM_PROMPT_PATH), { recursive: true });
  await fs.writeFile(BRAIN_SYSTEM_PROMPT_PATH, prompt, "utf8");
  return {
    prompt,
    path: BRAIN_SYSTEM_PROMPT_PATH
  };
}

class SharedDocuments {
  constructor(documents = {}) {
    this.documents = Object.freeze({ ...documents });
    this.behaviorGuidePath = documents.behaviorGuidePath || BEHAVIOR_GUIDE_PATH;
  }

  get(name) {
    return this.documents[name] || "";
  }

  snapshot() {
    return {
      brainSystemPromptPath: this.documents.brainSystemPromptPath || BRAIN_SYSTEM_PROMPT_PATH,
      brainMemorySystemPromptPath: BRAIN_MEMORY_SYSTEM_PROMPT_PATH,
      brainMemorySystemPrompt: this.documents.brainMemorySystemPrompt ? "loaded" : "missing",
      policySystemPrompt: this.documents.policySystemPrompt ? "loaded" : "missing",
      brainSystemPrompt: this.documents.brainSystemPrompt ? "loaded" : "missing",
      gtaKnowledge: this.documents.gtaKnowledge ? "loaded" : "default",
      behaviorGuide: this.documents.behaviorGuide ? "loaded" : "none"
    };
  }

  async persistBehaviorGuideEntries(entries = []) {
    const additions = entries
      .filter((entry) => entry && String(entry.text || "").trim())
      .map((entry) => ({
        text: String(entry.text).trim(),
        sourceObservationId: entry.sourceObservationId || null,
        sourceSessionId: entry.sourceSessionId || null,
        source: entry.source || "brain_tutorial_ocr",
        confidence: entry.confidence ?? null,
        createdAt: entry.createdAt || new Date().toISOString()
      }));
    if (additions.length === 0) return [];
    let existing = [];
    try {
      existing = JSON.parse(await fs.readFile(this.behaviorGuidePath, "utf8"));
      if (!Array.isArray(existing)) existing = [];
    } catch {
      existing = [];
    }
    const seen = new Set(existing.map((entry) => String(entry?.text || "").trim()).filter(Boolean));
    const unique = additions.filter((entry) => {
      if (seen.has(entry.text)) return false;
      seen.add(entry.text);
      return true;
    });
    if (unique.length === 0) return [];
    await fs.mkdir(path.dirname(this.behaviorGuidePath), { recursive: true });
    const temporaryPath = `${this.behaviorGuidePath}.${process.pid}.tmp`;
    await fs.writeFile(temporaryPath, `${JSON.stringify([...existing, ...unique], null, 2)}\n`, "utf8");
    await fs.rename(temporaryPath, this.behaviorGuidePath);
    return unique;
  }

  static async load(options = {}) {
    const brainMemorySystemPrompt = await loadBrainMemorySystemPrompt();
    const [policySystemPrompt, brainSystemPrompt, gtaKnowledge, behaviorGuide] = await Promise.all([
      options.policySystemPromptProvider ? options.policySystemPromptProvider() : fs.readFile(POLICY_SYSTEM_PROMPT_PATH, "utf8"),
      options.brainSystemPromptProvider ? options.brainSystemPromptProvider() : loadBrainSystemPrompt(),
      fs.readFile(path.join(__dirname, "../prompts/high-frequency-controller-knowledge.md"), "utf8"),
      fs.readFile(BEHAVIOR_GUIDE_PATH, "utf8").catch(() => "")
    ]);
    return new SharedDocuments({
      brainMemorySystemPrompt,
      policySystemPrompt: String(policySystemPrompt).trim(),
      brainSystemPrompt: String(brainSystemPrompt).trim(),
      gtaKnowledge,
      behaviorGuide,
      brainSystemPromptPath: BRAIN_SYSTEM_PROMPT_PATH
    });
  }
}

module.exports = {
  BRAIN_MEMORY_SYSTEM_PROMPT_PATH, loadBrainMemorySystemPrompt, saveBrainMemorySystemPrompt,
  BRAIN_SYSTEM_PROMPT_PATH,
  BEHAVIOR_GUIDE_PATH,
  DEFAULT_BRAIN_SYSTEM_PROMPT,
  loadBrainSystemPrompt,
  saveBrainSystemPrompt,
  SharedDocuments
};

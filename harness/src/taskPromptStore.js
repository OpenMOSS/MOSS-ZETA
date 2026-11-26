const fs = require("node:fs/promises");
const path = require("node:path");
const { configuredPath } = require("./projectPaths");

const DEFAULT_TASK_PROMPT = "Current task: freely explore the world. Prioritize staying safe, observe the surroundings, and explore interesting places by driving or walking.";
const TASK_PROMPT_PATH = configuredPath(process.env.TASK_PROMPT_PATH, "prompts", "task-prompt.txt");

async function ensurePromptFile() {
  await fs.mkdir(path.dirname(TASK_PROMPT_PATH), { recursive: true });

  try {
    await fs.access(TASK_PROMPT_PATH);
  } catch (error) {
    await fs.writeFile(TASK_PROMPT_PATH, DEFAULT_TASK_PROMPT, "utf8");
  }
}

async function loadTaskPrompt() {
  await ensurePromptFile();
  return fs.readFile(TASK_PROMPT_PATH, "utf8");
}

async function saveTaskPrompt(prompt) {
  if (typeof prompt !== "string") {
    throw new Error("Task prompt must be a string.");
  }

  await fs.mkdir(path.dirname(TASK_PROMPT_PATH), { recursive: true });
  await fs.writeFile(TASK_PROMPT_PATH, prompt, "utf8");
  return {
    prompt,
    path: TASK_PROMPT_PATH
  };
}

module.exports = {
  DEFAULT_TASK_PROMPT,
  TASK_PROMPT_PATH,
  loadTaskPrompt,
  saveTaskPrompt
};

const fs = require("node:fs/promises");
const { parseArgs } = require("node:util");

const USAGE = "Usage: npm start -- [--config FILE] [--task TEXT]";

async function main() {
  const { values } = parseArgs({
    options: {
      mode: { type: "string" },
      config: { type: "string" },
      task: { type: "string" },
      help: { type: "boolean", short: "h" }
    }
  });
  if (values.help) {
    console.log(USAGE);
    return;
  }

  const config = values.config
    ? JSON.parse((await fs.readFile(values.config, "utf8")).replace(/^\uFEFF/, ""))
    : {};
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    throw new Error("The configuration must be a JSON object.");
  }
  const mode = values.mode ?? config.agentMode ?? process.env.AGI_GTA_AGENT_MODE ?? "high_frequency";
  if (mode !== "high_frequency") throw new Error("Only high_frequency mode is supported.");
  const inferenceBackend = config.inferenceBackend ?? process.env.AGI_GTA_INFERENCE_BACKEND ?? "openai_http";
  if (inferenceBackend !== "openai_http") {
    throw new Error("Use an HTTP model API for this release.");
  }

  const { sanitizeConfig } = require("./config");
  const { InputController } = require("./inputController");
  const { loadTaskPrompt } = require("./taskPromptStore");
  const { closeWindowsScreenshotHelper } = require("./windowsScreenshotHelper");
  const configForRun = {
    ...config,
    agentMode: mode,
    ocrManagedService: false,
    taskResultOcrManagedService: false,
    taskResultOcrEnabled: config.taskResultOcrEnabled ?? process.env.AGI_GTA_TASK_RESULT_OCR_ENABLED ?? false
  };
  const effective = sanitizeConfig(configForRun);
  if (!effective.baseUrl || !effective.modelName) {
    throw new Error("Set OPENAI_BASE_URL and OPENAI_MODEL, or provide baseUrl and modelName in the configuration.");
  }
  if (!effective.ocrEnabled) {
    throw new Error("Enable OCR so that the local Harness can verify the pause menu before resuming.");
  }
  require('./policyOcrConfig').validateOcrEndpoints(effective);
  const inputController = new InputController();
  if (!inputController.capabilities().gamepad) {
    throw new Error("The local CLI requires the Windows virtual Xbox controller backend.");
  }

  const { HighFrequencyAgent } = require("./highFrequencyAgent");
  const taskInstruction = values.task ?? await loadTaskPrompt();
  const agent = new HighFrequencyAgent({ inputController, defaultConfig: effective });
  agent.on("log", (message) => console.log(message));
  agent.on("errorEvent", ({ message }) => console.error(message));

  let stopSignal;
  const stop = (signal) => {
    if (stopSignal) return;
    stopSignal = signal;
    agent.stop(signal);
  };
  const interrupt = () => stop("SIGINT");
  const terminate = () => stop("SIGTERM");
  process.on("SIGINT", interrupt);
  process.on("SIGTERM", terminate);

  try {
    await agent.start(configForRun, { taskInstruction });
    await agent.runPromise;
    process.exitCode = agent.lastError ? 1 : stopSignal === "SIGINT" ? 130 : stopSignal === "SIGTERM" ? 143 : 0;
  } finally {
    try {
      if (inputController.ready) await inputController.gamepadReset();
    } finally {
      inputController.close();
      await closeWindowsScreenshotHelper();
      process.off("SIGINT", interrupt);
      process.off("SIGTERM", terminate);
    }
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});

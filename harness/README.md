# ZETA Agent Harness

[English](README.md) · [简体中文](README.zh-CN.md)

Windows game host for asynchronous screenshots/OCR, Policy inference, external Brain updates, task-state management, verified pause/resume and optional standby-service handoff. The shared 200ms implementation lives in [cadence/](../cadence/); model serving lives in [infra/](../infra/).

## Install

Use Node.js 22+, Python 3.11 and ViGEmBus on the Windows game host. From the **repository root**:

```bash
npm ci
python -m pip install vgamepad
python -m pip install -r harness/requirements-ocr.txt
python harness/scripts/ocr_service.py --device cpu --warmup
```

Set `AGI_GTA_POLICY_OCR_URL` to the OCR server you started (including its `/ocr` path). The reference profile uses that endpoint for both Policy OCR and pause verification. OCR must be available to confirm the pause menu before a full Brain request begins.

## Reference configuration

Starting without a configuration file now uses the same settings as [configs/report.example.json](configs/report.example.json): 200ms capture and action windows, an 80ms capture-to-submission offset, 200ms inference allowance, streaming compact responses and 16/8 history. Brain uses its own configured endpoint. Two ordered 100ms button states use two groups of three repeated native chunks. Fixed-cadence duration mismatches fail before startup. Explicit configuration values take precedence over environment variables; both take precedence over defaults.

No Harness service URL or API key is preconfigured, including local OCR addresses. Set these environment variables in the host shell, or use an ignored `*.local.json` configuration file. Supply credentials only when your service requires authentication; never commit them to the repository.

| Variable | Value |
| --- | --- |
| OPENAI_BASE_URL | ZETA Policy gateway, including its `/v1` API prefix |
| OPENAI_MODEL | Model identifier exposed by the Policy gateway |
| OPENAI_API_KEY | Policy credential, if required |
| AGI_GTA_BRAIN_BASE_URL | Independent Brain API endpoint |
| AGI_GTA_BRAIN_MODEL | Brain model identifier |
| AGI_GTA_BRAIN_API_KEY | Brain credential, if required |
| AGI_GTA_POLICY_OCR_URL | Required Policy/pause OCR endpoint, including its `/ocr` path |

If task-result OCR is explicitly enabled, also set `AGI_GTA_TASK_RESULT_OCR_URL` (or `taskResultOcrUrl`). The legacy OCR profile uses `AGI_GTA_OCR_URL` (or `ocrUrl`). Enabled OCR services require an explicit HTTP(S) endpoint.

Run from the repository root:

```bash
npm start -- --task "Follow the current on-screen objective."
```

`npm start` enters this Harness workspace; `--config` paths are relative to `harness/`. The reference run has no response-count cap (`maxStepsUnlimited: true`) and a 40-minute wall-clock budget (`runDurationMs: 2400000`). The budget starts when the inference run starts, includes model waits and manual/game pauses, and uses the normal stop/input-release path. Set `runDurationMs: 0` to disable this budget; set `maxStepsUnlimited: false` and `maxSteps` to restore a response-count limit. Ctrl+C releases input and stops the agent.

Use a ZETA-compatible Policy service with matching action/think protocols, tokenizer, processor and chat template. The bundled template hash must match the serving template. See the [Infra deployment guide](../infra/docs/deployment.md) for runtime requirements and service startup.

## Memory refresh and output budgets

The default settings are:

| Setting | Default | Effect |
| --- | --- | --- |
| `brainScheduler.memoryFallbackIntervalMs` | `10000` | Request a memory/local-reasoning update after 10 seconds of active gameplay without a committed refresh or completed recovery. |
| `brainScheduler.instructionFallbackEnabled` | `false` | Disable the automatic instruction-age trigger; explicit Policy Brain calls still replan the stage. |
| `brainProvider.maxTokens` | `16384` | Brain completion-token limit. |
| `maxTokens` | `null` | No application-level Policy output cap; server/context limits still apply. |
| `maxStepsUnlimited` | `true` | Do not stop after the stored `maxSteps` response count. |
| `runDurationMs` | `2400000` | Stop after 40 minutes of wall-clock time. |

Memory refresh uses a monotonic gameplay clock, including active intervals with no completed action. Pause, task-transition and recovery handling stop that clock. A buffered, failed or superseded Brain request does not reset the refresh obligation; an accepted revision does. The default fallback updates memory and local reasoning under the existing stage instruction.

For count-based operation, set `brainScheduler.memoryFallbackIntervalMs: 0`; `maxActionsWithoutMemoryUpdate` then applies (stored default `20`). The stored `maxActionsWithoutInstructionUpdate: 40` is inactive unless `instructionFallbackEnabled` is explicitly enabled. Override the timer/flag through `AGI_GTA_BRAIN_MEMORY_FALLBACK_INTERVAL_MS` and `AGI_GTA_BRAIN_INSTRUCTION_FALLBACK_ENABLED`, and the run budget through `AGI_GTA_RUN_DURATION_MS`. Existing `AGI_GTA_MAX_TOKENS` and `AGI_GTA_BRAIN_MAX_TOKENS` allow numeric output caps. Explicit JSON values take precedence.

## Optional handoff and background Brain prefill

For a second full Policy service, set `policyHotSwapEnabled` to true and set `AGI_GTA_POLICY_STANDBY_BASE_URL` (and its API key when required). Both services must use the same Policy weights and expose the matching dual-route prepare/readiness contract. Harness activates a new task context only together with the acknowledged native ownership transfer.

`brainPrefillEnabled` is disabled by default. Enable it only for the documented self-hosted Brain transport that honours zero-output prefill and prefix reuse.

An empty think retains the current judgment. A Policy local update supplies new reasoning and asks Brain for memory; the timed fallback asks Brain for memory and reasoning under the current instruction. A full Brain call reviews all three fields after confirming pause. Accepted context and the Policy session take effect together at the configured handoff boundary.

Policy dialogue and execution receipts serve different purposes: a valid generated response may remain in dialogue even when an action expires or executes partially. Brain uses execution receipts and visual evidence to assess what happened. Malformed responses are not admitted to normal Policy history.

For legacy 400ms/stateless operation, explicitly set `hfScheduling: "legacy"`, `hfCaptureIntervalMs: 400`, `hfPackageDurationMs: 400`, `hfSubmissionOffsetMs: 0`, `policyStreaming: false`, `policyTransport: "stateless"` and `policyRendering: null`.

## Tests

```bash
npm run test:harness
```

Run this at the repository root. The tests cover pause handling, OCR, configuration validation and dialogue admission.

# Inference configuration and code entry points

[English](infra.md) · [简体中文](infra.zh-CN.md)

Start with the [deployment guide](deployment.md). Paths below are relative to `infra/`; the shared scheduler is in the root `cadence/` directory.

| Feature | Entry point |
| --- | --- |
| Exact 16/8 history reconstruction, persistent task context and delivered OCR | `serving/dual_pipeline/lumine_dual/protocol.py`, `client.py` |
| Alternating TP4 generation and prefix preparation | `serving/dual_pipeline/lumine_dual/client.py`, `cache_policy.py` |
| Exact token and ordered-media matching for prefix reuse | `serving/lumine_protocol/__init__.py`, `serving/dual_pipeline/lumine_dual/worker.py` |
| Overlapping image upload and prefix preparation | `serving/dual_pipeline/lumine_dual/overlap_upload.py` |
| Compiled execution, decoder and vision-encoder CUDA Graphs | `configs/a.example.json`, `b.example.json`, `runtime/frozen_vllm/vllm/` |
| Shared-memory image caching and asynchronous scheduling | Worker `AsyncLLM`, `mm_processor_cache_type=shm`, `async_scheduling=true` |
| Direct image-token replacement and placeholder forward scan | `serving/dual_pipeline/lumine_dual/fast_image_prompt.py` |
| Materialized token-ID validation and processor threads | `runtime/helpers/fast_mm_check.py`, worker `preprocess_threads=16` |
| Minimap RGB PNG encoding | `serving/dual_pipeline/lumine_dual/lossless_png.py` |
| Session prefix caching and Mamba/attention state | Worker session scope, `mamba_cache_mode=align`, scheduler helpers |
| Independent encoder-cache and compute budgets | vLLM `config/scheduler.py` and worker budget checks |
| Fixed capture/submission grid | `../cadence/src/fixedSubmissionClock.js`, `../cadence/scripts/observation-clock.py` |
| Complete-response dependencies and skipped stale slots | `../cadence/src/generationAdmission.js` |
| SSE state parsing | `../cadence/src/policyActionStream.js` |
| Execution windows, state holding and watchdog | `../cadence/src/streamCadenceExecutor.js`, `../cadence/scripts/gamepad_stream_scheduler.py` |

## Defaults

- One 8×H200 node: GPUs 0–3 run route A and GPUs 4–7 run route B, each with TP4 and BF16.
- Maximum context: 65,536 tokens. Batch/prefill budget: 16,384 tokens. Independent encoder cache: 32,768 embeddings.
- CUDA Graph ceiling: 8,192 tokens, with vision-encoder graphs enabled. Capture sizes and vision budgets are in `configs/a.example.json` and `b.example.json`.
- Session-scoped prefix reuse with exact token/media matching. The gateway explicitly supplies 16/8 history; supply the same window when using the client directly.
- Submission offset: 80ms. Submission grace period: 100ms. The 200ms inference allowance and 200ms execution window are separate settings; see the [scheduler guide](cadence.md).

Generate instance files with `tools/configure_release.py`, then configure model, credential and runtime paths in the instance directory. Harness endpoints, output budgets and run limits are documented in [Harness configuration](../../harness/README.md#reference-configuration).

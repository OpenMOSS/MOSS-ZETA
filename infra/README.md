# ZETA Inference Infrastructure

[English](README.md) · [简体中文](README.zh-CN.md)

ZETA model serving and inference acceleration, integrated with the shared **200ms observation/submission clock, streamed action executor and native scheduler**.

The default deployment uses one 8×H200 node, two independent TP4 routes, BF16 and a 16-round history window retaining eight rounds. Harness can connect a second complete service for context handoff; each service has its own two inference routes.

Run commands from `infra/` (`cd infra` at the repository root). Shared scheduling code lives in the root `cadence/` directory and is also imported by Harness.

## Contents

| Directory | Purpose |
| --- | --- |
| serving/ | Dual-route prefix preparation, overlapping uploads, SSE, task memory/OCR, dialogue history and recovery |
| runtime/frozen_vllm/ | Customized vLLM source with an independent encoder-cache budget |
| runtime/helpers/ | Request scheduling and multimodal validation |
| ../cadence/ | 200ms clocks, submission gates, streamed parsing/execution, Python scheduler and admission adapter |
| configs/, tools/ | Example configurations, runtime overlay, worker launch and source verification |
| docs/infra.md | Code entry points and default configuration |
| docs/deployment.md, docs/cadence.md | Deployment and scheduler integration |
| provenance/ | Source attribution and integrity manifests |

## Deploy

Follow the [deployment guide](docs/deployment.md) to prepare model weights, compiled extensions and credentials, start both workers and the gateway, then connect the [Windows Harness](../harness/README.md). Use the customized runtime with matching CUDA, Torch and Python ABIs.

Generate instance configurations in a new directory:

```bash
python tools/configure_release.py --model-dir /ASSETS/ZETA/model --private-dir /PRIVATE --run-dir /NEW_RUN
```

See [configuration and code entry points](docs/infra.md) for defaults. Observations and action windows use a 200ms period. The first complete control state can execute while generation continues; the next request waits for the completed response. Custom hosts can use the [scheduler interface](docs/cadence.md).

## Checks

Python component checks require Pillow and Jinja2. Use Node.js 22+ as specified by the repository.

```bash
python tools/verify_snapshot.py
python tools/cpu_smoke.py
node --test ../cadence/scripts/*-test.js
python ../cadence/scripts/stream-cadence-test.py
python ../cadence/scripts/stream-handoff-test.py
python ../cadence/scripts/stream-watchdog-notification-test.py
```

These checks cover source integrity, protocols, history trimming, prefix reuse and control scheduling.

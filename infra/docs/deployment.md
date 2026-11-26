# Deployment

[English](deployment.md) · [简体中文](deployment.zh-CN.md)

Run the commands below from the repository's `infra/` directory. Use a new environment and instance directory.

## Requirements

Prepare Python 3.10, CUDA, Torch, Transformers and serving dependencies matching `runtime/observed-packages.txt`, together with the customized vLLM compiled libraries listed in `runtime/binary-manifest.json`.

Provide model weights, a tokenizer and a processor compatible with the ZETA protocol. Match the model architecture, action/think protocol, chat template and configured `model_revision` to the deployed model.

The default deployment uses one 8×H200 node, with four GPUs per worker. The workers check for SM90 H200 hardware. Use the customized vLLM source and matching compiled extensions.

Create two nonempty credential files outside the repository in `/PRIVATE`: `policy.key` authenticates the gateway to the workers, and `gateway.key` authenticates Harness to the gateway. Restrict read access to the service account.

## Environment and configuration

Build the runtime overlay from the matching compiled libraries, then generate instance configurations:

```bash
python tools/build_runtime_overlay.py \
  --binary-root /EXTERNAL_MATCHING_ENV --output /NEW_ENV/zeta-vllm
python tools/configure_release.py \
  --model-dir /ASSETS/ZETA/model --private-dir /PRIVATE --run-dir /NEW_RUN
```

`/EXTERNAL_MATCHING_ENV` must contain a `vllm/` package directory with matching native libraries. The overlay tool verifies their hashes, copies the published Python source and links the external libraries. CUDA, Torch and Python ABIs must match. Both output directories must be new.

The configuration tool creates `a.json`, `b.json`, `gateway.json` and `cadence.json`, binding model, credential, template and runtime-state paths. Defaults are dual TP4, BF16, Graph8192, an independent 32768-entry encoder cache and 16/8 history.

## Start both workers

Run these commands in separate terminals:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/launch_worker.py \
  --config /NEW_RUN/a.json --vllm-runtime /NEW_ENV/zeta-vllm
```

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/launch_worker.py \
  --config /NEW_RUN/b.json --vllm-runtime /NEW_ENV/zeta-vllm
```

The launcher checks TP4/BF16/Graph8192 settings, the source overlay, native libraries and imported vLLM path, then sets the independent encoder-cache budget.

## Start the gateway and connect Harness

After both workers load the model, run in the serving environment:

```bash
export PYTHONPATH="$PWD/serving/dual_pipeline:$PWD/serving:$PWD/runtime/helpers"
python -m lumine_dual.workstation_gateway \
  --config /NEW_RUN/gateway.json --host 127.0.0.1 --port 18806
```

The gateway connects to local workers on ports 18807 and 18808. Request its `/health` endpoint with `Authorization: Bearer <contents of gateway.key>` to check service status.

Make the gateway reachable from the Windows host. For connections across machines, bind to the appropriate network interface and configure access and authentication. Follow the [Harness guide](../../harness/README.md) to set the Policy endpoint, gateway credential, independent Brain endpoint, OCR and input device. See the [cadence guide](cadence.md) for 200ms scheduler integration.

## Component checks

Run `python tools/cpu_smoke.py` to check configuration, protocols, history windows and prefix reuse. Before starting a game task, confirm model responses, OCR, pause handling and gamepad input on the deployed system.

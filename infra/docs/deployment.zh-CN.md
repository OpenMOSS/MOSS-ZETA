# 新实例部署

[English](deployment.md) · [简体中文](deployment.zh-CN.md)

以下命令从本仓库的 `infra/` 目录执行。使用新的环境和实例目录，保留原有模型与运行环境。

## 外置资产

需要匹配 `runtime/observed-packages.txt` 的 Python 3.10、CUDA、Torch、Transformers 和服务依赖，以及定制 vLLM 编译库；原生库身份见 `runtime/binary-manifest.json`。

模型目录必须包含兼容 ZETA 服务协议的权重、tokenizer 和 processor。部署时核对模型架构、动作／think 协议、chat template 和配置中的 `model_revision` 标识。

默认一台 8×H200，每个 worker 使用 4 张卡；worker 保留 SM90 H200 检查。使用定制 vLLM 源码及匹配的编译扩展。

在仓库外的 `/PRIVATE` 目录准备两个非空密钥文件：`policy.key` 用于 gateway 访问两个 worker，`gateway.key` 用于 Harness 访问 gateway。仅向服务账户开放文件读取权限。

## 环境和配置

源码不包含编译扩展，先组合一个新的 overlay，再生成新实例配置：

```bash
python tools/build_runtime_overlay.py \
  --binary-root /EXTERNAL_MATCHING_ENV --output /NEW_ENV/zeta-vllm
python tools/configure_release.py \
  --model-dir /ASSETS/ZETA/model --private-dir /PRIVATE --run-dir /NEW_RUN
```

`/EXTERNAL_MATCHING_ENV` 下应有包含匹配原生库的 `vllm/` 包目录。overlay 工具核对每个原生库哈希，复制已发布 Python 源码并链接外置库。CUDA／Torch／Python ABI 也须匹配；两个输出目录必须尚不存在。

配置工具生成 `a.json`、`b.json`、`gateway.json` 和 `cadence.json`，绑定模型、密钥、模板和运行状态路径。实例默认保留双路 TP4、BF16、Graph8192、独立 encoder cache32768 及 16/8 历史。

## 启动两个 worker

在两个独立终端分别运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/launch_worker.py \
  --config /NEW_RUN/a.json --vllm-runtime /NEW_ENV/zeta-vllm
```

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/launch_worker.py \
  --config /NEW_RUN/b.json --vllm-runtime /NEW_ENV/zeta-vllm
```

launcher 检查 TP4／BF16／Graph8192 配置、源码 overlay、原生库和导入的 vLLM 路径，并设置独立 encoder cache 预算。

## 启动 gateway 并接入 Harness

两个 worker 完成模型加载后，在服务环境运行：

```bash
export PYTHONPATH="$PWD/serving/dual_pipeline:$PWD/serving:$PWD/runtime/helpers"
python -m lumine_dual.workstation_gateway \
  --config /NEW_RUN/gateway.json --host 127.0.0.1 --port 18806
```

gateway 默认访问本机 18807 和 18808 两个 worker。携带 `Authorization: Bearer <gateway.key 文件内容>` 请求 gateway 的 `/health` 可检查服务状态。

Windows Harness 需能访问 gateway；跨机器部署时，将监听地址改为相应网络接口，并配置网络访问及鉴权。按 [Harness 说明](../../harness/README.zh-CN.md) 设置 Policy endpoint、gateway 密钥、独立 Brain endpoint、OCR 和游戏控制设备。200ms 调度接入见 [cadence 文档](cadence.zh-CN.md)。

## 组件检查

运行 `python tools/cpu_smoke.py` 检查配置、协议、历史窗口和前缀复用。接入游戏前，在部署的模型与硬件上确认生成、OCR、暂停和手柄输入可用。

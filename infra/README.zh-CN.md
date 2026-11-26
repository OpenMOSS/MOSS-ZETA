# ZETA Inference Infrastructure

[English](README.md) · [简体中文](README.zh-CN.md)

ZETA 的模型服务与推理加速实现，配合共享的 **200ms 观测／提交节拍器、流式动作执行器和原生调度器**。

默认一台 8×H200、两路独立 TP4、BF16、16 轮历史保留 8 轮。Harness 可另外连接一组完整服务作为状态更新 standby；每组服务内部均有两条推理路由。

本目录命令从 `infra/` 执行（先在仓库根运行 `cd infra`）。共享调度源码位于仓库根的 `cadence/`，Harness直接引用同一实现。

## 内容

| 目录 | 实现 |
| --- | --- |
| serving/ | 双路精确前缀准备、上传重叠、SSE、task memory／OCR协议、生成历史与恢复 |
| runtime/frozen_vllm/ | 定制 vLLM 源码，包括独立 encoder cache 预算 |
| runtime/helpers/ | 请求调度和多模态校验优化 |
| ../cadence/ | 200ms 时钟、提交门、流式 parser／executor、Python 调度器与提交协调器 |
| configs/、tools/ | 示例配置、环境 overlay、worker 启动和源码验证 |
| docs/infra.zh-CN.md | 代码入口与默认配置 |
| docs/deployment.zh-CN.md、docs/cadence.zh-CN.md | 新实例部署顺序与节拍器接入约定 |
| provenance/ | 源码来源与完整性校验清单 |

模型权重、编译扩展和服务密钥需自行准备。

## 部署

按[部署步骤](docs/deployment.zh-CN.md)准备匹配的运行时、启动两个 worker 和 gateway，再连接 [Windows Harness](../harness/README.zh-CN.md)。使用定制运行时，并准备匹配 CUDA／Torch／Python ABI 的编译扩展。

配置工具在新目录生成实例配置：

```bash
python tools/configure_release.py --model-dir /ASSETS/ZETA/model --private-dir /PRIVATE --run-dir /NEW_RUN
```

默认参数与代码入口见[配置说明](docs/infra.zh-CN.md)。200ms 是观测和执行窗口的周期；首个完整控制状态可先执行，下一请求等待上一完整合法响应。自定义宿主接入见[节拍器接口](docs/cadence.zh-CN.md)。

## 检查

Python 组件检查需要 Pillow 和 Jinja2；Node.js 使用仓库要求的 22+。

```bash
python tools/verify_snapshot.py
python tools/cpu_smoke.py
node --test ../cadence/scripts/*-test.js
python ../cadence/scripts/stream-cadence-test.py
python ../cadence/scripts/stream-handoff-test.py
python ../cadence/scripts/stream-watchdog-notification-test.py
```

这些检查覆盖源码完整性、协议、历史裁剪、前缀复用和控制调度。

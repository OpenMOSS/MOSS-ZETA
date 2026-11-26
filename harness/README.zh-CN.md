# ZETA Agent Harness

[English](README.md) · [简体中文](README.zh-CN.md)

运行在 Windows 游戏主机上的 Harness，负责异步截图与 OCR、Policy 推理、外部 Brain 更新、任务状态、暂停／恢复和可选的备用服务交接。共享 200ms 调度实现见 [cadence](../cadence/README.zh-CN.md)，模型服务见 [infra](../infra/README.zh-CN.md)。

## 安装

在 Windows 游戏主机安装 Node.js 22+、Python 3.11 和 ViGEmBus。从**仓库根目录**执行：

```bash
npm ci
python -m pip install vgamepad
python -m pip install -r harness/requirements-ocr.txt
python harness/scripts/ocr_service.py --device cpu --warmup
```

将 `AGI_GTA_POLICY_OCR_URL` 设为启动的 OCR 服务地址，包含 `/ocr` 路径。参考配置使用同一端点提供 Policy OCR 和暂停确认。完整 Brain 请求开始前，Harness 通过 OCR 确认游戏已进入暂停菜单。

## 参考配置

未指定配置文件时，采用 [configs/report.example.json](configs/report.example.json) 中的设置：200ms 捕获与动作窗口、80ms 捕获到提交偏移、200ms 推理预算、流式 compact 响应和 16/8 历史。Brain 使用单独配置的端点。两个有序 100ms 按键状态分别编码为三次重复的原生 chunk。固定节拍下的时长不匹配会在启动前报错。显式配置值优先于环境变量，环境变量优先于默认值。

Harness 的服务 URL 和 API key 均需自行配置，包括本地 OCR 地址。可使用主机环境变量或被 Git 忽略的 `*.local.json` 文件；服务需要鉴权时填写凭据。

| 环境变量 | 含义 |
| --- | --- |
| OPENAI_BASE_URL | ZETA Policy gateway 地址，包含 `/v1` 前缀 |
| OPENAI_MODEL | Policy gateway 提供的模型标识 |
| OPENAI_API_KEY | Policy 服务凭据，按需配置 |
| AGI_GTA_BRAIN_BASE_URL | 独立 Brain API 地址 |
| AGI_GTA_BRAIN_MODEL | Brain 模型标识 |
| AGI_GTA_BRAIN_API_KEY | Brain 服务凭据，按需配置 |
| AGI_GTA_POLICY_OCR_URL | Policy 与暂停确认使用的 OCR 地址，包含 `/ocr` 路径 |

启用任务结果 OCR 时，设置 `AGI_GTA_TASK_RESULT_OCR_URL`（或 `taskResultOcrUrl`）。旧 OCR 模式使用 `AGI_GTA_OCR_URL`（或 `ocrUrl`）。启用的 OCR 服务需要明确的 HTTP(S) 地址。

从仓库根目录启动：

```bash
npm start -- --task "Follow the current on-screen objective."
```

`npm start` 进入 Harness workspace，`--config` 路径相对于 `harness/`。默认运行不限响应次数（`maxStepsUnlimited: true`），墙钟预算为 40 分钟（`runDurationMs: 2400000`），从推理运行开始计时，包含模型等待与手动／游戏暂停；到期后按正常停止流程释放输入。设 `runDurationMs: 0` 可关闭时长限制；设 `maxStepsUnlimited: false` 并配置 `maxSteps` 可按响应次数限制运行。Ctrl+C 停止智能体并释放输入。

Policy 服务应兼容 ZETA 的动作／think 协议，并使用匹配的 tokenizer、processor 和 chat template。随仓库提供的模板哈希应与服务模板一致。运行时要求与启动步骤见 [Infra 部署指南](../infra/docs/deployment.zh-CN.md)。

## 记忆刷新与输出预算

默认设置如下：

| 设置 | 默认值 | 作用 |
| --- | --- | --- |
| `brainScheduler.memoryFallbackIntervalMs` | `10000` | 有效游戏时间累计 10 秒未提交刷新或完成恢复时，请求记忆／局部思考更新。 |
| `brainScheduler.instructionFallbackEnabled` | `false` | 关闭按指令时长触发的自动更新；Policy 显式调用 Brain 仍可重新规划阶段。 |
| `brainProvider.maxTokens` | `16384` | Brain 输出 token 上限。 |
| `maxTokens` | `null` | 应用层不限制 Policy 输出长度，服务与上下文限制仍生效。 |
| `maxStepsUnlimited` | `true` | 不按保存的 `maxSteps` 响应次数停止。 |
| `runDurationMs` | `2400000` | 墙钟运行 40 分钟后停止。 |

记忆刷新使用单调游戏时钟，计入游戏进行中尚无动作完成的时段；暂停、任务切换和恢复期间停止计时。Brain 请求排队、失败或被替代时，刷新计时继续保留；接受的新状态会重置计时。默认兜底更新在当前阶段指令下刷新记忆和局部思考。

需要按动作计数时，设 `brainScheduler.memoryFallbackIntervalMs: 0`，此时使用 `maxActionsWithoutMemoryUpdate`（保存值为 `20`）。`maxActionsWithoutInstructionUpdate: 40` 仅在显式开启 `instructionFallbackEnabled` 时生效。可通过 `AGI_GTA_BRAIN_MEMORY_FALLBACK_INTERVAL_MS`、`AGI_GTA_BRAIN_INSTRUCTION_FALLBACK_ENABLED` 和 `AGI_GTA_RUN_DURATION_MS` 调整计时、开关与运行时长。`AGI_GTA_MAX_TOKENS` 和 `AGI_GTA_BRAIN_MAX_TOKENS` 支持数值输出上限。显式 JSON 配置优先。

## 可选服务交接与后台 Brain prefill

使用第二组完整 Policy 服务时，设 `policyHotSwapEnabled` 为 true，并填写 `AGI_GTA_POLICY_STANDBY_BASE_URL` 及所需 API key。两组服务使用相同 Policy 权重，并提供匹配的双路准备／就绪接口。新任务上下文与已确认的原生控制权交接一起生效。

`brainPrefillEnabled` 默认关闭；支持零输出 prefill 和前缀复用的自托管 Brain 服务可启用此功能。

空 think 沿用当前判断。Policy 局部更新提供新判断并请求 Brain 刷新记忆；定时兜底在当前指令下请求记忆与局部思考；完整 Brain 调用在确认暂停后审视全部三个字段。接受的上下文与 Policy 会话在配置的交接边界一起生效。

Policy 对话与执行回执各自记录生成和执行：合法响应即使执行过期或只执行了一部分，也可留在对话中。Brain 根据执行回执和视觉信息判断实际进展；格式错误的响应会被历史接收检查拒绝。

使用旧 400ms／stateless 模式时，显式设置 `hfScheduling: "legacy"`、`hfCaptureIntervalMs: 400`、`hfPackageDurationMs: 400`、`hfSubmissionOffsetMs: 0`、`policyStreaming: false`、`policyTransport: "stateless"` 和 `policyRendering: null`。

## 检查

```bash
npm run test:harness
```

从仓库根目录运行，覆盖暂停处理、OCR、配置校验和对话接收。

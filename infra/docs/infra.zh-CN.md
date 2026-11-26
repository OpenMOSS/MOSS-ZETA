# 推理服务配置与代码入口

[English](infra.md) · [简体中文](infra.zh-CN.md)

部署步骤见[部署指南](deployment.zh-CN.md)。以下源码路径相对于 `infra/`，调度组件位于仓库根目录 `cadence/`。

| 功能 | 代码入口 |
| --- | --- |
| 16/8 精确历史重建；task context 与交付 OCR 跨视觉裁剪保留 | `serving/dual_pipeline/lumine_dual/protocol.py`、`client.py` |
| 两路 TP4 交替生成与准备，未完成响应不进入下一前缀 | `serving/dual_pipeline/lumine_dual/client.py`、`cache_policy.py` |
| token 和有序 media 身份匹配才允许复用，不匹配重新 prefill | `serving/lumine_protocol/__init__.py`、`serving/dual_pipeline/lumine_dual/worker.py` |
| 上传新图与前缀准备并行 | `serving/dual_pipeline/lumine_dual/overlap_upload.py` |
| 语言侧编译／CUDA Graph 与 vision encoder Graph | `configs/a.example.json`、`b.example.json` 和 `runtime/frozen_vllm/vllm/` |
| SHM 图像缓存和异步调度 | worker 的 `AsyncLLM`、`mm_processor_cache_type=shm`、`async_scheduling=true` |
| 直接图像 token 替换、placeholder forward scan | `serving/dual_pipeline/lumine_dual/fast_image_prompt.py` |
| materialized token ID 校验与 processor 线程 | `runtime/helpers/fast_mm_check.py`、worker 的 `preprocess_threads=16` |
| 小地图 RGB PNG 编码 | `serving/dual_pipeline/lumine_dual/lossless_png.py` |
| session 前缀缓存与 Mamba／attention 状态 | worker 的 session scope、`mamba_cache_mode=align`、两个 scheduler helper |
| encoder cache 与计算预算独立 | vLLM `config/scheduler.py`；worker 核对实际 scheduler 预算 |
| 固定捕获／提交网格 | `../cadence/src/fixedSubmissionClock.js`、`../cadence/scripts/observation-clock.py` |
| 完整响应依赖、错过时槽不补发 | `../cadence/src/generationAdmission.js` |
| SSE 状态解析 | `../cadence/src/policyActionStream.js` |
| 执行窗、状态持有与 watchdog | `../cadence/src/streamCadenceExecutor.js`、`../cadence/scripts/gamepad_stream_scheduler.py` |

## 默认配置

- 一台 8×H200，GPU 0–3 为 A 路、4–7 为 B 路，各 TP4，BF16。
- 最大上下文65,536；batch/prefill预算16,384；encoder cache独立设为32,768。
- CUDA Graph ceiling8,192，vision encoder graph开启；capture sizes和vision budgets见 `configs/a.example.json`、`b.example.json`。
- session-scoped prefix reuse，精确token/media前缀；gateway显式传入16/8。直接使用底层client时也需显式传入该历史窗口。
- 示例采用80ms提交偏移、100ms提交宽限；200ms inference allowance与200ms执行窗独立。具体调度行为见[节拍器接口](cadence.zh-CN.md)。

使用 `tools/configure_release.py` 生成实例配置后，在实例目录配置本机模型、密钥和运行路径。Harness 的服务地址、输出预算和运行时限见 [Harness 配置](../../harness/README.zh-CN.md#参考配置)。

# ZETA 共享调度器

[English](README.md) · [简体中文](README.zh-CN.md)

Harness 与 Infra 共用动作解析器、绝对 200ms 观测／提交时钟、流式执行器、原生调度器、控制权交接和 watchdog。

| 入口 | 作用 |
| --- | --- |
| src/fixedSubmissionClock.js | 独立捕获网格与有时限的提交门 |
| src/policyActionStream.js | 增量控制状态与两个有序 100ms 按键状态的校验 |
| src/streamCadenceExecutor.js | 原生执行窗、执行回执与原子控制权交接 |
| scripts/gamepad_stream_scheduler.py | 绝对执行窗、截尾、状态持有和独立 watchdog |
| src/actionRuntime.js | 基础动作运行时；Harness 通过子类增加控制连续性 |
| src/generationAdmission.js | 供自定义宿主循环使用的提交协调器 |

完整 Harness 使用自身协调循环并导入这些共享组件。原生设备 I/O 和暂停菜单确认由宿主负责。备用 Policy 服务与每组服务内部的两条交替推理路由分别配置。

时序、流式状态和原生输入接口见[接入指南](../infra/docs/cadence.zh-CN.md)。

从仓库根目录运行：

```bash
npm run test:cadence
python cadence/scripts/stream-cadence-test.py
python cadence/scripts/stream-handoff-test.py
python cadence/scripts/stream-watchdog-notification-test.py
```

这些组件测试覆盖时钟、提交门、执行窗、控制权交接和 watchdog。

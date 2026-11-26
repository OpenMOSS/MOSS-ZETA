# 200ms 节拍器与流式控制

[English](cadence.md) · [简体中文](cadence.zh-CN.md)

共享源码位于仓库根目录 [cadence/](../../cadence/)，Harness 与 Infra 使用同一实现。下列源码路径相对于仓库根目录。

## 组件

- `cadence/scripts/observation-clock.py`：独立进程输出200ms捕获和目标提交时槽，落后则跳过旧时槽。
- `cadence/src/fixedSubmissionClock.js`：换算到Node单调时钟并构造SubmissionGate；payload在提交宽限内就绪才可发送，动作起点不随迟到顺延。
- `cadence/src/policyActionStream.js`、`highFrequencyActionParser.js`：SSE中只有四轴加完整按键字段才能执行。六个chunk保留两个有序100ms状态的重复编码约束。
- `cadence/src/streamCadenceExecutor.js`：向已校准native helper发送状态，区分完整响应和执行回执，记录完整／部分／未执行包。
- `cadence/scripts/gamepad_stream_scheduler.py`：原生绝对执行窗、截尾、状态持有、owner/version交接和独立watchdog。

`generationAdmission.js` 为自定义宿主提供“等待完整响应、只保留最新待提交观察”的协调逻辑。使用完整 Harness 时由其自身循环协调；Brain 状态更新、菜单识别和 watchdog 恢复由宿主处理。

## 时间与依赖

捕获网格为 `t0+n*200ms`；目标提交为网格加submission offset；绝对动作起点为目标提交加inference allowance；动作终点恒为原动作起点加200ms。

1. 每拍取新图；截图尚未完成时跳过新捕获回调，不用旧图补发。
2. 上一条响应必须完整生成、通过协议并进入generated dialogue；首个streamed state不释放依赖。
3. `gate.wait(signal)`放行后才能发送payload。同一dialogue只有一个生成在途，过期的未发请求直接跳过。
4. 提前状态等原定起点；迟到状态从到达时开始，尾部到原终点截断；过期数据不执行。
5. 边界保持最后实际状态。默认1000ms watchdog仅由新包首个有效完整状态续期；重复字段、旧epoch、普通token和过期数据不续期。
6. watchdog先释放输入、锁停，再请求暂停。宿主通过截图／OCR 确认实际暂停菜单。

## 接入

```js
const { FixedSubmissionClock, GenerationAdmission, StreamCadenceExecutor,
  fetchPolicyStream } = require('../cadence/src');
```

宿主提供新图捕获、Policy请求构造、完整响应校验、历史接受、Brain屏障、设备I/O和暂停确认。`GenerationAdmission.generate`在gate放行后调用；`validate`必须检查真实完整性、协议与committed状态；`commit`完成历史接受后才放行下一条。`generate`必须尊重AbortSignal，取消后不能写回状态。

流式回调将`state`交给 `executor.streamState(event, metadata)`，其中metadata至少含唯一requestId和gate的scheduledStartPerfMs。完整输出合法后调用`executor.finalize`；异常时`executor.clearQueue`并由宿主停止／暂停。Brain-call状态不执行，交给完整输出后的阶段重规划屏障。

Native input接口如下，回执通过EventEmitter的`event`事件提供：

```text
streamCadenceSupported = true
calibrateGamepadClock() -> { offsetMs, ... }
send(command, payload) -> acknowledgement
```

命令包括`gamepad_stream_arm/state/finalize/cancel`及可选handoff。同一个offset用于FixedSubmissionClock和native window；Node performance.now与Python perf_counter起点不同，必须真实IPC往返校准。

Python StreamScheduler接受`apply(state)`、`release()`、`emit(receipt)`和可选`pause()`。宿主在`update`前验证整组状态（可使用`_complete_state`检查），保留owner/epoch/version约束。真实设备写入和暂停确认留在宿主。

## 检查

按 [cadence 测试说明](../../cadence/README.zh-CN.md) 检查时钟、提交门、流式解析、执行窗和 watchdog；完整暂停／恢复流程见 [Harness 说明](../../harness/README.zh-CN.md)。

# 200ms cadence and streamed control

[English](cadence.md) · [简体中文](cadence.zh-CN.md)

Harness and Infra share the implementation in [cadence/](../../cadence/). Source paths below are relative to the repository root.

## Components

- `cadence/scripts/observation-clock.py`: a separate process emits 200ms capture and target-submission slots, skipping slots that are already past.
- `cadence/src/fixedSubmissionClock.js`: converts to the Node monotonic clock and constructs submission gates. Payloads must be ready within the submission grace period; lateness keeps the original action start.
- `cadence/src/policyActionStream.js` and `highFrequencyActionParser.js`: parse executable SSE states containing all four stick axes and a complete button field. Six chunks encode two ordered 100ms states through repeated groups.
- `cadence/src/streamCadenceExecutor.js`: submits states to the calibrated native helper, separates complete responses from execution receipts and records full, partial and unexecuted packets.
- `cadence/scripts/gamepad_stream_scheduler.py`: manages absolute execution windows, clipped tails, held states, owner/version handoff and an independent watchdog.

`generationAdmission.js` coordinates custom hosts by waiting for each complete response and retaining the latest pending observation. The full Harness has its own coordination loop. The host handles Brain updates, menu recognition and watchdog recovery.

## Timing and dependencies

Capture slots follow `t0+n*200ms`. Target submission adds the submission offset; the absolute action start adds the inference allowance. The end remains the original action start plus 200ms.

1. Capture a fresh image each tick. If capture is still in progress, skip the new callback.
2. Complete and validate the preceding response, then admit it to generated dialogue before the next request. A streamed state alone keeps this dependency pending.
3. Send the payload after `gate.wait(signal)` permits submission. Each dialogue has one generation in flight; skip expired unsent requests.
4. Early states wait for the assigned start. Late states execute from arrival until the original end. Discard expired states.
5. Hold the last actual state at the boundary. The default 1000ms watchdog is renewed by the first valid complete state of a new packet. Duplicate fields, old epochs, ordinary tokens and expired states leave its deadline unchanged.
6. The watchdog releases input, locks execution and requests pause. The host confirms the actual menu through screenshots/OCR.

## Integration

```js
const { FixedSubmissionClock, GenerationAdmission, StreamCadenceExecutor,
  fetchPolicyStream } = require('../cadence/src');
```

The host supplies image capture, Policy requests, complete-response validation, history admission, Brain barriers, device I/O and pause confirmation. `GenerationAdmission.generate` runs after the gate opens; `validate` checks completeness, protocol and committed state; `commit` admits history before allowing the next request. `generate` must honor AbortSignal and stop writing state after cancellation.

Pass streamed `state` events to `executor.streamState(event, metadata)`. Metadata includes a unique requestId and the gate's scheduledStartPerfMs. Call `executor.finalize` after validating the complete output. On failure, call `executor.clearQueue` and stop/pause through the host. Brain-call states enter the stage-replanning barrier after the complete response.

The native input interface emits receipts through EventEmitter's `event` event:

```text
streamCadenceSupported = true
calibrateGamepadClock() -> { offsetMs, ... }
send(command, payload) -> acknowledgement
```

Commands include `gamepad_stream_arm/state/finalize/cancel` and optional handoff. Use the same offset for FixedSubmissionClock and native windows. Calibrate through an actual IPC round trip because Node performance.now and Python perf_counter have different origins.

Python StreamScheduler accepts `apply(state)`, `release()`, `emit(receipt)` and optional `pause()`. Validate complete state groups before `update` (the `_complete_state` helper is available), and retain owner/epoch/version checks. The host handles physical input and pause confirmation.

## Checks

Follow the [cadence test instructions](../../cadence/README.md) for clocks, gates, streamed parsing, execution windows and watchdog behavior. Full pause/resume integration is covered in the [Harness guide](../../harness/README.md).

# Shared ZETA cadence

[English](README.md) · [简体中文](README.zh-CN.md)

One shared implementation of the action parser, absolute 200ms observation/submission clock, streamed executor, native scheduler, owner/version handoff and watchdog. Both Harness and Infra use this directory.

| Entry | Role |
| --- | --- |
| src/fixedSubmissionClock.js | Independent capture grid and bounded submission gates |
| src/policyActionStream.js | Incremental executable states and ordered 100ms button-state validation |
| src/streamCadenceExecutor.js | Native window submission, execution receipts and atomic ownership transfer |
| scripts/gamepad_stream_scheduler.py | Absolute windows, clipped tails, state holding and independent watchdog |
| src/actionRuntime.js | Base action runtime; Harness adds continuity through a subclass |
| src/generationAdmission.js | Portable admission adapter for hosts that provide their own loop |

The complete Harness uses its own coordination loop and imports these shared timing components. Native device I/O and confirmed pause/menu handling remain host responsibilities. A standby Policy service and a service's internal alternating inference routes are distinct layers.

See the [integration guide](../infra/docs/cadence.md) for timing, streaming and native input interfaces.

From the repository root:

```bash
npm run test:cadence
python cadence/scripts/stream-cadence-test.py
python cadence/scripts/stream-handoff-test.py
python cadence/scripts/stream-watchdog-notification-test.py
```

These component tests cover clocks, submission gates, execution windows, ownership transfer and watchdog behavior.

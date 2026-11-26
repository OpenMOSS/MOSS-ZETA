const { outcome } = require("./executionOutcome");
const { EventEmitter } = require("node:events");
const { sleep } = require("../../cadence/src/actionExecutor");
const { ActionRuntime, postprocessPausedControllerState, packageCarryState, neutralControllerState } = require("./actionRuntime");

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function nowIso() {
  return new Date().toISOString();
}

function boundedInteger(value, fallback, min, max) {
  const number = Math.trunc(Number(value ?? fallback));
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, number));
}

async function applyControllerState(input, state) {
  if (typeof input?.gamepadState === "function") {
    await input.gamepadState(state);
    return;
  }

  if (typeof input?.gamepadReset === "function") {
    await input.gamepadReset();
  }
  const left = state?.leftStick || {};
  const right = state?.rightStick || {};
  await input.gamepadStick("left", left.x || 0, left.y || 0);
  await input.gamepadStick("right", right.x || 0, right.y || 0);
  await input.gamepadTrigger("left", state?.leftTrigger || 0);
  await input.gamepadTrigger("right", state?.rightTrigger || 0);
  for (const button of state?.buttons || []) {
    await input.gamepadButtonDown(button);
  }
}

async function executeControllerStates(input, states, options = {}) {
  const durationMs = Math.max(1, Number(options.chunkDurationMs) || 400 / 6);
  const shouldStop = typeof options.shouldStop === "function" ? options.shouldStop : () => false;
  const onChunk = typeof options.onChunk === "function" ? options.onChunk : () => {};

  const started = performance.now();
  for (let index = 0; index < states.length; index += 1) {
    if (shouldStop()) {
      return false;
    }
    onChunk(index, states[index]);
    await applyControllerState(input, states[index]);
    await sleep(Math.max(0, started + (index + 1) * durationMs - performance.now()));
  }

  return true;
}

class HighFrequencyExecutor extends EventEmitter {
  constructor(inputController, options = {}) {
    super();
    this.input = inputController;
    this.actionRuntime = options.actionRuntime || new ActionRuntime({ inputController, recordEvent: options.recordEvent });
    this.configure(options);
    this.running = false;
    this.stopRequested = false;
    this.stopAfterDrain = false;
    this.accepting = true;
    this.activePackage = null;
    this.activeChunkIndex = null;
    this.pendingQueue = [];
    this.preemptRequested = false;
    this.loopPromise = null;
    this.wakeResolver = null;
    this.neutralApplied = false;
    this.idleLeftTriggerHeld = false;
    this.requestIds = new Set();
    this.packageSeq = 0;
    this.executedPackageSeq = 0;
    this.completedPackages = 0;
    this.preemptedPackages = 0;
    this.droppedPackages = 0;
    this.rejectedPackages = 0;
    this.queueHighWatermark = 0;
    this.lastExecution = null;
    this.lastReset = null;
  }

  configure(options = {}) {
    this.chunkDurationMs = Math.max(1, Number(options.chunkDurationMs) || 400 / 6);
    this.maxPendingPackages = boundedInteger(options.maxPendingPackages, 1, 1, 20);
    this.isPackageValid = options.isPackageValid || this.isPackageValid || (() => true);
    this.beforeExecute = options.beforeExecute || this.beforeExecute || (async () => {});
    this.sharedProcessing = options.sharedProcessing ?? this.sharedProcessing ?? false;
    this.carryEnabled = options.carryEnabled ?? this.carryEnabled ?? false;
    this.carryMaxMs = boundedInteger(options.carryMaxMs ?? this.carryMaxMs, 5000, 100, 60000);
    this.packageEndHoldMs = options.packageEndHoldMs ?? this.packageEndHoldMs ?? null;
    this.longHoldThresholdChunks = options.longHoldThresholdChunks ?? this.longHoldThresholdChunks ?? 12;
    this.longHoldReleaseMs = options.longHoldReleaseMs ?? this.longHoldReleaseMs ?? 0;
  }

  getStatus() {
    return {
      running: this.running,
      accepting: this.accepting,
      stopRequested: this.stopRequested,
      stopAfterDrain: this.stopAfterDrain,
      active: Boolean(this.activePackage),
      activePackageSeq: this.activePackage?.seq || null,
      activeChunkIndex: this.activeChunkIndex,
      preemptRequested: this.preemptRequested,
      idleLeftTriggerHeld: this.sharedProcessing ? Boolean(this.actionRuntime.carry?.state.leftTrigger) : this.idleLeftTriggerHeld,
      carry: this.actionRuntime.getStatus(),
      packageEndHoldMs: this.packageEndHoldMs,
      longHoldThresholdChunks: this.longHoldThresholdChunks,
      longHoldReleaseMs: this.longHoldReleaseMs,
      pendingQueueLength: this.pendingQueue.length,
      maxPendingPackages: this.maxPendingPackages,
      queueHighWatermark: this.queueHighWatermark,
      packageSeq: this.packageSeq,
      executedPackageSeq: this.executedPackageSeq,
      completedPackages: this.completedPackages,
      preemptedPackages: this.preemptedPackages,
      droppedPackages: this.droppedPackages,
      rejectedPackages: this.rejectedPackages,
      lastExecution: this.lastExecution,
      lastReset: this.lastReset
    };
  }

  hasCapacity() {
    return this.accepting && this.pendingQueue.length < this.maxPendingPackages;
  }

  start() {
    if (this.running) {
      return;
    }

    this.running = true;
    this.stopRequested = false;
    this.stopAfterDrain = false;
    this.accepting = true;
    this.loopPromise = this.runLoop().finally(() => {
      this.running = false;
      this.stopRequested = false;
      this.stopAfterDrain = false;
      this.accepting = false;
      this.activePackage = null;
      this.activeChunkIndex = null;
      this.preemptRequested = false;
      this.resolveWake();
      this.emitStatus();
    });
    this.emitStatus();
  }

  enqueue(actionPackage) {
    if (actionPackage.requestId && this.requestIds.has(actionPackage.requestId)) return false;
    if (!this.accepting) {
      this.rejectedPackages += 1;
      this.emitStatus();
      return false;
    }

    if (this.pendingQueue.length >= this.maxPendingPackages) {
      this.rejectedPackages += 1;
      this.emitStatus();
      return false;
    }

    if (actionPackage.requestId) this.requestIds.add(actionPackage.requestId);
    this.packageSeq += 1;
    const queued = {
      ...actionPackage,
      seq: this.packageSeq,
      enqueuedAt: nowIso(),
      enqueuedPerfMs: performance.now()
    };
    this.pendingQueue.push(queued);
    this.queueHighWatermark = Math.max(this.queueHighWatermark, this.pendingQueue.length);
    this.neutralApplied = false;
    this.resolveWake();
    const statusEmitStarted = performance.now();
    this.emitStatus();
    queued.enqueueStatusEmitMs = Number(
      (performance.now() - statusEmitStarted).toFixed(3)
    );
    return queued;
  }

  replaceWithLatest(actionPackage) {
    if (actionPackage.requestId && this.requestIds.has(actionPackage.requestId)) return false;
    if (!this.accepting) {
      this.rejectedPackages += 1;
      this.emitStatus();
      return false;
    }

    const dropped = this.pendingQueue.length;
    if (dropped > 0) {
      const cancelled=this.pendingQueue;
      this.pendingQueue = [];
      for(const pkg of cancelled){this.emit('packageDiscarded',{package:pkg,reason:typeof reason==='string'?reason:'replaced'});this.emit('packageOutcome',outcome({package:pkg,reason:typeof reason==='string'?reason:'replaced'},'cancelled'));}
      this.droppedPackages += dropped;
    }

    if (actionPackage.requestId) this.requestIds.add(actionPackage.requestId);
    this.packageSeq += 1;
    const queued = {
      ...actionPackage,
      seq: this.packageSeq,
      enqueuedAt: nowIso(),
      enqueuedPerfMs: performance.now()
    };
    this.pendingQueue.push(queued);
    this.queueHighWatermark = Math.max(this.queueHighWatermark, this.pendingQueue.length);
    if (this.activePackage) {
      this.preemptRequested = true;
      if (typeof this.input?.cancelRealtimeGamepadPackage === "function") {
        void this.input.cancelRealtimeGamepadPackage(this.activePackage.seq).catch(() => {});
      }
    }
    this.neutralApplied = false;
    this.resolveWake();
    if (dropped > 0) {
      this.emit("log", `high-frequency executor dropped ${dropped} stale pending package(s) for latest prediction`);
    }
    this.emitStatus();
    return queued;
  }

  clearQueue(reason = "clear") {
    const dropped = this.pendingQueue.length;
    if (dropped > 0) {
      const cancelled=this.pendingQueue;
      this.pendingQueue = [];
      for(const pkg of cancelled){this.emit('packageDiscarded',{package:pkg,reason:typeof reason==='string'?reason:'replaced'});this.emit('packageOutcome',outcome({package:pkg,reason:typeof reason==='string'?reason:'replaced'},'cancelled'));}
      this.droppedPackages += dropped;
      this.emit("log", `high-frequency executor dropped ${dropped} pending package(s): ${reason}`);
      this.emitStatus();
    }
    return dropped;
  }

  async waitForIdle() {
    if (!this.activePackage && !this.pendingQueue.length) return;
    await new Promise((resolve) => {
      const onStatus = () => {
        if (!this.activePackage && !this.pendingQueue.length) {
          this.off("status", onStatus);
          resolve();
        }
      };
      this.on("status", onStatus);
      onStatus();
    });
  }

  async drainToBoundary(reason = "brain_barrier") {
    this.accepting = false;
    this.clearQueue(reason);
    await this.waitForIdle();
    await this.reset();
    this.neutralApplied = true;
    return this.getStatus();
  }

  resumeAccepting() {
    if (!this.running || this.stopRequested || this.stopAfterDrain) return false;
    this.accepting = true;
    this.resolveWake();
    this.emitStatus();
    return true;
  }

  async stop(options = {}) {
    const drain = Boolean(options.drain);
    this.accepting = false;
    if (drain) {
      this.stopAfterDrain = true;
    } else {
      this.stopRequested = true;
      this.clearQueue(options.reason || "stop");
    }
    this.resolveWake();

    if (this.loopPromise) {
      await this.loopPromise;
    } else {
      await this.reset();
    }

    return this.getStatus();
  }

  resolveWake() {
    if (this.wakeResolver) {
      const resolve = this.wakeResolver;
      this.wakeResolver = null;
      resolve();
    }
  }

  waitForWork() {
    if (this.pendingQueue.length > 0 || this.stopRequested || this.stopAfterDrain) {
      return Promise.resolve();
    }

    return new Promise((resolve) => {
      this.wakeResolver = resolve;
    });
  }

  async runLoop() {
    try {
      while (!this.stopRequested) {
        if (!this.activePackage) {
          this.activePackage = this.pendingQueue.shift() || null;
          if (!this.activePackage) {
            if (this.stopAfterDrain) {
              break;
            }
            await this.resetOnce();
            this.emitStatus();
            await this.waitForWork();
            continue;
          }
        }

        if (!await this.isPackageValid(this.activePackage)) {
          this.droppedPackages += 1;
          this.emit("packageDiscarded", { package: cloneJson(this.activePackage), reason: "invalid_candidate" });
          this.emit("packageOutcome",outcome({package:this.activePackage,reason:"invalid_candidate"},"cancelled"));
          this.activePackage = null;
          this.emitStatus();
          continue;
        }
        await this.beforeExecute(this.activePackage);
        // A hook may await IO; revalidate after it before applying any input.
        if (this.stopRequested || !await this.isPackageValid(this.activePackage)) {
          this.droppedPackages += 1;
          this.emit("packageDiscarded", { package: cloneJson(this.activePackage), reason: "invalid_candidate" });
          this.emit("packageOutcome",outcome({package:this.activePackage,reason:"invalid_candidate"},"cancelled"));
          this.activePackage = null;
          this.emitStatus();
          continue;
        }
        const completed = await this.executePackage(this.activePackage);
        if (!completed) {
          this.preemptedPackages += 1;
          this.emit("log", `high-frequency package ${this.activePackage.seq} preempted by latest prediction`);
        }
        this.activePackage = null;
        this.activeChunkIndex = null;
        this.preemptRequested = false;
        this.emitStatus();

        if (this.stopAfterDrain && this.pendingQueue.length === 0) {
          break;
        }
      }
    } catch(error) {
      if(this.activePackage)this.emit('packageOutcome',outcome({package:this.activePackage,reason:String(error)},'unknown'));
      this.clearQueue('execution_error');throw error;
    } finally {
      this.activePackage = null;
      this.activeChunkIndex = null;
      this.pendingQueue = [];
      this.preemptRequested = false;
      await this.reset();
    }
  }

  async resetOnce() {
    if (this.sharedProcessing && (this.actionRuntime.carry || this.actionRuntime.packageEndHold)) return;
    if (!this.neutralApplied) {
      if (this.idleLeftTriggerHeld) {
        await this.applyState({
          leftStick: { x: 0, y: 0 },
          rightStick: { x: 0, y: 0 },
          leftTrigger: 1,
          rightTrigger: 0,
          buttons: []
        });
      } else {
        await this.reset();
      }
      this.neutralApplied = true;
    }
  }

  async reset() {
    this.idleLeftTriggerHeld = false;
    const started = performance.now();
    const resetStartedAt = nowIso();
    let resetResult = null;
    resetResult = await this.actionRuntime.reset("executor_reset");
    this.lastReset = {
      startedAt: resetStartedAt,
      at: nowIso(),
      durationMs: Number((performance.now() - started).toFixed(3)),
      inputIpcRoundTripMs: Number.isFinite(Number(resetResult?.ipcRoundTripMs))
        ? Number(resetResult.ipcRoundTripMs)
        : null,
      virtualGamepadUpdateMs: Number.isFinite(Number(resetResult?.gamepadUpdateMs))
        ? Number(resetResult.gamepadUpdateMs)
        : null
    };
    this.emit("reset", { reset: cloneJson(this.lastReset) });
  }

  async executePackage(actionPackage) {
    actionPackage.requestedStates = cloneJson(actionPackage.requestedStates || actionPackage.states || []);
    actionPackage.appliedStates = this.sharedProcessing
      ? actionPackage.requestedStates.map(postprocessPausedControllerState)
      : cloneJson(actionPackage.requestedStates);
    actionPackage.states = actionPackage.appliedStates;
    const runtimeOptions = {
      candidateId: actionPackage.candidateId,
      actionSeq: actionPackage.seq,
      carryEnabled: this.carryEnabled && actionPackage.carryEnabled !== false,
      carryMaxMs: this.carryMaxMs,
      ...(this.packageEndHoldMs != null ? { packageEndHoldMs: actionPackage.carryEnabled === false ? 0 : this.packageEndHoldMs,
        longHoldThresholdChunks: this.longHoldThresholdChunks,
        longHoldReleaseMs: actionPackage.carryEnabled === false ? 0 : this.longHoldReleaseMs } : {}),
      shouldStop: () => this.stopRequested || this.preemptRequested
    };
    if (this.sharedProcessing && !await this.actionRuntime.beginPackage(actionPackage.states[0], runtimeOptions)) {
      this.emit("packageOutcome", outcome({package:actionPackage,reason:"begin_package_interrupted"}, "unknown"));
      return false;
    }
    const startedAt = nowIso();
    const startedMs = Date.now();
    const perfStarted = performance.now();
    const continuitySequence = this.actionRuntime.continuity.sequence;
    const traceContinuity = this.sharedProcessing && this.packageEndHoldMs != null && this.longHoldReleaseMs > 0;
    const queueWaitMs = Number(
      Math.max(0, perfStarted - Number(actionPackage.enqueuedPerfMs || perfStarted)).toFixed(3)
    );
    const executablePerfMs = Number(actionPackage.firstExecutablePerfMs);
    const executableToEnqueueMs = Number.isFinite(executablePerfMs)
      ? Number(Math.max(0, Number(actionPackage.enqueuedPerfMs) - executablePerfMs).toFixed(3))
      : null;
    const executableToStartMs = Number.isFinite(executablePerfMs)
      ? Number(Math.max(0, perfStarted - executablePerfMs).toFixed(3))
      : null;
    this.activeChunkIndex = 0;
    this.executedPackageSeq = actionPackage.seq;
    this.emitStatus();

    const chunkTimings = [];
    let realtimeHelperCompletion = null;
    const useRealtimeHelper = (
      // Stateful per-control release must see actual executed chunks. Reuse the
      // existing chunk loop and native helper's gamepad_state IPC; do not expand
      // a whole candidate package into speculative counters/timer programs.
      !(this.sharedProcessing && this.packageEndHoldMs != null && this.longHoldReleaseMs > 0)
      &&
      typeof this.input?.supportsRealtimeGamepadPackage === "function"
      && this.input.supportsRealtimeGamepadPackage()
      && typeof this.input?.gamepadPackageRealtime === "function"
    );
    let completed;
    if (useRealtimeHelper) {
      const realtimeResult = await this.executePackageWithRealtimeHelper(
        actionPackage,
        chunkTimings,
        perfStarted
      );
      completed = realtimeResult.completed;
      realtimeHelperCompletion = realtimeResult.completion;
    } else {
      completed = await this.executePackageInNode(
        actionPackage,
        chunkTimings,
        perfStarted,
        runtimeOptions
      );
    }
    if (this.sharedProcessing) {
      if (completed) await this.actionRuntime.finishPackage(actionPackage.states.at(-1), runtimeOptions);
      else await this.actionRuntime.reset("package_cancelled");
      this.neutralApplied = true;
    } else if (completed) {
      const lastState = actionPackage.states?.at(-1);
      this.idleLeftTriggerHeld = Number(lastState?.leftTrigger || 0) > 0;
      this.neutralApplied = Boolean(
        realtimeHelperCompletion?.idleStateApplied
        && this.pendingQueue.length === 0
      );
    }

    const completedAt = nowIso();
    const durationMs = Date.now() - startedMs;
    this.lastExecution = {
      seq: actionPackage.seq,
      startedAt,
      completedAt,
      durationMs,
      completed,
      timing: {
        chunkCount: useRealtimeHelper
          ? (actionPackage.states || []).length
          : chunkTimings.length,
        queueWaitMs,
        executableToEnqueueMs,
        executableToStartMs,
        enqueueStatusEmitMs: Number.isFinite(Number(actionPackage.enqueueStatusEmitMs))
          ? Number(actionPackage.enqueueStatusEmitMs)
          : null,
        firstStateApplyFromExecutableMs: (
          executableToStartMs != null
          && chunkTimings[0]?.applyStartedOffsetMs != null
        )
          ? Number((
              executableToStartMs
              + chunkTimings[0].applyStartedOffsetMs
            ).toFixed(3))
          : null,
        chunks: chunkTimings,
        ...(traceContinuity ? { controllerTransitions: this.actionRuntime.continuity.traceSince(continuitySequence, perfStarted),
          controllerTransitionTimeBasis: 'device write acknowledgement; post-package holds use controller_delayed_release events' } : {}),
        realtimeHelperCompletion,
        actualDurationMs: Number((performance.now() - perfStarted).toFixed(3)),
        configuredDurationMs:
          (actionPackage.states || []).length *
          (actionPackage.chunkDurationMs || this.chunkDurationMs),
        executionMode: useRealtimeHelper ? "realtime_helper" : "node_chunk_loop",
        preempted: !completed
      }
    };
    const terminal=outcome({package:cloneJson(actionPackage),startedAt,completedAt,timing:this.lastExecution.timing,releaseConfirmed:this.sharedProcessing&&!this.actionRuntime.carry&&!this.actionRuntime.packageEndHold&&!this.actionRuntime.packageEndHoldReleasing&&!this.actionRuntime.packageEndHoldError},completed?'completed':'partial');
    this.emit('packageOutcome',terminal);
    if (completed) {
      this.completedPackages += 1;
      this.emit("packageComplete", {
        package: cloneJson(actionPackage),
        startedAt,
        completedAt,
        durationMs,
        timing: this.lastExecution.timing
      });
    } else {
      this.emit("packagePreempted", {
        package: cloneJson(actionPackage),
        startedAt,
        completedAt,
        durationMs,
        timing: this.lastExecution.timing
      });
    }
    this.emitStatus();
    return completed;
  }

  async executePackageWithRealtimeHelper(
    actionPackage,
    chunkTimings = [],
    packagePerfStarted = performance.now()
  ) {
    const durationMs = actionPackage.chunkDurationMs || this.chunkDurationMs;
    const configuredDurationMs = (
      (actionPackage.states || []).length * durationMs
    );
    const lastState = actionPackage.states?.at(-1) || {};
    await this.input.gamepadPackageRealtime(actionPackage.states, {
      packageSeq: actionPackage.seq,
      chunkDurationMs: durationMs,
      idleHoldLeftTrigger: !this.sharedProcessing && Number(lastState.leftTrigger || 0) > 0,
      ...(this.sharedProcessing ? {
        idleState: this.packageEndHoldMs != null
          ? (this.packageEndHoldMs > 0 && actionPackage.carryEnabled !== false ? lastState : neutralControllerState())
          : this.carryEnabled && actionPackage.carryEnabled !== false
          ? packageCarryState(lastState)
          : neutralControllerState()
      } : {})
    });
    const completionPromise = typeof this.input.waitForRealtimeGamepadPackageCompletion === "function"
      ? this.input.waitForRealtimeGamepadPackageCompletion(
        actionPackage.seq,
        Math.max(1000, configuredDurationMs + 2000)
      )
      : null;
    const scheduleStarted = performance.now();
    const helperStartOffsetMs = scheduleStarted - packagePerfStarted;

    if (completionPromise) {
      let completion = null;
      let completionError = null;
      void completionPromise.then(
        (result) => {
          completion = result;
        },
        (error) => {
          completionError = error;
        }
      );
      while (!completion && !completionError) {
        if (this.stopRequested || this.preemptRequested) {
          if (typeof this.input.cancelRealtimeGamepadPackage === "function") {
            await this.input.cancelRealtimeGamepadPackage(actionPackage.seq);
          }
          let cancelledCompletion=null;
          try { cancelledCompletion=await completionPromise; } catch (_) { /* Unknown if no terminal receipt. */ }
          return { completed: false, completion: cancelledCompletion };
        }
        await sleep(10);
      }
      if (completionError) {
        throw completionError;
      }
      return {
        completed: completion.event === "gamepad_package_completed",
        completion: {
          ...completion,
          helperStartOffsetMs: Number(helperStartOffsetMs.toFixed(3)),
          observedAt: nowIso(),
          notificationLagMs: Number.isFinite(Number(completion.completedAtUnixMs))
            ? Number(Math.max(0, Date.now() - Number(completion.completedAtUnixMs)).toFixed(3))
            : null
        }
      };
    }

    const deadline = scheduleStarted + configuredDurationMs;
    while (true) {
      if (this.stopRequested || this.preemptRequested) {
        if (typeof this.input.cancelRealtimeGamepadPackage === "function") {
          await this.input.cancelRealtimeGamepadPackage(actionPackage.seq);
        }
        return { completed: false, completion: null };
      }
      const remaining = deadline - performance.now();
      if (remaining <= 0) {
        break;
      }
      await sleep(Math.min(remaining, 10));
    }
    return { completed: true, completion: null };
  }

  async executePackageInNode(
    actionPackage,
    chunkTimings = [],
    packagePerfStarted = performance.now(),
    runtimeOptions = {}
  ) {
    const states = Array.isArray(actionPackage.states) ? actionPackage.states : [];
    const durationMs = actionPackage.chunkDurationMs || this.chunkDurationMs;
    const started = performance.now();
  for (let index = 0; index < states.length; index += 1) {
      if (this.stopRequested || this.preemptRequested) {
        return false;
      }
      this.activeChunkIndex = index;
      this.emitStatus();
      const chunkStarted = performance.now();
      const applyStarted = performance.now();
      const receipt = this.sharedProcessing && this.packageEndHoldMs != null
        ? await this.actionRuntime.applyChunk(states[index], { source: { candidateId: actionPackage.candidateId, actionSeq: actionPackage.seq }, shouldStop: runtimeOptions.shouldStop })
        : { state: states[index], result: await this.applyState(states[index]) };
      if (receipt.cancelled) return false;
      const applyResult = receipt.result;
      const applyFinished = performance.now();
      const sleepStarted = performance.now();
      await sleep(Math.max(0, started + (index + 1) * durationMs - performance.now()));
      if (this.sharedProcessing && this.packageEndHoldMs != null) await this.actionRuntime.completeChunk(receipt, runtimeOptions.shouldStop);
      const sleepFinished = performance.now();
      chunkTimings.push({
        index,
        requestedState: cloneJson(states[index]),
        appliedState: cloneJson(receipt.state),
        startedAtOffsetMs: Number(
          (chunkStarted - packagePerfStarted).toFixed(3)
        ),
        applyStartedOffsetMs: Number(
          (applyStarted - packagePerfStarted).toFixed(3)
        ),
        applyMs: Number((applyFinished - applyStarted).toFixed(3)),
        inputIpcRoundTripMs: Number.isFinite(Number(applyResult?.ipcRoundTripMs))
          ? Number(applyResult.ipcRoundTripMs)
          : null,
        inputHelperStateMs: Number.isFinite(Number(applyResult?.helperStateMs))
          ? Number(applyResult.helperStateMs)
          : null,
        virtualGamepadUpdateMs: Number.isFinite(Number(applyResult?.gamepadUpdateMs))
          ? Number(applyResult.gamepadUpdateMs)
          : null,
        sleepMs: Number((sleepFinished - sleepStarted).toFixed(3)),
        actualDurationMs: Number((sleepFinished - chunkStarted).toFixed(3)),
        deadlineJitterMs: Number(
          (
            sleepFinished -
            packagePerfStarted -
            (index + 1) * durationMs
          ).toFixed(3)
        ),
        preempted: Boolean(this.stopRequested || this.preemptRequested)
      });
      if (this.stopRequested || this.preemptRequested) {
        return false;
      }
    }
    return true;
  }

  async applyState(state) {
    if (typeof this.input?.gamepadState === "function") {
      return this.input.gamepadState(state);
    }
    return applyControllerState(this.input, state);
  }

  emitStatus() {
    this.emit("status", this.getStatus());
  }
}

module.exports = {
  HighFrequencyExecutor,
  applyControllerState,
  executeControllerStates
};

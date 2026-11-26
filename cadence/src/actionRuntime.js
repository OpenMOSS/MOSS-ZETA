const { sleep } = require("./actionExecutor");

const DOWN_BUTTON = "dpad_down";
const PAUSED_STICK_MIN_RADIUS = 0.3;

function neutralControllerState() {
  return {
    leftStick: { x: 0, y: 0 },
    rightStick: { x: 0, y: 0 },
    leftTrigger: 0,
    rightTrigger: 0,
    buttons: []
  };
}

function normalizedState(state = {}) {
  return {
    leftStick: {
      x: Number(state.leftStick?.x || 0),
      y: Number(state.leftStick?.y || 0)
    },
    rightStick: {
      x: Number(state.rightStick?.x || 0),
      y: Number(state.rightStick?.y || 0)
    },
    leftTrigger: Number(state.leftTrigger || 0),
    rightTrigger: Number(state.rightTrigger || 0),
    buttons: Array.isArray(state.buttons) ? [...state.buttons] : []
  };
}

function clampPausedStickRadius({ x, y }) {
  const radius = Math.hypot(x, y);
  if (!Number.isFinite(radius)) throw new Error("Paused stick coordinates must be finite.");
  if (radius === 0) return { x: 0, y: 0 };
  if (radius >= PAUSED_STICK_MIN_RADIUS && radius <= 1) return { x, y };
  const scale = Math.max(PAUSED_STICK_MIN_RADIUS, Math.min(1, radius)) / radius;
  return { x: x * scale, y: y * scale };
}

function postprocessPausedControllerState(state) {
  const result = normalizedState(state);
  result.leftStick = clampPausedStickRadius(result.leftStick);
  result.rightStick = clampPausedStickRadius(result.rightStick);
  return result;
}

// Only controls explicitly supported across package boundaries may remain held.
function packageCarryState(lastState = {}) {
  const state = normalizedState(lastState);
  const carry = neutralControllerState();
  carry.leftTrigger = state.leftTrigger > 0 ? state.leftTrigger : 0;
  carry.buttons = state.buttons.includes(DOWN_BUTTON) ? [DOWN_BUTTON] : [];
  return carry;
}

async function applyControllerState(input, state) {
  if (typeof input?.gamepadState === "function") {
    await input.gamepadState(normalizedState(state));
    return;
  }

  if (typeof input?.gamepadReset === "function") {
    await input.gamepadReset();
  }
  const normalized = normalizedState(state);
  await input.gamepadStick("left", normalized.leftStick.x, normalized.leftStick.y);
  await input.gamepadStick("right", normalized.rightStick.x, normalized.rightStick.y);
  await input.gamepadTrigger("left", normalized.leftTrigger);
  await input.gamepadTrigger("right", normalized.rightTrigger);
  for (const button of normalized.buttons) {
    await input.gamepadButtonDown(button);
  }
}

class ActionRuntime {
  constructor(options = {}) {
    this.sessionState = options.sessionState;
    this.inputController = options.inputController;
    this.recordEvent = typeof options.recordEvent === "function"
      ? options.recordEvent
      : async () => {};
    this.carry = null;
    this.suspendedCarry = null;
    this.pendingTimer = null;
    this.carryTimer = null;
    this.carryGeneration = 0;
  }

  recordExecutedAction(action) {
    if (!action?.candidateId) {
      throw new Error("ExecutedAction requires a PolicyActionCandidate id.");
    }
    return this.sessionState.addExecutedAction(action);
  }

  usesRemoteRuntime() {
    return this.inputController?.supportsControllerRuntime?.() === true;
  }

  async remoteRuntime(operation, args = {}) {
    const response = await this.inputController.controllerRuntime(operation, args);
    this.remoteStatus = response.runtime;
    return response;
  }

  getStatus() {
    if (this.usesRemoteRuntime()) return this.remoteStatus || { carryActive: false, suspendedControls: [], carry: null };
    return {
      carryActive: Boolean(this.carry),
      suspendedControls: this.suspendedCarry?.controls || [],
      carry: this.carry ? {
        controls: [...this.carry.controls],
        startedAt: this.carry.startedAt,
        expiresAt: this.carry.expiresAt,
        sourceCandidateId: this.carry.sourceCandidateId,
        sourceActionSeq: this.carry.sourceActionSeq
      } : null
    };
  }

  detachCarry() {
    const carry = this.carry;
    this.carry = null;
    if (this.carryTimer) {
      clearTimeout(this.carryTimer);
      this.carryTimer = null;
    }
    this.carryGeneration += 1;
    return carry;
  }

  async reset(reason = "reset") {
    if (this.usesRemoteRuntime()) { return this.remoteRuntime("reset", { reason }); }
    const carry = this.detachCarry() || this.suspendedCarry;
    this.suspendedCarry = null;
    clearTimeout(this.pendingTimer);
    this.pendingTimer = null;
    if (typeof this.inputController?.gamepadReset === "function") {
      await this.inputController.gamepadReset();
    }
    if (carry) {
      await this.recordEvent("controller_hold_released", {
        controls: carry.controls,
        reason,
        sourceCandidateId: carry.sourceCandidateId,
        sourceActionSeq: carry.sourceActionSeq
      });
    }
  }

  async releaseCarry(reason = "release") {
    if (this.usesRemoteRuntime()) return Boolean((await this.remoteRuntime("releaseCarry", { reason })).value);
    if (!this.carry) return false;
    await this.reset(reason);
    return true;
  }

  intersection(carry, firstState) {
    const state = neutralControllerState();
    if (carry?.controls.includes("LT") && Number(firstState?.leftTrigger) > 0) {
      state.leftTrigger = Math.min(carry.state.leftTrigger, Number(firstState.leftTrigger));
    }
    if (carry?.controls.includes("down") && firstState?.buttons?.includes(DOWN_BUTTON)) {
      state.buttons = [DOWN_BUTTON];
    }
    return state;
  }

  controlsFor(state) {
    return [...(state.leftTrigger > 0 ? ["LT"] : []), ...(state.buttons.includes(DOWN_BUTTON) ? ["down"] : [])];
  }

  async preparePauseCapture() {
    if (this.usesRemoteRuntime()) { return this.remoteRuntime("preparePauseCapture"); }
    const generation = this.carryGeneration;
    await applyControllerState(this.inputController, this.carry?.state || neutralControllerState());
    if (generation !== this.carryGeneration) await this.inputController?.gamepadReset?.();
  }

  async suspendCarry(reason = "pause_harness") {
    if (this.usesRemoteRuntime()) { return this.remoteRuntime("suspendCarry", { reason }); }
    const carry = this.detachCarry();
    clearTimeout(this.pendingTimer);
    this.pendingTimer = null;
    if (carry) this.suspendedCarry = carry;
    await this.inputController?.gamepadReset?.();
    if (carry) await this.recordEvent("controller_hold_suspended", {
      reason, controls: carry.controls, sourceCandidateId: carry.sourceCandidateId
    });
  }

  resumePendingTimeout() {
    if (this.usesRemoteRuntime()) {
      this.resumePendingPromise = this.remoteRuntime("resumePendingTimeout");
      void this.resumePendingPromise.catch(error => this.recordEvent("controller_runtime_error", { operation: "resumePendingTimeout", error: error.message }));
      return;
    }
    if (!this.suspendedCarry) return;
    clearTimeout(this.pendingTimer);
    const carry = this.suspendedCarry;
    this.pendingTimer = setTimeout(() => {
      if (this.suspendedCarry !== carry) return;
      this.suspendedCarry = null;
      this.pendingTimer = null;
      void this.recordEvent("controller_hold_timeout", {
        controls: carry.controls, maxMs: carry.maxMs, suspended: true,
        sourceCandidateId: carry.sourceCandidateId
      }).catch(() => {});
    }, carry.maxMs);
    this.pendingTimer.unref?.();
  }

  async selectNextCarry(firstState) {
    if (this.usesRemoteRuntime()) { return this.remoteRuntime("selectNextCarry", { firstState }); }
    const carry = this.carry || this.suspendedCarry;
    if (!carry) return;
    const state = this.intersection(carry, firstState);
    const controls = this.controlsFor(state);
    const released = carry.controls.filter((control) => !controls.includes(control));
    if (this.carry && released.length) {
      const generation = this.carryGeneration;
      await applyControllerState(this.inputController, state);
      if (generation !== this.carryGeneration) {
        await this.inputController?.gamepadReset?.();
        return;
      }
    }
    carry.controls = controls;
    carry.state = state;
    if (released.length) await this.recordEvent("controller_hold_released", {
      reason: "next_package_intersection", controls: released,
      sourceCandidateId: carry.sourceCandidateId, sourceActionSeq: carry.sourceActionSeq
    });
    if (!controls.length) {
      this.detachCarry();
      this.suspendedCarry = null;
      clearTimeout(this.pendingTimer);
      this.pendingTimer = null;
    }
  }

  async restoreCarry(firstState, windowMs, shouldStop = () => false) {
    if (this.usesRemoteRuntime()) {
      if (shouldStop()) return false;
      const response = await this.remoteRuntime("restoreCarry", { firstState, windowMs });
      if (shouldStop()) { await this.reset("stop_during_resume"); return false; }
      return Boolean(response.value);
    }
    const carry = this.suspendedCarry;
    if (!carry || !firstState || shouldStop()) return false;
    const state = this.intersection(carry, firstState);
    this.suspendedCarry = null;
    clearTimeout(this.pendingTimer);
    this.pendingTimer = null;
    if (!this.controlsFor(state).length) return false;
    await this.startCarry(state, Math.max(carry.maxMs, windowMs + 1000), {
      candidateId: carry.sourceCandidateId, actionSeq: carry.sourceActionSeq,
      source: "resume_intersection"
    });
    if (shouldStop()) { await this.reset("stop_during_resume"); return false; }
    await this.recordEvent("controller_hold_restored", {
      controls: this.controlsFor(state), sourceCandidateId: carry.sourceCandidateId, windowMs
    });
    return Boolean(this.carry);
  }

  async bridgeIntoNextPackage(carry, firstState, shouldStop, context) {
    if (!carry) return !shouldStop();
    if (shouldStop()) { await this.reset("stop_before_bridge"); return false; }
    const state = this.intersection(carry, firstState);
    await applyControllerState(this.inputController, state);
    await this.recordEvent("controller_hold_bridged", {
      controls: this.controlsFor(state), overlapMs: 0,
      sourceCandidateId: carry.sourceCandidateId, targetCandidateId: context.candidateId
    });
    if (shouldStop()) { await this.reset("stop_during_bridge"); return false; }
    return true;
  }

  async startCarry(lastState, maxMs, context) {
    const state = normalizedState(lastState);
    const controls = [];
    if (state.leftTrigger > 0) controls.push("LT");
    if (state.buttons.includes(DOWN_BUTTON)) controls.push("down");
    if (controls.length === 0) {
      await this.reset("package_complete");
      return;
    }

    const carryState = packageCarryState(state);
    const generation = ++this.carryGeneration;
    await applyControllerState(this.inputController, carryState);
    if (generation !== this.carryGeneration) {
      await this.inputController?.gamepadReset?.();
      return;
    }

    const startedAtMs = Date.now();
    this.carry = {
      controls,
      maxMs,
      state: carryState,
      startedAt: new Date(startedAtMs).toISOString(),
      expiresAt: new Date(startedAtMs + maxMs).toISOString(),
      sourceCandidateId: context.candidateId,
      sourceActionSeq: context.actionSeq
    };
    this.carryTimer = setTimeout(() => {
      if (!this.carry || this.carryGeneration !== generation) return;
      const expired = this.detachCarry();
      void (async () => {
        if (typeof this.inputController?.gamepadReset === "function") {
          await this.inputController.gamepadReset();
        }
        await this.recordEvent("controller_hold_timeout", {
          controls: expired?.controls || controls,
          maxMs,
          sourceCandidateId: expired?.sourceCandidateId || context.candidateId,
          sourceActionSeq: expired?.sourceActionSeq || context.actionSeq
        });
      })().catch(() => {});
    }, maxMs);
    this.carryTimer.unref?.();
    await this.recordEvent("controller_hold_started", {
      controls,
      maxMs,
      source: context.source || "final_package_chunk",
      sourceCandidateId: context.candidateId,
      sourceActionSeq: context.actionSeq
    });
  }

  async beginPackage(firstState, options = {}) {
    const shouldStop = options.shouldStop || (() => false);
    await this.selectNextCarry(firstState);
    const previousCarry = this.detachCarry() || this.suspendedCarry;
    this.suspendedCarry = null;
    clearTimeout(this.pendingTimer);
    this.pendingTimer = null;
    return this.bridgeIntoNextPackage(previousCarry, firstState, shouldStop, options);
  }

  async finishPackage(lastState, options = {}) {
    if (options.carryEnabled === true && lastState && !options.shouldStop?.()) {
      await this.startCarry(lastState, Math.max(100, Math.trunc(Number(options.carryMaxMs) || 5000)), options);
    } else {
      await this.reset("package_complete");
    }
  }

  async executePausedPackage(states, options = {}) {
    const controllerStates = Array.isArray(states) ? states.map(postprocessPausedControllerState) : [];
    const chunkDurationMs = Math.max(1, Number(options.chunkDurationMs) || 200 / 6);
    const shouldStop = typeof options.shouldStop === "function" ? options.shouldStop : () => false;
    const carryEnabled = options.carryEnabled === true;
    const carryMaxMs = Math.max(100, Math.trunc(Number(options.carryMaxMs) || 5000));
    const context = {
      candidateId: options.candidateId || null,
      actionSeq: Number(options.actionSeq || this.sessionState?.timeline?.actionSeq + 1 || 0) || null
    };
    if (this.usesRemoteRuntime()) {
      // Resume issues this RPC without awaiting it. Do not race it against a
      // whole action on the host's single-owner executor.
      await this.resumePendingPromise;
      if (shouldStop()) { await this.reset("stop_before_package"); return false; }
      if (typeof this.inputController.executePausedPackage !== "function") {
        throw new Error("Remote runtime requires whole-package paused execution.");
      }
      let response;
      try {
        response = await this.inputController.executePausedPackage(controllerStates, {
          ...context, chunkDurationMs, carryEnabled, carryMaxMs
        });
      } catch (error) {
        const expected = shouldStop() && error.code === 'native_host_error'
          && /^(?:Page\.evaluate: )?Error: Action cancelled(?:\r?\n|$)/.test(error.message || '');
        return this.finishRemotePackage(context, { error, expected, stopRequested: shouldStop() });
      }
      this.remoteStatus = response.runtime;
      if (shouldStop()) return this.finishRemotePackage(context, { response, expected: true, stopRequested: true });
      return true;
    }
    if (!await this.beginPackage(controllerStates[0], { ...context, shouldStop })) return false;

    const packageStarted = performance.now();
    for (const [index, state] of controllerStates.entries()) {
      if (shouldStop()) {
        await this.reset("stop_during_package");
        return false;
      }
      await applyControllerState(this.inputController, state);
      await sleep(Math.max(0, packageStarted + (index + 1) * chunkDurationMs - performance.now()));
    }

    await this.finishPackage(controllerStates.at(-1), {
      ...context, carryEnabled, carryMaxMs, shouldStop
    });
    return !shouldStop();
  }

  async finishRemotePackage(context, { error = null, response = null, expected, stopRequested }) {
    const describe = value => value ? { code: value.code || null, message: value.message || String(value) } : null;
    let resetResponse, resetError, evidenceError;
    try { resetResponse = await this.reset(error ? 'remote_package_failed' : 'stop_during_package'); }
    catch (failure) { resetError = failure; }
    try {
      await this.recordEvent('remote_package_terminal', {
        ...context, status: response ? 'completed' : 'unknown', stopRequested,
        // Unknown execution has no invented sub-blocks. A successful in-flight
        // response remains evidence even when stop prevents another Agent step.
        ...(response ? { response } : { actualSegments: [] }),
        error: describe(error), expectedCancellation: Boolean(error && expected),
        cleanup: { response: resetResponse || null, error: describe(resetError) }
      });
    } catch (failure) { evidenceError = failure; }
    if ((error && !expected) || resetError || evidenceError) {
      const primary = error || resetError || evidenceError;
      primary.cleanupErrors = [...(primary.cleanupErrors || []),
        ...(resetError && resetError !== primary ? [{ phase: 'remote_reset', ...describe(resetError) }] : []),
        ...(evidenceError && evidenceError !== primary ? [{ phase: 'evidence', ...describe(evidenceError) }] : [])];
      throw primary;
    }
    return false;
  }
}

module.exports = {
  ActionRuntime,
  PAUSED_STICK_MIN_RADIUS,
  clampPausedStickRadius,
  postprocessPausedControllerState,
  applyControllerState,
  neutralControllerState,
  packageCarryState
};

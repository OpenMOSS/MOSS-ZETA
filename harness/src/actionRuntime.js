// Harness-specific continuity adapts the shared scheduling runtime.
// The base implementation in cadence/ remains bound to its original source hash.
const shared = require('../../cadence/src/actionRuntime');
const { ControllerContinuity } = require('./controllerContinuity');
const { neutralControllerState } = shared;
const DOWN_BUTTON = 'dpad_down';
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

async function applyControllerState(input, state) {
  return typeof input?.gamepadState === 'function'
    ? input.gamepadState(normalizedState(state)) : shared.applyControllerState(input, state);
}

class ActionRuntime extends shared.ActionRuntime {
  constructor(options = {}) {
    super(options);
    this.continuity = new ControllerContinuity({
      write: state => applyControllerState(this.inputController, state),
      reset: () => this.inputController?.gamepadReset?.(), recordEvent: this.recordEvent,
      ...(options.now ? { now: options.now } : {})
    });
  }
  getStatus() {
    const status = super.getStatus();
    return this.usesRemoteRuntime() ? status : { ...status, ...this.continuity.status() };
  }

  async reset(reason = "reset") {
    if (this.usesRemoteRuntime()) { return this.remoteRuntime("reset", { reason }); }
    const carry = this.detachCarry() || this.suspendedCarry;
    this.suspendedCarry = null;
    clearTimeout(this.pendingTimer);
    this.pendingTimer = null;
    const result = await this.continuity.reset(reason);
    if (carry) {
      await this.recordEvent("controller_hold_released", {
        controls: carry.controls,
        reason,
        sourceCandidateId: carry.sourceCandidateId,
        sourceActionSeq: carry.sourceActionSeq
      });
    }
    return result;
  }

  get packageEndHold() { return this.continuity.fullHold; }

  get packageEndHoldTimer() { return this.continuity.timer; }

  get packageEndHoldRelease() { return this.continuity.tail; }

  get packageEndHoldReleasing() { return this.continuity.releasing; }

  get packageEndHoldError() { return this.continuity.error; }

  async endPackageEndHold(reason) {
    await this.continuity.clear(reason);
  }

  async startPackageEndHold(lastState, maxMs, context) {
    this.detachCarry();
    this.suspendedCarry = null;
    clearTimeout(this.pendingTimer); this.pendingTimer = null;
    if (context.shouldStop?.()) return this.reset("stop_at_package_end");
    this.continuity.enabled = true;
    await this.continuity.finishPackage(lastState, maxMs, context);
  }

  async applyChunk(state, options = {}) {
    if (this.continuity.enabled) return this.continuity.applyChunk(normalizedState(state), options);
    return { state: normalizedState(state), result: await applyControllerState(this.inputController, state) };
  }

  async completeChunk(receipt, shouldStop) {
    if (this.continuity.enabled) await this.continuity.completeChunk(receipt, shouldStop);
  }

  async releaseCarry(reason = "release") {
    if (this.usesRemoteRuntime()) return Boolean((await this.remoteRuntime("releaseCarry", { reason })).value);
    if (!this.carry && !this.continuity.enabled && !this.packageEndHoldReleasing) return false;
    await this.reset(reason);
    return true;
  }

  async suspendCarry(reason = "pause_harness") {
    if (this.usesRemoteRuntime()) { return this.remoteRuntime("suspendCarry", { reason }); }
    const carry = this.detachCarry();
    clearTimeout(this.pendingTimer);
    this.pendingTimer = null;
    if (carry) this.suspendedCarry = carry;
    await this.continuity.reset(reason);
    if (carry) await this.recordEvent("controller_hold_suspended", {
      reason, controls: carry.controls, sourceCandidateId: carry.sourceCandidateId
    });
  }

  async beginPackage(firstState, options = {}) {
    const shouldStop = options.shouldStop || (() => false);
    await this.continuity.tail;
    const useContinuity = options.packageEndHoldMs != null || this.continuity.enabled;
    // With long-release disabled the bulk native helper remains available:
    // fence its previous package-end timer before it starts the new packet.
    if (useContinuity && (options.longHoldReleaseMs ?? this.continuity.releaseMs) === 0) await this.continuity.clear('next_package');
    if (useContinuity) {
      this.continuity.enabled = true;
      this.continuity.configure({ thresholdChunks: options.longHoldThresholdChunks ?? 12, releaseMs: options.longHoldReleaseMs ?? 0 });
    }
    await this.selectNextCarry(firstState);
    const previousCarry = this.detachCarry() || this.suspendedCarry;
    this.suspendedCarry = null;
    clearTimeout(this.pendingTimer);
    this.pendingTimer = null;
    return this.bridgeIntoNextPackage(previousCarry, firstState, shouldStop, options);
  }

  async finishPackage(lastState, options = {}) {
    if (options.packageEndHoldMs != null) {
      const ms = Number(options.packageEndHoldMs);
      if (!Number.isFinite(ms) || ms < 0 || ms > 60000) throw Error("Invalid packageEndHoldMs");
      await this.startPackageEndHold(lastState, ms, options);
      return;
    }
    if (options.carryEnabled === true && lastState && !options.shouldStop?.()) {
      await this.startCarry(lastState, Math.max(100, Math.trunc(Number(options.carryMaxMs) || 5000)), options);
    } else {
      await this.reset("package_complete");
    }
  }

  async executeRecoveryPackage(states, options = {}) {
    return super.executePausedPackage(states, options);
  }
}
module.exports = { ...shared, ActionRuntime, applyControllerState };

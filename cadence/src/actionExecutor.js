function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const ACTION_CAPABILITIES = {
  mouse_move: ["mouseMove", "mouse movement"],
  mouse_down: ["mouseButtons", "mouse button input"],
  mouse_up: ["mouseButtons", "mouse button input"],
  key_down: ["keyboard", "keyboard input"],
  key_up: ["keyboard", "keyboard input"],
  wheel: ["wheel", "mouse wheel input"],
  gamepad_stick: ["gamepad", "gamepad input"],
  gamepad_trigger: ["gamepad", "gamepad input"],
  gamepad_button_down: ["gamepad", "gamepad input"],
  gamepad_button_up: ["gamepad", "gamepad input"],
  gamepad_reset: ["gamepad", "gamepad input"]
};

function backendName(input) {
  if (input?.backend?.name) {
    return input.backend.name;
  }

  if (typeof input?.selectedBackend === "function") {
    try {
      return input.selectedBackend()?.name || "unknown";
    } catch (error) {
      return "unknown";
    }
  }

  return "unknown";
}

class ActionExecutor {
  constructor(inputController) {
    this.input = inputController;
  }

  async execute(action) {
    this.assertActionSupported(action);

    switch (action.type) {
      case "mouse_move":
        await this.executeMouseMove(action);
        return;
      case "mouse_down":
        await this.input.mouseDown(action.button);
        return;
      case "mouse_up":
        await this.input.mouseUp(action.button);
        return;
      case "key_down":
        await this.input.keyDown(action.key);
        return;
      case "key_up":
        await this.input.keyUp(action.key);
        return;
      case "wait":
        await sleep(action.ms);
        return;
      case "wheel":
        await this.input.wheel(action.amount);
        return;
      case "gamepad_stick":
        await this.executeGamepadStick(action);
        return;
      case "gamepad_trigger":
        await this.executeGamepadTrigger(action);
        return;
      case "gamepad_button_down":
        await this.input.gamepadButtonDown(action.button);
        return;
      case "gamepad_button_up":
        await this.input.gamepadButtonUp(action.button);
        return;
      case "gamepad_reset":
        await this.input.gamepadReset();
        return;
      default:
        throw new Error(`Cannot execute unsupported action type: ${action.type}`);
    }
  }

  assertActionSupported(action) {
    const requirement = ACTION_CAPABILITIES[action?.type];
    if (!requirement || typeof this.input?.capabilities !== "function") {
      return;
    }

    const [capability, label] = requirement;
    const capabilities = this.input.capabilities();
    if (capabilities?.[capability] === false) {
      throw new Error(`${label} is unavailable on input backend ${backendName(this.input)}.`);
    }
  }

  async executeMouseMove(action) {
    const dx = Number(action.dx) || 0;
    const dy = Number(action.dy) || 0;
    const durationMs = Number(action.durationMs) || 0;

    if (durationMs <= 0) {
      await this.input.mouseMove(dx, dy);
      return;
    }

    const steps = Math.max(1, Math.min(60, Math.ceil(durationMs / 16)));
    let sentX = 0;
    let sentY = 0;

    for (let step = 1; step <= steps; step += 1) {
      const targetX = Math.round((dx * step) / steps);
      const targetY = Math.round((dy * step) / steps);
      const stepX = targetX - sentX;
      const stepY = targetY - sentY;

      sentX = targetX;
      sentY = targetY;

      if (stepX !== 0 || stepY !== 0) {
        await this.input.mouseMove(stepX, stepY);
      }

      if (step < steps) {
        await sleep(Math.max(1, Math.round(durationMs / steps)));
      }
    }
  }

  async executeGamepadStick(action) {
    const stick = action.stick || "left";
    const x = Number(action.x) || 0;
    const y = Number(action.y) || 0;
    const durationMs = Number(action.durationMs) || 0;

    await this.input.gamepadStick(stick, x, y);
    if (durationMs > 0) {
      await sleep(durationMs);
      await this.input.gamepadStick(stick, 0, 0);
    }
  }

  async executeGamepadTrigger(action) {
    const trigger = action.trigger || "right";
    const value = Number(action.value) || 0;
    const durationMs = Number(action.durationMs) || 0;

    await this.input.gamepadTrigger(trigger, value);
    if (durationMs > 0) {
      await sleep(durationMs);
      await this.input.gamepadTrigger(trigger, 0);
    }
  }
}

module.exports = {
  ActionExecutor,
  sleep
};

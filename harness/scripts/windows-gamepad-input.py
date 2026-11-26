import ctypes
import json
import sys
import threading
import time
from pathlib import Path

# Shared 200ms scheduler; one implementation for Harness and Infra.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cadence/scripts"))
from gamepad_cadence_scheduler import CadenceScheduler

OUTPUT_LOCK = threading.Lock()

KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_long),
        ("y", ctypes.c_long),
    ]


def write_json(payload):
    with OUTPUT_LOCK:
        print(json.dumps(payload, separators=(",", ":")), flush=True)


def clamp(value, minimum, maximum):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(minimum, min(maximum, number))


def win32_bridge():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
    user32.GetCursorPos.restype = ctypes.c_bool
    user32.MapVirtualKeyW.argtypes = [ctypes.c_uint, ctypes.c_uint]
    user32.MapVirtualKeyW.restype = ctypes.c_uint
    user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint, ctypes.c_void_p]
    user32.mouse_event.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return user32


def load_vgamepad():
    try:
        import vgamepad as vg
    except Exception as error:
        write_json({
            "event": "error",
            "error": (
                "Python package vgamepad is required for windows-vgamepad-x360. "
                "Install vgamepad and the ViGEmBus driver, then restart the server. "
                f"Import failed: {error}"
            ),
        })
        sys.exit(1)

    try:
        gamepad = vg.VX360Gamepad()
    except Exception as error:
        write_json({
            "event": "error",
            "error": (
                "Could not create a virtual Xbox 360 controller. "
                "Install or repair the ViGEmBus driver, then restart the server. "
                f"vgamepad error: {error}"
            ),
        })
        sys.exit(1)

    buttons = {
        "a": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
        "b": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
        "x": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
        "y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        "lb": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
        "rb": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
        "back": vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
        "start": vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
        "guide": vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE,
        "left_thumb": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
        "right_thumb": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
        "dpad_up": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
        "dpad_down": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
        "dpad_left": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
        "dpad_right": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
    }
    return gamepad, buttons


USER32 = win32_bridge()
GAMEPAD, BUTTONS = load_vgamepad()
GAMEPAD_LOCK = threading.Lock()
REALTIME_CONDITION = threading.Condition()
REALTIME_PENDING = None
REALTIME_ACTIVE = None
REALTIME_CANCELLED = set()
REALTIME_STOP = False


def apply_gamepad_state_locked(state):
    return gamepad_state(state)


def realtime_scheduler():
    global REALTIME_PENDING, REALTIME_ACTIVE
    while True:
        with REALTIME_CONDITION:
            while REALTIME_PENDING is None and not REALTIME_STOP:
                REALTIME_CONDITION.wait()
            if REALTIME_STOP:
                return
            package = REALTIME_PENDING
            REALTIME_PENDING = None
            REALTIME_ACTIVE = package

        package_seq = package.get("packageSeq")
        states = package.get("states") or []
        duration_s = max(0.001, float(package.get("chunkDurationMs") or 33) / 1000.0)
        started = time.perf_counter()
        cancelled = False
        has_successor = False
        first_apply_ms = None
        last_apply_ms = None
        actual_events = []
        for index, state in enumerate(states):
            with REALTIME_CONDITION:
                if package_seq in REALTIME_CANCELLED:
                    REALTIME_CANCELLED.discard(package_seq)
                    cancelled = True
                    break
                if REALTIME_PENDING is not None:
                    cancelled = True
                    break

            target = started + index * duration_s
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

            with REALTIME_CONDITION:
                if package_seq in REALTIME_CANCELLED or REALTIME_PENDING is not None:
                    if package_seq in REALTIME_CANCELLED:
                        REALTIME_CANCELLED.discard(package_seq)
                    cancelled = True
                    break
            apply_gamepad_state_locked(state)
            applied_ms = (time.perf_counter() - started) * 1000.0
            if first_apply_ms is None:
                first_apply_ms = applied_ms
            last_apply_ms = applied_ms
            actual_events.append({"actualMs": applied_ms, "state": state})

        if not cancelled:
            deadline = started + len(states) * duration_s
            while True:
                with REALTIME_CONDITION:
                    if package_seq in REALTIME_CANCELLED:
                        REALTIME_CANCELLED.discard(package_seq)
                        cancelled = True
                        break
                    if REALTIME_PENDING is not None:
                        has_successor = True
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.01))

        actual_duration_ms = (time.perf_counter() - started) * 1000.0
        idle_state_applied = False
        if not cancelled and not has_successor:
            if package.get("resetAfter"):
                with GAMEPAD_LOCK:
                    gamepad_reset_locked()
                idle_state_applied = True
            elif package.get("idleState") is not None:
                gamepad_state(package["idleState"])
                idle_state_applied = True
            elif package.get("idleHoldLeftTrigger"):
                gamepad_state({
                    "leftStick": {"x": 0, "y": 0},
                    "rightStick": {"x": 0, "y": 0},
                    "leftTrigger": 1,
                    "rightTrigger": 0,
                    "buttons": [],
                })
                idle_state_applied = True
            else:
                gamepad_reset()
                idle_state_applied = True

        write_json({
            "event": "gamepad_package_cancelled" if cancelled else "gamepad_package_completed",
            "packageSeq": package_seq,
            "actualDurationMs": actual_duration_ms,
            "events": actual_events,
            "helperDurationMs": round((time.perf_counter() - started) * 1000.0, 3),
            "completedAtUnixMs": round(time.time() * 1000.0, 3),
            "firstStateApplyMs": round(first_apply_ms, 3) if first_apply_ms is not None else None,
            "lastStateApplyMs": round(last_apply_ms, 3) if last_apply_ms is not None else None,
            "idleStateApplied": idle_state_applied,
            "successorQueued": has_successor,
            "idleLeftTriggerHeld": bool(package.get("idleHoldLeftTrigger")) and not cancelled,
        })
        with REALTIME_CONDITION:
            REALTIME_ACTIVE = None


REALTIME_THREAD = threading.Thread(target=realtime_scheduler, name="gamepad-realtime", daemon=True)
REALTIME_THREAD.start()


def get_cursor():
    point = POINT()
    if not USER32.GetCursorPos(ctypes.byref(point)):
        raise RuntimeError("GetCursorPos failed")
    return point.x, point.y


def key_event(vk, down):
    vk = int(vk)
    scan = USER32.MapVirtualKeyW(vk, 0)
    flags = 0 if down else KEYEVENTF_KEYUP
    USER32.keybd_event(vk, scan, flags, None)


def mouse_button(button, down):
    normalized = str(button or "left").lower()
    flags = {
        ("left", True): MOUSEEVENTF_LEFTDOWN,
        ("left", False): MOUSEEVENTF_LEFTUP,
        ("right", True): MOUSEEVENTF_RIGHTDOWN,
        ("right", False): MOUSEEVENTF_RIGHTUP,
        ("middle", True): MOUSEEVENTF_MIDDLEDOWN,
        ("middle", False): MOUSEEVENTF_MIDDLEUP,
    }
    flag = flags.get((normalized, bool(down)))
    if flag is None:
        raise ValueError(f"Unsupported mouse button: {button}")
    USER32.mouse_event(flag, 0, 0, 0, None)


def gamepad_update():
    started = time.perf_counter()
    GAMEPAD.update()
    return round((time.perf_counter() - started) * 1000.0, 3)


def gamepad_reset():
    with GAMEPAD_LOCK:
        return gamepad_reset_locked()


def gamepad_reset_locked():
    if hasattr(GAMEPAD, "reset"):
        GAMEPAD.reset()
    else:
        GAMEPAD.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        GAMEPAD.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
        GAMEPAD.left_trigger_float(value_float=0.0)
        GAMEPAD.right_trigger_float(value_float=0.0)
        for button in BUTTONS.values():
            GAMEPAD.release_button(button=button)
    return {"gamepadUpdateMs": gamepad_update()}


def gamepad_button(button, down):
    normalized = str(button or "").lower()
    gamepad_button_value = BUTTONS.get(normalized)
    if gamepad_button_value is None:
        raise ValueError(f"Unsupported gamepad button: {button}")
    if down:
        GAMEPAD.press_button(button=gamepad_button_value)
    else:
        GAMEPAD.release_button(button=gamepad_button_value)
    gamepad_update()


def gamepad_state(state):
    with GAMEPAD_LOCK:
        return gamepad_state_locked(state)


def gamepad_state_locked(state):
    state_started = time.perf_counter()
    data = state if isinstance(state, dict) else {}
    left_stick = data.get("leftStick") or {}
    right_stick = data.get("rightStick") or {}
    buttons = data.get("buttons") or []

    GAMEPAD.left_joystick_float(
        x_value_float=clamp(left_stick.get("x"), -1.0, 1.0),
        y_value_float=clamp(left_stick.get("y"), -1.0, 1.0),
    )
    GAMEPAD.right_joystick_float(
        x_value_float=clamp(right_stick.get("x"), -1.0, 1.0),
        y_value_float=clamp(right_stick.get("y"), -1.0, 1.0),
    )
    GAMEPAD.left_trigger_float(value_float=clamp(data.get("leftTrigger"), 0.0, 1.0))
    GAMEPAD.right_trigger_float(value_float=clamp(data.get("rightTrigger"), 0.0, 1.0))

    pressed = {str(button or "").lower() for button in buttons}
    for button_name, button_value in BUTTONS.items():
        if button_name in pressed:
            GAMEPAD.press_button(button=button_value)
        else:
            GAMEPAD.release_button(button=button_value)
    update_ms = gamepad_update()
    return {
        "gamepadStateMs": round((time.perf_counter() - state_started) * 1000.0, 3),
        "gamepadUpdateMs": update_ms,
    }


def gamepad_stick(stick, x, y):
    normalized = str(stick or "left").lower()
    x_value = clamp(x, -1.0, 1.0)
    y_value = clamp(y, -1.0, 1.0)
    if normalized == "left":
        GAMEPAD.left_joystick_float(x_value_float=x_value, y_value_float=y_value)
    elif normalized == "right":
        GAMEPAD.right_joystick_float(x_value_float=x_value, y_value_float=y_value)
    else:
        raise ValueError(f"Unsupported gamepad stick: {stick}")
    gamepad_update()


def gamepad_package(states, chunk_duration_ms, reset_after=False):
    if not isinstance(states, list) or not states:
        raise ValueError("gamepad_package requires a non-empty states list")

    duration_s = max(0.001, float(chunk_duration_ms or 33) / 1000.0)
    deadline = time.perf_counter()
    for state in states:
        gamepad_state(state)
        deadline += duration_s
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    if reset_after:
        gamepad_reset()


def gamepad_package_realtime(package_seq, states, chunk_duration_ms, reset_after=False, idle_hold_left_trigger=False, idle_state=None):
    if not isinstance(states, list) or not states:
        raise ValueError("gamepad_package_realtime requires a non-empty states list")
    if CADENCE.busy() or STREAM.busy():
        raise RuntimeError("Cannot mix legacy and cadence packages")
    with REALTIME_CONDITION:
        global REALTIME_PENDING
        REALTIME_PENDING = {
            "packageSeq": package_seq,
            "states": states,
            "chunkDurationMs": chunk_duration_ms,
            "resetAfter": reset_after,
            "idleHoldLeftTrigger": bool(idle_hold_left_trigger),
            "idleState": idle_state,
        }
        REALTIME_CONDITION.notify()
    return {"accepted": True, "packageSeq": package_seq}


def gamepad_package_realtime_cancel(package_seq):
    with REALTIME_CONDITION:
        if package_seq is not None:
            REALTIME_CANCELLED.add(package_seq)
        REALTIME_CONDITION.notify()
    return {"cancelRequested": True, "packageSeq": package_seq}


def gamepad_trigger(trigger, value):
    normalized = str(trigger or "right").lower()
    trigger_value = clamp(value, 0.0, 1.0)
    if normalized == "left":
        GAMEPAD.left_trigger_float(value_float=trigger_value)
    elif normalized == "right":
        GAMEPAD.right_trigger_float(value_float=trigger_value)
    else:
        raise ValueError(f"Unsupported gamepad trigger: {trigger}")
    gamepad_update()


from gamepad_stream_scheduler import StreamScheduler
def stream_watchdog_pause():
    started = time.perf_counter()*1000
    # Use the same P key as the existing GTA pause harness. A newly attached
    # virtual controller may not yet be selected by the game at startup.
    key_event(0x50, True)
    try:
        time.sleep(0.15)
    finally:
        key_event(0x50, False)
        gamepad_reset()
    return {"pauseSent": True, "pauseKey": "p", "pauseStartedHelperMs": started, "pauseReleasedHelperMs": time.perf_counter()*1000}

STREAM = StreamScheduler(gamepad_state, gamepad_reset, write_json, pause=stream_watchdog_pause)
CADENCE = CadenceScheduler(gamepad_state, gamepad_reset, write_json)
CADENCE_TIMER = None


def ensure_cadence_timer():
    # Condition/Event waits otherwise inherit coarse Windows timer ticks. This
    # request lives only for this helper process and is paired with timeEndPeriod.
    global CADENCE_TIMER
    if CADENCE_TIMER is None:
        timer = ctypes.WinDLL("winmm")
        timer.timeBeginPeriod.argtypes = [ctypes.c_uint]
        timer.timeBeginPeriod.restype = ctypes.c_uint
        timer.timeEndPeriod.argtypes = [ctypes.c_uint]
        timer.timeEndPeriod.restype = ctypes.c_uint
        if timer.timeBeginPeriod(1) != 0:
            raise RuntimeError("Could not enable 1ms cadence timer resolution")
        CADENCE_TIMER = timer


def handle_message(msg):
    kind = msg.get("type")
    if kind == "get_cursor":
        x, y = get_cursor()
        return {"x": x, "y": y}
    if kind == "key_down":
        key_event(msg.get("vk"), True)
        return {}
    if kind == "key_up":
        key_event(msg.get("vk"), False)
        return {}
    if kind == "mouse_move":
        USER32.mouse_event(MOUSEEVENTF_MOVE, int(msg.get("dx") or 0), int(msg.get("dy") or 0), 0, None)
        return {}
    if kind == "mouse_down":
        mouse_button(msg.get("button"), True)
        return {}
    if kind == "mouse_up":
        mouse_button(msg.get("button"), False)
        return {}
    if kind == "wheel":
        USER32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(msg.get("amount") or 0), None)
        return {}
    if kind == "gamepad_stick":
        gamepad_stick(msg.get("stick"), msg.get("x"), msg.get("y"))
        return {}
    if kind == "gamepad_trigger":
        gamepad_trigger(msg.get("trigger"), msg.get("value"))
        return {}
    if kind == "gamepad_button_down":
        gamepad_button(msg.get("button"), True)
        return {}
    if kind == "gamepad_button_up":
        gamepad_button(msg.get("button"), False)
        return {}
    if kind == "gamepad_state":
        return gamepad_state(msg.get("state"))
    if kind == "gamepad_package":
        gamepad_package(
            msg.get("states"),
            msg.get("chunkDurationMs"),
            bool(msg.get("resetAfter")),
        )
        return {}
    if kind == "gamepad_package_realtime":
        return gamepad_package_realtime(
            msg.get("packageSeq"),
            msg.get("states"),
            msg.get("chunkDurationMs"),
            bool(msg.get("resetAfter")),
            bool(msg.get("idleHoldLeftTrigger")),
            msg.get("idleState"),
        )
    if kind == "gamepad_package_realtime_cancel":
        return gamepad_package_realtime_cancel(msg.get("packageSeq"))
    if kind == "gamepad_clock":
        ensure_cadence_timer()
        return {"helperPerfMs": time.perf_counter() * 1000, "timerResolutionMs": 1}
    if kind == "gamepad_stream_arm":
        ensure_cadence_timer()
        if CADENCE.busy(): raise RuntimeError("Cannot mix cadence and stream packages")
        with REALTIME_CONDITION:
            if REALTIME_ACTIVE or REALTIME_PENDING: raise RuntimeError("Cannot mix legacy and stream packages")
        return STREAM.arm(msg.get("epoch"), msg.get("timeoutMs", 1000), msg.get("watchdogPause") is True, msg.get("referenceAtHelperMs"), msg.get("ownerId", "A"), msg.get("ownershipVersion", 0), msg.get("updateVersion", 0))
    if kind == "gamepad_stream_handoff": return STREAM.handoff(msg)
    if kind == "gamepad_stream_state": return STREAM.update(msg)
    if kind == "gamepad_stream_finalize": return STREAM.finalize(msg)
    if kind == "gamepad_stream_cancel": return STREAM.cancel(msg.get("reason", "cancelled"))
    if kind == "gamepad_package_cadence":
        if STREAM.busy(): raise RuntimeError("Cannot mix stream and cadence packages")
        ensure_cadence_timer()
        with REALTIME_CONDITION:
            if REALTIME_ACTIVE or REALTIME_PENDING:
                raise RuntimeError("Cannot mix legacy and cadence packages")
        return CADENCE.enqueue(msg)
    if kind == "gamepad_cadence_cancel":
        return CADENCE.cancel(bool(msg.get("pendingOnly")))
    if kind == "gamepad_reset":
        STREAM.cancel("reset")
        return CADENCE.cancel()
    raise ValueError(f"Unsupported input command: {kind}")


def main():
    write_json({"event": "ready", "backend": "windows-vgamepad-x360", "fixedCadence": True, "streamCadence": True, "streamHandoff": "atomic-owner-v1"})
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            command_id = None
            try:
                msg = json.loads(line)
                command_id = msg.get("id")
                result = handle_message(msg)
                result.update({"id": command_id, "ok": True})
                write_json(result)
            except Exception as error:
                write_json({"id": command_id, "ok": False, "error": str(error)})
    finally:
        try:
            STREAM.close()
            CADENCE.close()
        finally:
            if CADENCE_TIMER is not None:
                CADENCE_TIMER.timeEndPeriod(1)
        global REALTIME_STOP
        with REALTIME_CONDITION:
            REALTIME_STOP = True
            REALTIME_CONDITION.notify()
        REALTIME_THREAD.join(timeout=1.0)
        try:
            gamepad_reset()
            time.sleep(0.02)
        except Exception:
            pass


if __name__ == "__main__":
    main()

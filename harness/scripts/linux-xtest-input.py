import ctypes
import json
import os
import sys
import time
from ctypes.util import find_library


BUTTONS = {
    "left": 1,
    "middle": 2,
    "right": 3,
}

KEY_NAMES = {
    "enter": "Return",
    "return": "Return",
    "escape": "Escape",
    "esc": "Escape",
    "tab": "Tab",
    "space": "space",
    "backspace": "BackSpace",
    "delete": "Delete",
    "del": "Delete",
    "insert": "Insert",
    "home": "Home",
    "end": "End",
    "pageup": "Page_Up",
    "pagedown": "Page_Down",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    "shift": "Shift_L",
    "ctrl": "Control_L",
    "control": "Control_L",
    "alt": "Alt_L",
    "meta": "Super_L",
    "win": "Super_L",
    "windows": "Super_L",
    "pause": "Pause",
    "capslock": "Caps_Lock",
    "numlock": "Num_Lock",
    "scrolllock": "Scroll_Lock",
    "minus": "minus",
    "equal": "equal",
    "comma": "comma",
    "period": "period",
    "slash": "slash",
    "backslash": "backslash",
    "semicolon": "semicolon",
    "quote": "apostrophe",
    "apostrophe": "apostrophe",
    "backquote": "grave",
    "grave": "grave",
    "leftbracket": "bracketleft",
    "rightbracket": "bracketright",
    "bracketleft": "bracketleft",
    "bracketright": "bracketright",
    "numpad0": "KP_0",
    "numpad1": "KP_1",
    "numpad2": "KP_2",
    "numpad3": "KP_3",
    "numpad4": "KP_4",
    "numpad5": "KP_5",
    "numpad6": "KP_6",
    "numpad7": "KP_7",
    "numpad8": "KP_8",
    "numpad9": "KP_9",
    "add": "KP_Add",
    "subtract": "KP_Subtract",
    "multiply": "KP_Multiply",
    "divide": "KP_Divide",
    "decimal": "KP_Decimal",
}

for index in range(1, 13):
    KEY_NAMES[f"f{index}"] = f"F{index}"


def write_json(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def fail_startup(message):
    sys.stderr.write(message + "\n")
    sys.stderr.flush()
    sys.exit(1)


def load_library(name):
    path = find_library(name)
    if not path:
        fail_startup(f"linux-xtest-input requires lib{name}.")
    return ctypes.CDLL(path)


if not os.environ.get("DISPLAY"):
    fail_startup("linux-xtest-input requires an X11 DISPLAY.")

X11 = load_library("X11")
XTST = load_library("Xtst")

X11.XOpenDisplay.argtypes = [ctypes.c_char_p]
X11.XOpenDisplay.restype = ctypes.c_void_p
X11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
X11.XDefaultRootWindow.restype = ctypes.c_ulong
X11.XQueryPointer.argtypes = [
    ctypes.c_void_p,
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_uint),
]
X11.XQueryPointer.restype = ctypes.c_int
X11.XStringToKeysym.argtypes = [ctypes.c_char_p]
X11.XStringToKeysym.restype = ctypes.c_ulong
X11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
X11.XKeysymToKeycode.restype = ctypes.c_ubyte
X11.XFlush.argtypes = [ctypes.c_void_p]
X11.XCloseDisplay.argtypes = [ctypes.c_void_p]

XTST.XTestFakeRelativeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
XTST.XTestFakeRelativeMotionEvent.restype = ctypes.c_int
XTST.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
XTST.XTestFakeButtonEvent.restype = ctypes.c_int
XTST.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
XTST.XTestFakeKeyEvent.restype = ctypes.c_int

DISPLAY = X11.XOpenDisplay(os.environ.get("DISPLAY", "").encode() or None)
if not DISPLAY:
    fail_startup("linux-xtest-input could not open the X11 display.")

ROOT = X11.XDefaultRootWindow(DISPLAY)


def normalize_key(key):
    normalized = str(key or "").strip().lower().replace(" ", "")
    if len(normalized) == 1 and normalized.isalnum():
        return normalized
    if normalized in KEY_NAMES:
        return KEY_NAMES[normalized]
    raise ValueError(f"Unsupported key: {key}")


def keycode_for(key):
    keysym_name = normalize_key(key)
    keysym = X11.XStringToKeysym(keysym_name.encode())
    if not keysym:
        raise ValueError(f"Unsupported key: {key}")
    keycode = X11.XKeysymToKeycode(DISPLAY, keysym)
    if not keycode:
        raise ValueError(f"Unsupported key: {key}")
    return int(keycode)


def query_cursor():
    root_return = ctypes.c_ulong()
    child_return = ctypes.c_ulong()
    root_x = ctypes.c_int()
    root_y = ctypes.c_int()
    win_x = ctypes.c_int()
    win_y = ctypes.c_int()
    mask = ctypes.c_uint()
    ok = X11.XQueryPointer(
        DISPLAY,
        ROOT,
        ctypes.byref(root_return),
        ctypes.byref(child_return),
        ctypes.byref(root_x),
        ctypes.byref(root_y),
        ctypes.byref(win_x),
        ctypes.byref(win_y),
        ctypes.byref(mask),
    )
    if not ok:
        raise RuntimeError("XQueryPointer failed.")
    return {"x": root_x.value, "y": root_y.value}


def flush():
    X11.XFlush(DISPLAY)


def handle(message):
    kind = message.get("type")
    if kind == "get_cursor":
        return query_cursor()
    if kind == "mouse_move":
        XTST.XTestFakeRelativeMotionEvent(
            DISPLAY,
            int(float(message.get("dx") or 0)),
            int(float(message.get("dy") or 0)),
            0,
        )
        flush()
        return {}
    if kind == "mouse_down":
        button = BUTTONS.get(str(message.get("button") or "left").lower(), BUTTONS["left"])
        XTST.XTestFakeButtonEvent(DISPLAY, button, 1, 0)
        flush()
        return {}
    if kind == "mouse_up":
        button = BUTTONS.get(str(message.get("button") or "left").lower(), BUTTONS["left"])
        XTST.XTestFakeButtonEvent(DISPLAY, button, 0, 0)
        flush()
        return {}
    if kind == "key_down":
        XTST.XTestFakeKeyEvent(DISPLAY, keycode_for(message.get("key")), 1, 0)
        flush()
        return {}
    if kind == "key_up":
        XTST.XTestFakeKeyEvent(DISPLAY, keycode_for(message.get("key")), 0, 0)
        flush()
        return {}
    if kind == "wheel":
        amount = float(message.get("amount") or 0)
        button = 4 if amount >= 0 else 5
        clicks = max(1, min(20, round(abs(amount) / 120)))
        for _ in range(clicks):
            XTST.XTestFakeButtonEvent(DISPLAY, button, 1, 0)
            XTST.XTestFakeButtonEvent(DISPLAY, button, 0, 0)
        flush()
        return {}
    raise ValueError(f"Unsupported input command: {kind}")


write_json({"event": "ready", "backend": "linux-x11-python-xtest"})

try:
    for line in sys.stdin:
        if not line.strip():
            continue
        message = None
        try:
            message = json.loads(line)
            result = handle(message)
            write_json({"id": message.get("id"), "ok": True, **result})
        except Exception as error:
            write_json({"id": message.get("id") if message else None, "ok": False, "error": str(error)})
finally:
    X11.XCloseDisplay(DISPLAY)

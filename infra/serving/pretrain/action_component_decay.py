"""Independent direction/button repeat counts within one visible action window."""

import math
import re


WINDOW_SIZE = 5
BUTTONS = "LT LB L3 view up down left right RT RB R3 menu X Y B A".split()
BUTTON_TOKEN = re.compile("|".join(BUTTONS))
ACTION = re.compile(r"<a>((?:[+-][0-9]){4})([^<>\s]*)</a>\Z")


def parse_action(text):
    """Read the existing deadzone-normalized, two-100ms action labels unchanged."""
    match = ACTION.fullmatch(text)
    if not match or "-0" in match[1]:
        raise ValueError(f"Expected an action-only assistant message: {text!r}")
    axes = tuple(int(match[1][i:i + 2]) for i in range(0, 8, 2))
    if any(0 < abs(value) <= 2 for value in axes):
        raise ValueError("Expected existing deadzone-normalized axes (abs <= 2 mapped to 0)")
    chunks = tuple(match[2].split(";"))
    if len(chunks) != 6 or chunks[:3] != (chunks[0],) * 3 or chunks[3:] != (chunks[3],) * 3:
        raise ValueError("Expected two 100ms button states, each repeated three times")
    for chunk in chunks:
        tokens = BUTTON_TOKEN.findall(chunk)
        if "".join(tokens) != chunk or "".join(b for b in BUTTONS if b in tokens) != chunk:
            raise ValueError(f"Invalid canonical button state: {chunk!r}")
    return axes, chunks


def message_text(message):
    content = message["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(part.get("type") == "text" for part in content):
        return "".join(part["text"] for part in content)
    raise ValueError("Assistant action content must be text")


def window_component_weights(messages, gamma, amplitude_reset_threshold=None):
    """Return {message_index: (action, (left, right, buttons))}.

    History contributes to counts, even though its loss remains masked. Counts
    start at one at the visible window boundary; no state crosses samples.
    Direction is (sign(x), sign(y)), so any zero/sign transition resets that
    stick. The optional amplitude threshold compares adjacent actions using
    max(abs(delta_x), abs(delta_y)) independently for each stick. None retains
    the original direction-only rule; small consecutive changes do not accrue.
    """
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("action_decay_gamma must be finite and in (0, 1]")
    if amplitude_reset_threshold is not None and (
        type(amplitude_reset_threshold) is not int or amplitude_reset_threshold <= 0
    ):
        raise ValueError("action_amplitude_reset_threshold must be a positive integer or None")
    previous = None
    previous_axes = None
    counts = [0, 0, 0]
    result = {}
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        action = message_text(message)
        axes, chunks = parse_action(action)
        signs = tuple((value > 0) - (value < 0) for value in axes)
        states = (signs[:2], signs[2:], (chunks[0], chunks[3]))
        amplitude_resets = [False, False, False]
        if previous_axes is not None and amplitude_reset_threshold is not None:
            for stick, offset in enumerate((0, 2)):
                amplitude_resets[stick] = max(
                    abs(axes[axis] - previous_axes[axis]) for axis in (offset, offset + 1)
                ) >= amplitude_reset_threshold
        counts = [count + 1 if previous is not None and state == previous[i] and not amplitude_resets[i] else 1
                  for i, (state, count) in enumerate(zip(states, counts))]
        result[index] = (action, tuple(gamma ** (count - 1) for count in counts))
        previous = states
        previous_axes = axes
    if not 1 <= len(result) <= WINDOW_SIZE:
        raise ValueError("Expected one to five visible assistant actions per window")
    return result

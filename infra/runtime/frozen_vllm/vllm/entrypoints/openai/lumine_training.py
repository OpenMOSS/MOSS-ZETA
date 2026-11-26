# SPDX-License-Identifier: Apache-2.0
"""Canonical v0930/v8.9 contexts above the incremental Lumine KV engine.

Ordinary turns append. Archive rewrites and state updates invalidate the old
prefix and require a fresh engine session. Training's own OCR implementation
is imported lazily, only when this opt-in serving profile is enabled.
"""

from __future__ import annotations

import copy
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERSION = "v0930_runtime_v8_9_fixed_aggregate_20261004"
CALL_BRAIN = "<|call_brain|>"
NO_INSTRUCTION = "暂无明确的阶段目标，需要根据后续观测确认。"
BUTTONS = r"(?:LT|LB|L3|view|up|down|left|right|RT|RB|R3|menu|X|Y|B|A)*"
ASSISTANT = re.compile(
    r"<think>(.*?)</think>\s*<a>((?:[+-][0-9]){4}"
    + BUTTONS
    + r"(?:;"
    + BUTTONS
    + r"){5})</a>\Z",
    re.S,
)


class TrainingBusyError(ValueError):
    pass


class TrainingProtocol:
    def __init__(self):
        from .lumine_rendering import profile
        self.render_profile = profile()
        import logging, json
        from .lumine_rendering import public_profile
        logging.getLogger(__name__).info("Lumine rendering: %s", json.dumps(public_profile(self.render_profile), ensure_ascii=False, sort_keys=True))
        from pretrain import training_ocr_input, training_ocr_lag1
        from pretrain.v0930_fixed_aggregate import RULE
        from pretrain.v0930_fixed_aggregate import VERSION as contract

        if contract != VERSION:
            raise ValueError(
                "Training contract changed; revalidate the serving adapter"
            )
        self.ocr = training_ocr_input
        self.fuzzy_equal = training_ocr_lag1.fuzzy_equal
        root = Path(training_ocr_input.__file__).resolve().parents[1]
        template = (
            (root / "prompts/training-system-prompt-v0930-v8.txt").read_text().strip()
        )
        anchor = "图片和动作历史每满5轮保留最近3轮，再继续累积。"
        if template.count(anchor) != 1:
            raise ValueError("Unknown training system template")
        self.template = template.replace(anchor, RULE + anchor)

    def state(self, value):
        if not isinstance(value, dict) or set(value) != {
            "instruction",
            "memory",
            "model_thought",
        }:
            raise ValueError("lumine_state requires instruction, memory, model_thought")
        result = dict(value)
        for key, text in result.items():
            if key == "instruction" and text is None:
                continue
            if not isinstance(text, str) or any(
                marker in text
                for marker in ("<|", "<think>", "</think>", "<screen_ocr>", "<image>")
            ):
                raise ValueError(f"Invalid state field: {key}")
        return result

    def system(self, state):
        return self.template.format(
            instruction=state["instruction"] or NO_INSTRUCTION,
            memory=state["memory"] or "暂无。",
            model_thought=state["model_thought"],
        )

    def parse_system(self, text):
        try:
            _, tail = text.rsplit("\n\n当前有效目标：", 1)
            instruction, tail = tail.split("\n\nMemory：", 1)
            memory, thought = tail.split("\n\n模型思考：", 1)
        except (ValueError, AttributeError) as error:
            raise ValueError(
                "Expected the v8.9 system prompt or explicit lumine_state"
            ) from error
        state = self.state(
            dict(
                instruction=None if instruction == NO_INSTRUCTION else instruction,
                memory="" if memory == "暂无。" else memory,
                model_thought=thought,
            )
        )
        if self.system(state) != text:
            raise ValueError("System prompt differs from the current training template")
        return state

    def block(self, text, *, archive=False):
        if archive:
            text = text.removeprefix(self.ocr.HEADER)
        events = self.ocr.parse(text)
        if not archive and len(events) > 1:
            raise ValueError("One observation accepts at most one S/D event")
        for event in events:
            for value in event["changes"].values():
                if self.ocr.normalize_value(value) != value:
                    raise ValueError(
                        "Encoded OCR must already use training normalization"
                    )
        return text, events

    def observation(self, message):
        if message.get("role") != "user" or not isinstance(
            message.get("content"), list
        ):
            raise ValueError("Expected one user message containing an image_url")
        images, texts = [], []
        for part in message["content"]:
            if part.get("type") == "image_url":
                images.append(copy.deepcopy(part))
            elif part.get("type") == "text":
                texts.append(part["text"])
            else:
                raise ValueError("v8.9 observations accept only image_url and OCR text")
        if len(images) != 1:
            raise ValueError("Exactly one current image is required")
        text = "".join(texts).strip()
        return images[0], text

    def user(self, image, block):
        return dict(role="user", content=[dict(type="text", text=block + "\n"), image])

    def assistant(self, text):
        match = ASSISTANT.fullmatch(text)
        if not match or "-0" in match[2][:8]:
            raise ValueError(
                "Incomplete/invalid v8.9 think/action output; turn not committed"
            )
        thought = match[1].strip()
        brain = thought.endswith(CALL_BRAIN)
        judgment = thought.removesuffix(CALL_BRAIN).strip()
        if "<|" in judgment or "<|memory|>" in text or judgment.startswith("{"):
            raise ValueError("v8.9 requires natural judgment and input-only Memory")
        return judgment, brain

    def stabilize(self, raw, old_raw, stable):
        if not isinstance(raw, dict) or set(raw) != set(self.ocr.FIELDS):
            raise ValueError("lumine_ocr must be a full current P/B/C/K snapshot")
        if any(not isinstance(v, str) for v in raw.values()):
            raise ValueError("OCR values must be strings")
        normalized = {k: self.ocr.normalize_value(raw[k]) for k in self.ocr.FIELDS}
        result = dict(stable)
        for key, value in normalized.items():
            if old_raw is not None and value == old_raw[key]:
                continue
            if old_raw is None or not self.fuzzy_equal(stable[key], value):
                result[key] = value
        return normalized, result


@dataclass
class TrainingContext:
    state: dict[str, Any]
    mode: str
    system_used: str = ""
    archive: list[dict] = field(default_factory=list)
    turns: list[tuple[dict, str]] = field(default_factory=list)
    delivered: dict[str, str] | None = None
    raw_previous: dict[str, str] | None = None
    stable: dict[str, str] = field(default_factory=lambda: {k: "" for k in "PBCK"})
    reset_next: bool = False
    awaiting_brain: bool = False
    engine_tokens: list = field(default_factory=list)
    engine_media: list = field(default_factory=list)
    engine_fingerprint: str | None = None
    cache_audit: dict | None = None
    updated_at: float = field(default_factory=time.monotonic)


class TrainingTurn:
    def __init__(self, protocol, context, observation, reason):
        self.protocol = protocol
        self.context = context
        self.observation = observation
        self.reason = reason
        self.completed = False
        self.output = []
        self.generated_ids = []
        self.render_profile = protocol.render_profile
        self.tokenizer = None
        self.prefix_ids = []
        self.enable_thinking = True
        self.engine_tokens = []
        self.engine_media = []
        self.engine_fingerprint = None
        self.output_record = None

    def messages(self, *, incremental=False):
        p, c = self.protocol, self.context
        archive = p.ocr.archive(c.archive) if c.archive else p.ocr.EMPTY_BLOCK
        result = [
            dict(role="system", content=p.system(c.state)),
            dict(role="user", content=archive),
        ]
        if not incremental:
            for user, assistant in c.turns:
                result.extend([user, dict(role="assistant", content=assistant)])
        result.append(self.observation)
        return result

    def observe_output(self, output):
        for completion in output.outputs:
            self.output.append(completion.text)
            self.generated_ids.extend(getattr(completion, "token_ids", []))
            if completion.finish_reason is not None:
                from lumine_protocol import reconstruct
                self.output_record = reconstruct("".join(self.output), self.generated_ids,
                    completion.finish_reason, getattr(completion, "stop_reason", None),
                    self.render_profile, self.enable_thinking)
                # Preserve format violations in raw history if the existing action parser accepts them.
                # Truncated or unparseable actions never advance the session.
                if completion.finish_reason != "stop":
                    return
                assistant = self.output_record["full_content"]
                try:
                    judgment, brain = self.protocol.assistant(assistant)
                except ValueError:
                    return
                c = self.context
                c.system_used = self.protocol.system(c.state)
                c.turns.append((self.observation, assistant))
                # Calling brain is an asynchronous event, not a state update.
                # Local judgments still follow the normal next-turn rule while
                # the external goal/Memory update is pending.
                c.awaiting_brain = c.awaiting_brain or brain
                # Generated judgments are candidates. Only the workstation may
                # commit them after the associated actions actually execute.
                c.reset_next = False
                if self.tokenizer is not None:
                    from .lumine_rendering import history_token_audit
                    c.cache_audit = history_token_audit(self.tokenizer, self.render_profile["template"], assistant, self.generated_ids, self.prefix_ids)
                stop = self.tokenizer.encode("<|im_end|>", add_special_tokens=False) if self.tokenizer else []
                generated = list(self.generated_ids)
                if stop and generated[-len(stop):] == stop:
                    generated = generated[:-len(stop)]
                c.engine_tokens = self.engine_tokens + generated
                c.engine_media = self.engine_media
                c.engine_fingerprint = self.engine_fingerprint
                self.completed = True


class TrainingSessions:
    """Transactional text state; failed/cancelled turns never advance OCR/history."""

    def __init__(self, *, idle_timeout_s=1800):
        self.protocol = TrainingProtocol()
        self.idle_timeout_s = idle_timeout_s
        self.contexts: dict[str, TrainingContext] = {}
        from .lumine_requests import RequestLedger
        self.requests = RequestLedger()
        self.max_sessions = int(os.getenv("VLLM_LUMINE_MAX_SESSIONS", "64"))
        self.busy: set[str] = set()

    def reset(self, session_id):
        if session_id in self.busy:
            raise TrainingBusyError(
                "Finish the active turn before resetting this session"
            )
        self.requests.retire(session_id)
        return self.contexts.pop(session_id, None) is not None

    def begin(
        self, session_id, messages, *, state=None, raw_ocr=None, force_reset=False
    ):
        if session_id in self.busy:
            raise TrainingBusyError("Another request is active for this session")
        now = time.monotonic()
        if self.idle_timeout_s > 0:
            for key, value in list(self.contexts.items()):
                if (
                    key not in self.busy
                    and now - value.updated_at > self.idle_timeout_s
                ):
                    self.contexts.pop(key)
                    self.requests.retire(key)
        turn = self._prepare(session_id, messages, state, raw_ocr, force_reset)
        self.busy.add(session_id)
        return turn

    def end(self, session_id, turn, *, commit):
        if commit and turn.completed:
            turn.context.updated_at = time.monotonic()
            self.contexts[session_id] = turn.context
        self.busy.discard(session_id)

    def _prepare(self, session_id, messages, state, raw_ocr, force_reset):
        p = self.protocol
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("The request must end with the current image user message")
        systems = [m for m in messages if m.get("role") == "system"]
        if len(systems) > 1 or (systems and messages[0] is not systems[0]):
            raise ValueError("Only one leading system message is supported")
        if any(m.get("role") not in ("system", "user", "assistant") for m in messages):
            raise ValueError("Unsupported training message role")
        supplied = p.state(state) if state is not None else None
        original = self.contexts.get(session_id)
        if systems:
            parsed = p.parse_system(systems[0]["content"])
            if supplied is not None and supplied != parsed:
                raise ValueError("lumine_state disagrees with the supplied system")
            if original is None or systems[0]["content"] != original.system_used:
                supplied = parsed
        mode = "raw" if raw_ocr is not None else "encoded"
        if raw_ocr is not None:
            raise ValueError("Workstation protocol requires encoded OCR; no server lag")
        if original is None:
            if supplied is None:
                raise ValueError("Session missing/expired: restore with a new ID and complete lumine_state")
            if len(self.contexts) >= self.max_sessions:
                raise TrainingBusyError("Session capacity reached; release an idle session")
            c = TrainingContext(state=supplied, mode=mode)
            reason = "cold_start"
            initial = messages[1:] if systems else messages
            if len(initial) != 1:
                raise ValueError("New session accepts only the current observation; aggregate starts empty")
        else:
            c = copy.deepcopy(original)
            if mode != c.mode:
                raise ValueError("Do not switch raw/encoded OCR mode within a session")
            changed = supplied is not None and supplied != c.state
            if changed or force_reset:
                raise ValueError("State rebuild requires a new session ID")
            reason = "state_update" if c.reset_next or changed or force_reset else ""
            if supplied is not None:
                c.state = supplied
                # Explicit state_reset acknowledges even an unchanged brain
                # result. Re-sending an identical state alone is not a reply.
                if changed or force_reset:
                    c.awaiting_brain = False
            if reason:
                c.turns.clear()
                c.archive.clear()
                c.reset_next = False
        if c.cache_audit and not c.cache_audit["equal"]:
            reason = reason or "history_token_mismatch"
        if len(c.turns) >= 5:
            self._evict(c)
            reason = reason or "archive_update"
        image, block = p.observation(messages[-1])
        fresh = not c.turns and not c.archive
        if mode == "raw":
            if block:
                raise ValueError("Use either lumine_ocr or an encoded inline OCR block")
            available = dict(c.stable)
            c.raw_previous, c.stable = p.stabilize(raw_ocr, c.raw_previous, c.stable)
            changes = {
                k: v
                for k, v in available.items()
                if (bool(v) if fresh else v != c.delivered[k])
            }
            block = (
                p.ocr.render("S" if fresh else "D", changes)
                if fresh or changes
                else p.ocr.EMPTY_BLOCK
            )
            c.delivered = available
        else:
            block = block or p.ocr.EMPTY_BLOCK
            _, events = p.block(block)
            if original is None and fresh and (not events or events[0]["mode"] != "S"):
                raise ValueError(
                    "Cold start requires an S snapshot (empty S for a new recording)"
                )
            c.delivered = p.ocr.replay(block, c.delivered)
            if fresh:
                block = p.ocr.render("S", {k: v for k, v in c.delivered.items() if v})
            elif events and events[0]["mode"] == "S":
                raise ValueError(
                    "An ordinary continuation must use D or an empty OCR block"
                )
        return TrainingTurn(p, c, p.user(image, block), reason)

    def _hydrate(self, context, messages):
        """Accept a canonical training prefix for cold start/resume."""
        p, c = self.protocol, context
        prefix = list(messages[:-1])
        if (
            prefix
            and prefix[0].get("role") == "user"
            and isinstance(prefix[0].get("content"), str)
        ):
            block, events = p.block(prefix.pop(0)["content"], archive=True)
            if events:
                for event in events:
                    c.archive.append(
                        dict(block=p.ocr.render(event["mode"], event["changes"]))
                    )
                c.delivered = p.ocr.replay(block)
        if c.mode == "raw" and (prefix or c.archive):
            raise ValueError(
                "Raw OCR mode starts with one current observation; "
                "use encoded OCR to resume history"
            )
        if len(prefix) % 2:
            raise ValueError("Historical image messages must have assistant responses")
        for index in range(0, len(prefix), 2):
            image, block = p.observation(prefix[index])
            block = block or p.ocr.EMPTY_BLOCK
            p.block(block)
            c.delivered = p.ocr.replay(block, c.delivered)
            answer = prefix[index + 1]
            if answer.get("role") != "assistant":
                raise ValueError("Expected historical assistant")
            judgment, brain = p.assistant(answer["content"])
            c.awaiting_brain = c.awaiting_brain or brain
            if len(c.turns) == 5:
                self._evict(c)
            c.turns.append((p.user(image, block), answer["content"]))

    def _evict(self, context):
        p = self.protocol
        for user, _ in context.turns[:2]:
            _, block = p.observation(user)
            if p.ocr.parse(block):
                context.archive.append(dict(block=block))
        del context.turns[:2]

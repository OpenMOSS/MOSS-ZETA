# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP-to-Engine bridge for Lumine's bounded streaming sessions.

The OpenAI API is request/response shaped, while the Lumine Engine path keeps
one ``AsyncLLM.generate`` call alive and feeds it one turn at a time.  This
module contains the small amount of lifecycle and queue management needed to
bridge those two interfaces.  It intentionally does not know about chat
templates or HTTP; the chat serving layer prepares the first and continuation
EngineInput objects and uses :class:`LumineSessionManager` to submit them.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import dataclass, replace
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.inputs import EngineInput
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import StreamingSessionConfig

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest


class LumineSessionError(RuntimeError):
    """An error that should be returned as an OpenAI-style API error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
        err_type: str = "LumineSessionError",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.param = param
        self.err_type = err_type


@dataclass(slots=True)
class LumineTurn:
    """The output stream and completion state for one HTTP turn."""

    turn_id: int
    output_queue: asyncio.Queue[RequestOutput | BaseException | None]
    done: asyncio.Event
    request_preprocess: dict[str, Any] | None = None
    closed: bool = False

    async def outputs(self) -> AsyncIterator[RequestOutput]:
        while True:
            item = await self.output_queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


def _shift_placeholders(
    placeholders: Mapping[str, list[Any]], offset_delta: int
) -> dict[str, list[Any]]:
    shifted: dict[str, list[Any]] = {}
    for modality, ranges in placeholders.items():
        shifted_ranges: list[Any] = []
        for item in ranges:
            new_offset = item.offset + offset_delta
            if new_offset < 0:
                raise LumineSessionError(
                    "A multimodal placeholder would precede the continuation prompt.",
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                )
            shifted_ranges.append(replace(item, offset=new_offset))
        shifted[modality] = shifted_ranges
    return shifted


def prepend_prompt_tokens(prompt: EngineInput, token_ids: list[int]) -> EngineInput:
    """Prepend carried streaming tokens while preserving image placeholders."""

    if not token_ids:
        return prompt
    updated = copy.copy(prompt)
    prompt_token_ids = list(prompt["prompt_token_ids"])
    updated["prompt_token_ids"] = [*token_ids, *prompt_token_ids]
    if placeholders := prompt.get("mm_placeholders"):
        updated["mm_placeholders"] = _shift_placeholders(
            placeholders, len(token_ids)
        )
    return updated


def slice_lumine_continuation(
    prompt: EngineInput,
    token_start: int,
) -> EngineInput:
    """Slice the last rendered user turn without reprocessing its media.

    Rendering the same image twice is unsafe with the cross-process MM cache:
    the second render can return a hash-only sender-cache hit before EngineCore
    has received the first render. Slice tokens, placeholders, hashes, and the
    already-rendered tensor items together so each OpenAI request performs one
    MM render and sends that exact result to EngineCore.
    """
    prompt_token_ids = list(prompt["prompt_token_ids"])
    if not 0 <= token_start < len(prompt_token_ids):
        raise LumineSessionError(
            "The Lumine continuation token boundary is outside the prompt.",
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            err_type="LumineUnsupportedPrompt",
        )

    updated = copy.copy(prompt)
    updated["prompt_token_ids"] = prompt_token_ids[token_start:]
    updated.pop("prompt", None)
    if prompt.get("type") != "multimodal":
        return updated

    sliced_placeholders: dict[str, list[Any]] = {}
    sliced_hashes: dict[str, list[str]] = {}
    sliced_kwargs: dict[str, list[Any]] = {}
    mm_hashes = prompt["mm_hashes"]
    mm_kwargs = prompt["mm_kwargs"]
    mm_placeholders = prompt["mm_placeholders"]
    if not (
        set(mm_placeholders) == set(mm_hashes) == set(mm_kwargs)
    ):
        raise LumineSessionError(
            "The rendered multimodal modalities are inconsistent.",
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            err_type="LumineUnsupportedPrompt",
        )
    for modality, placeholders in mm_placeholders.items():
        hashes = mm_hashes.get(modality, ())
        items = mm_kwargs.get(modality, ())
        if not (len(placeholders) == len(hashes) == len(items)):
            raise LumineSessionError(
                "The rendered multimodal placeholders, hashes, and data are "
                "inconsistent.",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                err_type="LumineUnsupportedPrompt",
            )

        for placeholder, item_hash, item in zip(placeholders, hashes, items):
            end = placeholder.offset + placeholder.length
            if end <= token_start:
                continue
            if placeholder.offset < token_start:
                raise LumineSessionError(
                    "A multimodal placeholder crosses the Lumine continuation "
                    "boundary.",
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                    err_type="LumineUnsupportedPrompt",
                )
            sliced_placeholders.setdefault(modality, []).append(
                replace(placeholder, offset=placeholder.offset - token_start)
            )
            sliced_hashes.setdefault(modality, []).append(item_hash)
            sliced_kwargs.setdefault(modality, []).append(item)

    updated["mm_placeholders"] = sliced_placeholders
    updated["mm_hashes"] = sliced_hashes
    updated["mm_kwargs"] = MultiModalKwargsItems(sliced_kwargs)
    return updated


def prompt_fingerprint(value: Any) -> str:
    """Return a stable short fingerprint for session configuration metadata."""

    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sampling_params_payload(params: SamplingParams) -> dict[str, Any]:
    """Return the sampling fields that must remain stable in one session.

    The streaming engine can accept a new ``SamplingParams`` object for each
    turn, but changing a parameter that affects token selection would make the
    session's behavior surprising.  Keep this list explicit instead of relying
    on msgspec internals so this bridge remains readable across vLLM versions.
    Internal derived fields (for example ``_eos_token_id``) are deliberately
    excluded.
    """

    fields = (
        "n",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "stop_token_ids",
        "ignore_eos",
        "max_tokens",
        "min_tokens",
        "logprobs",
        "prompt_logprobs",
        "logprob_token_ids",
        "flat_logprobs",
        "detokenize",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "include_stop_str_in_output",
        "structured_outputs",
        "logit_bias",
        "allowed_token_ids",
        "extra_args",
        "bad_words",
        "skip_reading_prefix_cache",
        "thinking_token_budget",
        "repetition_detection",
    )
    return {field: getattr(params, field, None) for field in fields}


async def aggregate_delta_outputs(
    outputs: AsyncIterator[RequestOutput],
) -> AsyncIterator[RequestOutput]:
    """Collapse one HTTP turn's DELTA outputs into one complete output.

    A persistent Engine request must use DELTA output mode; cumulative mode
    would otherwise include text from earlier turns in the same session. The
    normal non-streaming OpenAI response builder, however, expects one complete
    ``RequestOutput``. Aggregate only the deltas consumed for the current HTTP
    turn and yield exactly one independent result.
    """

    aggregate: RequestOutput | None = None
    async for output in outputs:
        if aggregate is None:
            aggregate = copy.deepcopy(output)
        else:
            aggregate.add(output, aggregate=True)
    if aggregate is not None:
        yield aggregate


def _output_finished(output: RequestOutput) -> bool:
    # ``finished`` closes the whole persistent request. A normal resumable
    # input chunk instead ends with ``streaming_turn_finished`` while the
    # Engine remains alive to accept the next Game Host frame.
    return bool(
        output.finished
        or getattr(output, "streaming_turn_finished", False)
    )


class _Session:
    def __init__(
        self,
        *,
        engine_client: EngineClient,
        session_id: str,
        request_id: str,
        sampling_params: SamplingParams,
        session_config: StreamingSessionConfig,
        config_fingerprint: str,
        first_prompt: EngineInput,
        initial_turn_spans: list[tuple[int, int, int]],
        first_retainable_token_ranges: list[tuple[int, int]],
        active_turn_start: int,
        active_retainable_prefix_end: int,
        continuation_boundary_tokens: list[int],
        cache_salt: str | None,
        lora_request: LoRARequest | None,
        trace_headers: Mapping[str, str] | None,
        priority: int,
        data_parallel_rank: int | None,
    ) -> None:
        self.engine_client = engine_client
        self.session_id = session_id
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.session_config = session_config
        self.config_fingerprint = config_fingerprint
        self.continuation_boundary_tokens = continuation_boundary_tokens
        self.initial_turn_spans = initial_turn_spans
        self.first_retainable_token_ranges = first_retainable_token_ranges
        self.active_turn_start = active_turn_start
        self.active_retainable_prefix_end = active_retainable_prefix_end
        self.cache_salt = cache_salt
        self.lora_request = lora_request
        self.trace_headers = trace_headers
        self.priority = priority
        self.data_parallel_rank = data_parallel_rank
        self.created_at = time.time()
        self.last_activity_at = self.created_at
        self.turn_count = 0
        self._next_turn_id = 1
        self._last_completed_turn_id = 0
        self.failed: BaseException | None = None
        # The scheduler/frontend already retain every committed output token
        # except the final sampled token. Carry only that terminal token for a
        # length stop so the continuation remains contiguous without replaying
        # the rest of the previous turn.
        self._carry_tokens: list[int] = []
        self._input_queue: asyncio.Queue[StreamingInput | None] = asyncio.Queue()
        self._active_turn: LumineTurn | None = None
        self._turn_lock = asyncio.Lock()
        self._closed = False
        self._engine_task = asyncio.create_task(
            self._run_engine(), name=f"lumine-session-{session_id}"
        )

        # The first prompt is submitted by reserve_turn().  Keeping it here
        # avoids a race between creating the session and the first HTTP turn.
        self._first_prompt = first_prompt

    async def _input_stream(self) -> AsyncGenerator[StreamingInput, None]:
        while True:
            item = await self._input_queue.get()
            if item is None:
                return
            yield item

    async def _run_engine(self) -> None:
        try:
            async for output in self.engine_client.generate(
                self._input_stream(),
                self.sampling_params,
                self.request_id,
                lora_request=self.lora_request,
                trace_headers=self.trace_headers,
                priority=self.priority,
                data_parallel_rank=self.data_parallel_rank,
            ):
                turn = self._active_turn
                if turn is None:
                    output_turn_id = getattr(output, "streaming_turn_id", None)
                    if (
                        output_turn_id is not None
                        and output_turn_id <= self._last_completed_turn_id
                    ):
                        # An output that arrives after the authoritative
                        # boundary for its turn is stale by definition. The
                        # turn ID makes it safe to discard without ever
                        # attributing it to the next Game Host request.
                        continue
                    raise LumineSessionError(
                        "Lumine Engine produced output without an active HTTP turn.",
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                output_turn_id = getattr(output, "streaming_turn_id", None)
                if output_turn_id is not None and output_turn_id != turn.turn_id:
                    if (
                        output_turn_id <= self._last_completed_turn_id
                    ):
                        continue
                    raise LumineSessionError(
                        "Lumine Engine output turn does not match the active HTTP turn: "
                        f"output={output_turn_id}, active={turn.turn_id}.",
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                turn.output_queue.put_nowait(output)
                if output.outputs:
                    completion = output.outputs[0]
                    if completion.finish_reason is not None:
                        self._carry_tokens = (
                            list(completion.token_ids[-1:])
                            if completion.finish_reason == "length"
                            else []
                        )
                if _output_finished(output):
                    self.turn_count += 1
                    self._last_completed_turn_id = turn.turn_id
                    self.last_activity_at = time.time()
                    self._finish_active_turn()
            if self._active_turn is not None:
                error = LumineSessionError(
                    "The Lumine engine stream ended before the active turn completed.",
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    err_type="LumineSessionFailed",
                )
                self.failed = error
                self._finish_active_turn(error)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self.failed = error
            self._finish_active_turn(error)

    def _finish_active_turn(self, error: BaseException | None = None) -> None:
        turn = self._active_turn
        if turn is None:
            return
        if error is not None:
            turn.output_queue.put_nowait(error)
        turn.output_queue.put_nowait(None)
        turn.done.set()
        self._active_turn = None
        # Session serialization belongs to the Engine turn, not to the HTTP
        # consumer. A disconnected/cancelled client may stop consuming before
        # generation finishes, but the next request becomes eligible as soon as
        # the Engine reaches this turn boundary.
        if self._turn_lock.locked():
            self._turn_lock.release()

    async def reserve_turn(
        self,
        *,
        continuation_prompt: EngineInput | None,
        sampling_params: SamplingParams,
        config_fingerprint: str,
        retainable_prefix_tokens: int = 0,
        retainable_token_ranges: list[tuple[int, int]] | None = None,
    ) -> LumineTurn:
        if self._closed:
            raise LumineSessionError(
                "The Lumine session is closed.",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                err_type="LumineSessionClosed",
            )
        if self._turn_lock.locked():
            raise LumineSessionError(
                "Another request is already active for this Lumine session.",
                status_code=HTTPStatus.CONFLICT,
                err_type="LumineSessionBusy",
            )
        await self._turn_lock.acquire()
        try:
            if self.failed is not None:
                raise LumineSessionError(
                    f"Lumine session is failed: {self.failed}",
                    status_code=HTTPStatus.CONFLICT,
                    err_type="LumineSessionFailed",
                )
            if self._active_turn is not None:
                raise LumineSessionError(
                    "Another request is already active for this Lumine session.",
                    status_code=HTTPStatus.CONFLICT,
                    err_type="LumineSessionBusy",
                )
            if config_fingerprint != self.config_fingerprint:
                raise LumineSessionError(
                    "The Lumine session configuration changed between turns.",
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                    err_type="LumineSessionConfigMismatch",
                )
            if self.turn_count == 0:
                prompt = self._first_prompt
                retained_ranges = self.first_retainable_token_ranges
                suffix_tokens = 0
                session_config = self.session_config
                initial_turn_spans = self.initial_turn_spans
                active_turn_start = self.active_turn_start
                active_retainable_prefix_end = self.active_retainable_prefix_end
            else:
                if continuation_prompt is None:
                    raise LumineSessionError(
                        "A continuation prompt is required after the first turn.",
                        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                    )
                carried_prefix = [
                    *self._carry_tokens,
                    *self.continuation_boundary_tokens,
                ]
                prompt = prepend_prompt_tokens(continuation_prompt, carried_prefix)
                retained_ranges = [
                    (start + len(carried_prefix), end + len(carried_prefix))
                    for start, end in retainable_token_ranges or ()
                ]
                suffix_tokens = len(carried_prefix)
                session_config = None
                initial_turn_spans = None
                active_turn_start = None
                active_retainable_prefix_end = None

            turn_id = self._next_turn_id
            self._next_turn_id += 1
            turn = LumineTurn(turn_id, asyncio.Queue(), asyncio.Event())
            turn.request_preprocess = prompt.get("request_preprocess")
            self._active_turn = turn
            self.last_activity_at = time.time()
            self._input_queue.put_nowait(
                StreamingInput(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    session_config=session_config,
                    previous_turn_suffix_tokens=suffix_tokens,
                    retainable_prefix_tokens=retainable_prefix_tokens,
                    initial_turn_spans=initial_turn_spans,
                    active_turn_start=active_turn_start,
                    active_retainable_prefix_end=active_retainable_prefix_end,
                    turn_id=turn_id,
                    retainable_token_ranges=retained_ranges or None,
                )
            )
            return turn
        except BaseException:
            if self._active_turn is not None and self._active_turn.done.is_set() is False:
                self._active_turn = None
            self._turn_lock.release()
            raise

    async def release_turn(self, turn: LumineTurn) -> None:
        if turn.closed:
            return
        turn.closed = True

    async def shutdown(self) -> None:
        self._closed = True
        if not self._engine_task.done():
            self._input_queue.put_nowait(None)
            self._engine_task.cancel()
        try:
            await self._engine_task
        except asyncio.CancelledError:
            pass
        except BaseException:
            pass
        if self._active_turn is not None and not self._active_turn.done.is_set():
            self._finish_active_turn(
                LumineSessionError(
                    "Lumine session was shut down.",
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    err_type="LumineSessionClosed",
                )
            )
        if self._turn_lock.locked():
            self._turn_lock.release()


class LumineSessionManager:
    """Owns the single-Agent (or explicit-ID) Lumine sessions."""

    def __init__(
        self,
        engine_client: EngineClient,
        *,
        high_watermark: int = 5,
        retain_turns: int = 3,
        window_size: int | None = None,
        eviction_policy: str = "lossy_hybrid_evict",
        idle_timeout_s: float = 1800.0,
    ) -> None:
        self.engine_client = engine_client
        if window_size is not None:
            high_watermark = window_size
        if not 1 <= retain_turns < high_watermark:
            raise ValueError(
                "Lumine window configuration must satisfy "
                "1 <= retain_turns < high_watermark"
            )
        self.high_watermark = high_watermark
        # Keep the former public name for callers that only configure the
        # single-window limit and do not need the 5 -> 3 distinction.
        self.window_size = high_watermark
        self.retain_turns = retain_turns
        self.eviction_policy = eviction_policy
        self.idle_timeout_s = idle_timeout_s
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        *,
        session_id: str,
        request_id: str,
        first_prompt: EngineInput,
        initial_turn_spans: list[tuple[int, int, int]] | None = None,
        first_retainable_token_ranges: list[tuple[int, int]] | None = None,
        active_turn_start: int | None = None,
        active_retainable_prefix_end: int | None = None,
        continuation_boundary_tokens: list[int],
        sink_token_count: int,
        sampling_params: SamplingParams,
        config_fingerprint: str,
        cache_salt: str | None,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> _Session:
        await self._cleanup_expired()
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                if session.config_fingerprint != config_fingerprint:
                    raise LumineSessionError(
                        "The Lumine session configuration changed between turns.",
                        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                        err_type="LumineSessionConfigMismatch",
                    )
                return session

            if sink_token_count < 1:
                raise LumineSessionError(
                    "Unable to identify a stable system prefix for Lumine streaming.",
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                    err_type="LumineUnsupportedPrompt",
                )
            if sampling_params.n != 1:
                raise LumineSessionError(
                    "Lumine streaming currently supports n=1 only.",
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                    param="n",
                )

            session_config = StreamingSessionConfig(
                window_size=self.high_watermark,
                sink_token_count=sink_token_count,
                eviction_policy=self.eviction_policy,
                retain_turns=self.retain_turns,
            )
            if active_turn_start is None:
                active_turn_start = sink_token_count
            if active_retainable_prefix_end is None:
                active_retainable_prefix_end = active_turn_start
            session = _Session(
                engine_client=self.engine_client,
                session_id=session_id,
                request_id=request_id,
                sampling_params=sampling_params,
                session_config=session_config,
                config_fingerprint=config_fingerprint,
                first_prompt=first_prompt,
                initial_turn_spans=initial_turn_spans or [],
                first_retainable_token_ranges=first_retainable_token_ranges or [],
                active_turn_start=active_turn_start,
                active_retainable_prefix_end=active_retainable_prefix_end,
                continuation_boundary_tokens=continuation_boundary_tokens,
                cache_salt=cache_salt,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
            )
            self._sessions[session_id] = session
            return session

    async def has_session(self, session_id: str) -> bool:
        expired_session: _Session | None = None
        async with self._lock:
            session = self._sessions.get(session_id)
            if (
                session is not None
                and self.idle_timeout_s > 0
                and session._active_turn is None
                and time.time() - session.last_activity_at > self.idle_timeout_s
            ):
                expired_session = self._sessions.pop(session_id)
            else:
                return session is not None
        assert expired_session is not None
        await expired_session.shutdown()
        return False

    async def _cleanup_expired(self) -> None:
        if self.idle_timeout_s <= 0:
            return
        now = time.time()
        async with self._lock:
            expired = [
                session_id
                for session_id, session in self._sessions.items()
                if session._active_turn is None
                and now - session.last_activity_at > self.idle_timeout_s
            ]
            sessions = [self._sessions.pop(session_id) for session_id in expired]
        for session in sessions:
            await session.shutdown()

    async def reset(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        await session.shutdown()
        return True

    async def shutdown(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await asyncio.gather(*(session.shutdown() for session in sessions))

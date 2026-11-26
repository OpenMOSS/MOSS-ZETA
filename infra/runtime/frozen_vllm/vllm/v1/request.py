# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
    StreamingSessionConfig,
    StreamingSessionLastEviction,
    StreamingSessionStats,
)
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.streaming_retention import validate_retained_ranges
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.core.sched.output import StreamingKVCompaction


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None
    session_config: StreamingSessionConfig | None
    previous_turn_suffix_tokens: int
    retainable_prefix_tokens: int = 0
    request_preprocess: dict[str, Any] | None = None
    streaming_turn_id: int | None = None
    retainable_token_ranges: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
            session_config=request.streaming_session_config,
            previous_turn_suffix_tokens=(
                request.streaming_previous_turn_suffix_tokens
            ),
            retainable_prefix_tokens=request.streaming_retainable_prefix_tokens,
            request_preprocess=request.request_preprocess,
            streaming_turn_id=request.streaming_turn_id,
            retainable_token_ranges=request.streaming_retainable_token_ranges,
        )


@dataclass
class StreamingTurnSpan:
    """Physical token span for one completed streaming input/output turn."""

    start: int
    end: int
    # Absolute end of the structured OCR message at the start of this turn.
    # ``None`` means the turn has no independently retainable prefix.
    retained_prefix_end: int | None = None
    # Additional absolute text ranges, e.g. OCR after an image. They must start
    # at/after retainable_prefix_end, so legacy prefixes are never copied twice.
    retained_token_ranges: tuple[tuple[int, int], ...] = ()

    @property
    def num_tokens(self) -> int:
        return self.end - self.start

    @property
    def retainable_prefix_end(self) -> int:
        return (
            self.start
            if self.retained_prefix_end is None
            else self.retained_prefix_end
        )

    @property
    def retention_ranges(self) -> tuple[tuple[int, int], ...]:
        prefix = (
            ((self.start, self.retainable_prefix_end),)
            if self.retainable_prefix_end > self.start
            else ()
        )
        return prefix + self.retained_token_ranges


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        streaming_session_config: StreamingSessionConfig | None = None,
        streaming_previous_turn_suffix_tokens: int = 0,
        streaming_retainable_prefix_tokens: int = 0,
        streaming_initial_turn_spans: list[tuple[int, int, int]] | None = None,
        streaming_active_turn_start: int | None = None,
        streaming_active_retainable_prefix_end: int | None = None,
        streaming_turn_id: int | None = None,
        streaming_retainable_token_ranges: list[tuple[int, int]] | None = None,
        request_preprocess: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            self.structured_output_request.reasoning_ended = reasoning_ended
            self.structured_output_request.reasoning_parser_kwargs = (
                reasoning_parser_kwargs
            )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        # Cache per-block prompt-embed hashes to avoid rehashing the same
        # tensor slices when generating extra keys.
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []
        self.request_preprocess = request_preprocess

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers

        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        self.prefill_stats: PrefillStats | None = PrefillStats()
        # Worker timing is produced for every scheduled step, including
        # chunked-prefill steps that intentionally emit no EngineCoreOutput.
        # Keep those events until the next output for this request makes them
        # observable by the frontend.
        self.timing_events: list[dict[str, Any]] = []
        self.timing_turn_id = 1

        self.block_hashes: list[BlockHash] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()
        self.skip_writing_prefix_cache = False

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

        self.streaming_session_config = streaming_session_config
        self.streaming_previous_turn_suffix_tokens = (
            streaming_previous_turn_suffix_tokens
        )
        self.streaming_retainable_prefix_tokens = streaming_retainable_prefix_tokens
        self.streaming_turn_id = streaming_turn_id
        self.streaming_retainable_token_ranges = validate_retained_ranges(
            streaming_retainable_token_ranges or (),
            0,
            self.num_tokens,
            (
                (feature.mm_position.offset,
                 feature.mm_position.offset + feature.mm_position.length)
                for feature in self.mm_features
            ),
        )
        if self.streaming_retainable_token_ranges and not resumable:
            raise ValueError("retained token ranges require a resumable request")
        self.streaming_turns: deque[StreamingTurnSpan] = deque()
        self.streaming_active_turn_start: int | None = None
        self.streaming_active_retainable_prefix_end: int | None = None
        self.streaming_active_retained_token_ranges: tuple[tuple[int, int], ...] = ()
        self.streaming_retained_prefix_end = 0
        self.streaming_evicted_tokens = 0
        self.streaming_rebuild_count = 0
        self.streaming_lossy_compaction_count = 0
        self.streaming_last_eviction: StreamingSessionLastEviction = "none"
        self.streaming_cache_transformed = False
        self.streaming_cache_salt_isolated = False
        self.streaming_kv_compaction: "StreamingKVCompaction | None" = None
        # One-shot scheduler-to-worker signal. Exact streaming rebuilds can
        # immediately recover cached prefix blocks, so num_computed_tokens is
        # not a reliable indication that worker-side recurrent state was
        # discarded.
        self.streaming_rebuild_pending = False
        self.streaming_compaction_source_ranges: tuple[tuple[int, int], ...] | None = (
            None
        )
        self.streaming_compaction_old_num_tokens: int | None = None
        if streaming_session_config is not None:
            if not resumable:
                raise ValueError(
                    "streaming_session_config requires a resumable request"
                )
            if self.prompt_token_ids is None:
                raise ValueError(
                    "streaming_session_config requires tokenized prompt input"
                )
            if streaming_session_config.window_size < 1:
                raise ValueError("streaming session window_size must be at least 1")
            retain_turns = streaming_session_config.retain_turns
            if retain_turns is not None and not (
                1 <= retain_turns < streaming_session_config.window_size
            ):
                raise ValueError(
                    "streaming session retain_turns must satisfy "
                    "1 <= retain_turns < window_size"
                )
            if streaming_session_config.eviction_policy not in (
                "rebuild_on_evict",
                "lossy_hybrid_evict",
            ):
                raise ValueError(
                    "unsupported streaming session eviction policy: "
                    f"{streaming_session_config.eviction_policy}"
                )
            if not 0 <= streaming_session_config.sink_token_count <= self.num_tokens:
                raise ValueError(
                    "streaming session sink_token_count must be within the "
                    "initial prompt"
                )
            sink_token_count = streaming_session_config.sink_token_count
            for mm_feature in self.mm_features:
                start = mm_feature.mm_position.offset
                end = start + mm_feature.mm_position.length
                if start < sink_token_count < end:
                    raise ValueError(
                        "streaming session sink boundary cannot cross a "
                        "multimodal placeholder"
                    )
            self.streaming_retained_prefix_end = sink_token_count
            unassigned_ranges = list(self.streaming_retainable_token_ranges)
            cursor = sink_token_count
            for start, end, retained_prefix_end in streaming_initial_turn_spans or ():
                if not (
                    start == cursor
                    and start <= retained_prefix_end <= end
                    and end <= self.num_tokens
                ):
                    raise ValueError(
                        "streaming initial turn spans must be contiguous and within "
                        "the initial prompt"
                    )
                turn_ranges = tuple(
                    pair for pair in unassigned_ranges if pair[0] < end
                )
                validate_retained_ranges(turn_ranges, retained_prefix_end, end)
                del unassigned_ranges[:len(turn_ranges)]
                self.streaming_turns.append(
                    StreamingTurnSpan(start, end, retained_prefix_end, turn_ranges)
                )
                cursor = end
            active_start = (
                cursor
                if streaming_active_turn_start is None
                else streaming_active_turn_start
            )
            if active_start != cursor or active_start >= self.num_tokens:
                raise ValueError(
                    "streaming active turn must immediately follow completed turns"
                )
            active_prefix_end = (
                active_start + streaming_retainable_prefix_tokens
                if streaming_active_retainable_prefix_end is None
                else streaming_active_retainable_prefix_end
            )
            if not active_start <= active_prefix_end <= self.num_tokens:
                raise ValueError(
                    "streaming active retainable prefix must be within the active turn"
                )
            self.streaming_active_turn_start = active_start
            self.streaming_active_retainable_prefix_end = active_prefix_end
            self.streaming_active_retained_token_ranges = validate_retained_ranges(
                unassigned_ranges, active_prefix_end, self.num_tokens
            )

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            streaming_session_config=request.streaming_session_config,
            streaming_previous_turn_suffix_tokens=(
                request.streaming_previous_turn_suffix_tokens
            ),
            streaming_retainable_prefix_tokens=(
                request.streaming_retainable_prefix_tokens
            ),
            streaming_initial_turn_spans=request.streaming_initial_turn_spans,
            streaming_active_turn_start=request.streaming_active_turn_start,
            streaming_active_retainable_prefix_end=(
                request.streaming_active_retainable_prefix_end
            ),
            streaming_turn_id=request.streaming_turn_id,
            streaming_retainable_token_ranges=request.streaming_retainable_token_ranges,
            request_preprocess=request.request_preprocess,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    def complete_streaming_turn(self) -> None:
        """Record the just-computed turn before the request waits for input."""
        if self.streaming_session_config is None:
            return
        start = self.streaming_active_turn_start
        if start is None:
            raise RuntimeError("streaming session has no active turn")
        end = self.num_computed_tokens
        if end < start:
            raise RuntimeError(
                f"streaming turn end {end} precedes its start {start}"
            )
        retained_prefix_end = self.streaming_active_retainable_prefix_end
        if retained_prefix_end is None:
            retained_prefix_end = start
        if not start <= retained_prefix_end <= end:
            raise RuntimeError(
                "streaming retainable prefix is outside its completed turn"
            )
        self.streaming_turns.append(
            StreamingTurnSpan(
                start=start,
                end=end,
                retained_prefix_end=retained_prefix_end,
                retained_token_ranges=validate_retained_ranges(
                    self.streaming_active_retained_token_ranges,
                    retained_prefix_end,
                    end,
                ),
            )
        )
        self.streaming_active_turn_start = None
        self.streaming_active_retainable_prefix_end = None
        self.streaming_active_retained_token_ranges = ()

    def get_streaming_session_stats(self) -> StreamingSessionStats | None:
        if self.streaming_session_config is None:
            return None
        return StreamingSessionStats(
            evicted_tokens=self.streaming_evicted_tokens,
            rebuild_count=self.streaming_rebuild_count,
            lossy_compaction_count=self.streaming_lossy_compaction_count,
            cache_transformed=self.streaming_cache_transformed,
            last_eviction=self.streaming_last_eviction,
            num_nans_in_logits=self.num_nans_in_logits,
            compaction_source_ranges=self.streaming_compaction_source_ranges,
            compaction_old_num_tokens=self.streaming_compaction_old_num_tokens,
        )

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def take_prefill_stats(self) -> PrefillStats | None:
        if self.prefill_stats is None:
            return None
        prefill_stats = self.prefill_stats
        self.prefill_stats = None
        return prefill_stats

    def take_timing_events(self) -> list[dict[str, Any]]:
        """Return and clear pending fine-grained timing events."""
        events = self.timing_events
        self.timing_events = []
        return events

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}

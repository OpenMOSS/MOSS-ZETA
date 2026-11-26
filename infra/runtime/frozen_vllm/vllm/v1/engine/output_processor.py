# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import msgspec
import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.outputs import (
    STREAM_FINISHED,
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
)
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    extract_trace_context,
    instrument_manual,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreRequest,
    FinishReason,
    StreamingSessionStats,
)
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import (
    IterationStats,
    LoRARequestStates,
    RequestStateStats,
    SchedulerStats,
)

# shared empty CPU tensor used as a placeholder pooling output
EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")


class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.request_id = request_id
        self.output: RequestOutput | PoolingRequestOutput | Exception | None = None
        self._pending_outputs: deque[
            RequestOutput | PoolingRequestOutput | Exception
        ] = deque()
        self.ready = asyncio.Event()

        self._input_stream_task: asyncio.Task | None = None

    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None:
        """Non-blocking put operation."""
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        elif isinstance(self.output, RequestOutput) and isinstance(
            output, RequestOutput
        ):
            current_turn_id = getattr(self.output, "streaming_turn_id", None)
            next_turn_id = getattr(output, "streaming_turn_id", None)
            if (
                current_turn_id is not None
                and next_turn_id is not None
                and current_turn_id != next_turn_id
            ):
                # A persistent streaming request reuses one collector across
                # turns. Never merge an old turn's late output into the next
                # turn; preserve both in arrival order so the session bridge
                # can route or discard them using their turn IDs.
                self._pending_outputs.append(output)
            else:
                # This ensures that request outputs with different request
                # indexes (if n > 1) do not override each other.
                self.output.add(output, aggregate=self.aggregate)
        elif isinstance(self.output, PoolingRequestOutput) and isinstance(
            output, PoolingRequestOutput
        ):
            self.output = output

    async def get(self) -> RequestOutput | PoolingRequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = self._pending_outputs.popleft() if self._pending_outputs else None
        if self.output is None:
            self.ready.clear()
        else:
            self.ready.set()
        if isinstance(output, Exception):
            raise output
        return output

    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = (
                self._pending_outputs.popleft()
                if self._pending_outputs
                else None
            )
            if self.output is None:
                self.ready.clear()
            else:
                self.ready.set()
        if isinstance(output, Exception):
            raise output
        return output

    def close(self):
        if self._input_stream_task is not None:
            self._input_stream_task.cancel()
        self._input_stream_task = None

    def __del__(self):
        if (task := self._input_stream_task) is not None:
            task.get_loop().call_soon_threadsafe(task.cancel)
            self._input_stream_task = None


@dataclass
class OutputProcessorOutput:
    request_outputs: list[RequestOutput | PoolingRequestOutput]
    reqs_to_abort: list[str]


@dataclass
class StreamingUpdate:
    """Streaming input update data for output processor.

    Contains the incremental prompt data to be applied to a request state
    when the current sub-request completes.
    """

    request: EngineCoreRequest
    prompt: str | None
    final: bool = False


class RequestState:
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        lora_request: LoRARequest | None,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: IncrementalDetokenizer | None,
        max_tokens_param: int | None,
        arrival_time: float,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
        top_p: float | None = None,
        n: int | None = None,
        temperature: float | None = None,
        stream_input: bool = False,
        streaming_sink_token_count: int | None = None,
        streaming_window_size: int | None = None,
        streaming_retain_turns: int | None = None,
        frontend_request: EngineCoreRequest | None = None,
        request_preprocess: dict[str, Any] | None = None,
        streaming_turn_id: int | None = None,
    ):
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.parent_req = parent_req
        self.request_index = request_index
        self.lora_request = lora_request
        self.lora_name = lora_request.lora_name if lora_request is not None else None
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        self.max_tokens_param = max_tokens_param
        self.top_p = top_p
        self.n = n
        self.temperature = temperature
        self.is_prefilling = True
        self.queue = queue
        self.num_cached_tokens = 0
        self.streaming_session_stats: StreamingSessionStats | None = None
        self.streaming_sink_token_count = streaming_sink_token_count
        self.streaming_window_size = streaming_window_size
        self.streaming_retain_turns = streaming_retain_turns
        self.streaming_evicted_tokens = 0
        self.frontend_request = frontend_request
        self.request_preprocess = request_preprocess
        self.streaming_turn_id = streaming_turn_id
        # Fine-grained per-turn event counters.  A Lumine session reuses the
        # internal engine request, so these must reset at every continuation
        # boundary rather than being keyed only by request_id.
        self.timing_event_index = 0
        self.timing_prefill_steps = 0
        self.timing_prefill_scheduled_tokens = 0
        self.timing_decode_steps = 0
        self.timing_scheduled_tokens = 0
        self.timing_generated_tokens = 0
        self.timing_accepted_tokens = 0
        self.timing_accepted_draft_tokens = 0
        self.timing_rejected_tokens = 0
        self.timing_events: list[dict[str, Any]] = []
        # EngineCore may produce a timing event on a chunked-prefill step
        # without producing a user-visible output (FINAL_ONLY and stream
        # interval filtering are the common cases). Keep those events until
        # the next output so fine_v2 does not silently lose forward steps.
        self.pending_timing_events: list[dict[str, Any]] = []

        self.stats = RequestStateStats(arrival_time=arrival_time) if log_stats else None

        # Stream Interval
        self.stream_interval = stream_interval
        self.sent_tokens_offset = 0  # Offset of sent tokens

        # Streaming input queue
        self.streaming_input = stream_input
        self.input_chunk_queue: deque[StreamingUpdate] | None = (
            deque() if stream_input else None
        )
        self.last_completed_streaming_turn_id: int | None = None

    def apply_streaming_update(
        self, update: StreamingUpdate, tokenizer: TokenizerLike | None
    ) -> None:
        """Start a new output turn for a resumable request.

        The scheduler turns all computed output tokens from the previous turn
        into prompt tokens and drops the final sampled token. Rebuild the
        frontend detokenization/logprobs state against the same physical prompt
        so DELTA offsets and per-turn sampling metadata do not leak across
        turns.
        """
        request = update.request
        sampling_params = request.sampling_params
        assert sampling_params is not None
        assert self.detokenizer is not None
        assert self.prompt_token_ids is not None

        previous_output_token_ids = self.detokenizer.output_token_ids
        computed_output_token_ids = previous_output_token_ids[:-1]
        prompt_token_ids = [
            *self.prompt_token_ids,
            *computed_output_token_ids,
            *(request.prompt_token_ids or ()),
        ]
        rebuilt_request = msgspec.structs.replace(
            request,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=None,
        )
        self.frontend_request = msgspec.structs.replace(
            rebuilt_request,
            mm_features=None,
        )

        self.streaming_input = not update.final
        if update.prompt:
            self.prompt = (
                (self.prompt + update.prompt) if self.prompt else update.prompt
            )
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = None
        self.prompt_len = len(prompt_token_ids)
        self.request_preprocess = request.request_preprocess
        self.streaming_turn_id = request.streaming_turn_id
        self.timing_event_index = 0
        self.timing_prefill_steps = 0
        self.timing_prefill_scheduled_tokens = 0
        self.timing_decode_steps = 0
        self.timing_scheduled_tokens = 0
        self.timing_generated_tokens = 0
        self.timing_accepted_tokens = 0
        self.timing_accepted_draft_tokens = 0
        self.timing_rejected_tokens = 0
        self.timing_events.clear()
        self.pending_timing_events.clear()

        if not sampling_params.detokenize:
            tokenizer = None
        self.detokenizer = IncrementalDetokenizer.from_new_request(
            tokenizer=tokenizer,
            request=rebuilt_request,
        )
        self.logprobs_processor = LogprobsProcessor.from_new_request(
            tokenizer=tokenizer,
            request=rebuilt_request,
        )
        self.output_kind = sampling_params.output_kind
        self.max_tokens_param = sampling_params.max_tokens
        self.top_p = sampling_params.top_p
        self.n = sampling_params.n
        self.temperature = sampling_params.temperature
        self.sent_tokens_offset = 0
        self.num_cached_tokens = 0
        self.streaming_session_stats = None
        if self.stats is not None:
            self.stats.reset_for_streaming_turn(request.arrival_time)
        self.is_prefilling = True

    def apply_streaming_session_stats(
        self,
        stats: StreamingSessionStats | None,
        tokenizer: TokenizerLike | None,
    ) -> None:
        """Apply scheduler-side bounded-session compaction to frontend state."""
        self.streaming_session_stats = stats
        if stats is None:
            return

        newly_evicted = stats.evicted_tokens - self.streaming_evicted_tokens
        if newly_evicted < 0:
            raise RuntimeError("streaming evicted-token counter moved backwards")
        if newly_evicted == 0:
            return

        sink_end = self.streaming_sink_token_count
        if sink_end is None or self.prompt_token_ids is None:
            raise RuntimeError(
                "received streaming eviction stats without a tokenized session prompt"
            )
        if self.frontend_request is None:
            raise RuntimeError("streaming session has no frontend request state")
        source_ranges = stats.compaction_source_ranges
        old_num_tokens = stats.compaction_old_num_tokens
        previous_suffix_start = sink_end
        if source_ranges is not None:
            if old_num_tokens is None or not 0 < old_num_tokens <= len(
                self.prompt_token_ids
            ):
                raise RuntimeError(
                    "streaming compaction has an invalid frontend token boundary"
                )
            previous_end = 0
            compact_prefix: list[int] = []
            for index, (start, end) in enumerate(source_ranges):
                if not 0 <= start < end <= old_num_tokens:
                    raise RuntimeError(
                        "streaming compaction source range exceeds the frontend prompt"
                    )
                if index and start < previous_end:
                    raise RuntimeError(
                        "streaming compaction source ranges are not ordered"
                    )
                previous_end = end
                compact_prefix.extend(self.prompt_token_ids[start:end])
            self.prompt_token_ids = [
                *compact_prefix,
                *self.prompt_token_ids[old_num_tokens:],
            ]
            previous_suffix_start = len(compact_prefix)
        else:
            suffix_start = sink_end + newly_evicted
            if suffix_start > len(self.prompt_token_ids):
                raise RuntimeError(
                    "streaming eviction exceeds the frontend physical prompt"
                )
            self.prompt_token_ids = [
                *self.prompt_token_ids[:sink_end],
                *self.prompt_token_ids[suffix_start:],
            ]
        effective_retain_turns = (
            self.streaming_window_size
            if self.streaming_retain_turns is None
            else self.streaming_retain_turns
        )
        if effective_retain_turns == 1:
            suffix_tokens = self.frontend_request.streaming_previous_turn_suffix_tokens
            if suffix_tokens:
                suffix_end = previous_suffix_start + suffix_tokens
                if suffix_end > len(self.prompt_token_ids):
                    raise RuntimeError(
                        "streaming turn suffix exceeds the frontend physical prompt"
                    )
                del self.prompt_token_ids[previous_suffix_start:suffix_end]
        self.prompt_len = len(self.prompt_token_ids)
        self.streaming_evicted_tokens = stats.evicted_tokens

        if self.detokenizer is None or self.detokenizer.num_output_tokens() != 0:
            raise RuntimeError(
                "streaming eviction arrived after frontend output processing began"
            )
        compact_request = msgspec.structs.replace(
            self.frontend_request,
            prompt_token_ids=self.prompt_token_ids,
            prompt_embeds=None,
        )
        sampling_params = compact_request.sampling_params
        assert sampling_params is not None
        if not sampling_params.detokenize:
            tokenizer = None
        self.detokenizer = IncrementalDetokenizer.from_new_request(
            tokenizer=tokenizer,
            request=compact_request,
        )
        self.logprobs_processor = LogprobsProcessor.from_new_request(
            tokenizer=tokenizer,
            request=compact_request,
        )
        self.sent_tokens_offset = 0
        self.frontend_request = compact_request

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
    ) -> "RequestState":
        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens
            top_p = sampling_params.top_p
            n = sampling_params.n
            temperature = sampling_params.temperature
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind

        assert request.external_req_id is not None
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_request=request.lora_request,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            top_p=top_p,
            n=n,
            temperature=temperature,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
            stream_interval=stream_interval,
            stream_input=request.resumable,
            streaming_sink_token_count=(
                request.streaming_session_config.sink_token_count
                if request.streaming_session_config is not None
                else None
            ),
            streaming_window_size=(
                request.streaming_session_config.window_size
                if request.streaming_session_config is not None
                else None
            ),
            streaming_retain_turns=(
                request.streaming_session_config.retain_turns
                if request.streaming_session_config is not None
                else None
            ),
            frontend_request=(
                msgspec.structs.replace(request, mm_features=None)
                if request.resumable
                else None
            ),
            request_preprocess=request.request_preprocess,
            streaming_turn_id=request.streaming_turn_id,
        )

    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        routed_experts: np.ndarray | None = None,
        timing: dict[str, Any] | None = None,
        timing_events: list[dict[str, Any]] | None = None,
        request_metrics: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput | None:
        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None

        if self.stream_interval > 1:
            assert self.detokenizer is not None

            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            if not (
                finished
                or self.sent_tokens_offset == 0
                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
                >= self.stream_interval
            ):
                return None

            if self.output_kind == RequestOutputKind.DELTA:
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                new_token_ids = self.detokenizer.output_token_ids[
                    self.sent_tokens_offset :
                ]
                self.sent_tokens_offset = self.detokenizer.num_output_tokens()

        external_req_id = self.external_req_id

        if pooling_output is not None:
            return self._new_request_output(
                external_req_id,
                [self._new_pooling_output(pooling_output)],
                finished,
            )

        output = self._new_completion_output(
            new_token_ids, finish_reason, stop_reason, routed_experts
        )

        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        return self._new_request_output(
            external_req_id,
            outputs,
            finished,
            kv_transfer_params,
            timing=timing,
            timing_events=timing_events,
            request_metrics=request_metrics,
        )

    def add_pending_timing_events(
        self, timing_events: list[dict[str, Any]] | None
    ) -> None:
        if timing_events:
            self.pending_timing_events.extend(
                event for event in timing_events if isinstance(event, dict)
            )
            if len(self.pending_timing_events) > 512:
                del self.pending_timing_events[:-512]

    def take_pending_timing_events(self) -> list[dict[str, Any]]:
        events = self.pending_timing_events
        self.pending_timing_events = []
        return events

    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list[CompletionOutput] | list[PoolingOutput],
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
        timing: dict[str, Any] | None = None,
        timing_events: list[dict[str, Any]] | None = None,
        request_metrics: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput:
        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
        assert prompt_token_ids is not None

        first_output = outputs[0]
        if isinstance(first_output, PoolingOutput):
            assert len(outputs) == 1
            return PoolingRequestOutput(
                request_id=external_req_id,
                outputs=first_output,
                num_cached_tokens=self.num_cached_tokens,
                prompt_token_ids=prompt_token_ids,
                finished=finished,
            )
        assert self.logprobs_processor is not None
        if self.output_kind == RequestOutputKind.DELTA:
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs

        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            streaming_session_stats=self.streaming_session_stats,
            timing=timing,
            timing_events=timing_events,
            request_metrics=request_metrics,
            request_preprocess=self.request_preprocess,
            num_cached_tokens=self.num_cached_tokens,
            metrics=self.stats,
        )

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        routed_experts: np.ndarray | None = None,
    ) -> CompletionOutput:
        assert self.detokenizer is not None
        assert self.logprobs_processor is not None
        finished = finish_reason is not None
        delta = self.output_kind == RequestOutputKind.DELTA

        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids

        # Prepare logprobs, based on delta mode
        logprobs = self.logprobs_processor.logprobs
        if delta and logprobs:
            logprobs = logprobs[-len(token_ids) :]

        return CompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            routed_experts=routed_experts,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
        )

    def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
        return PoolingOutput(data=pooling_output)


class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
    ):
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        self.stream_interval = stream_interval
        self.request_states: dict[str, RequestState] = {}
        self.parent_requests: dict[str, ParentRequest] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        self.lora_states = LoRARequestStates(log_stats)
        self.tracing_enabled = tracing_enabled

    def get_num_unfinished_requests(self):
        return len(self.request_states)

    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    @staticmethod
    def _request_metrics_snapshot(
        req_state: RequestState,
        engine_core_output: EngineCoreOutput,
    ) -> dict[str, Any] | None:
        """Build request-level timing for one output/continuation turn.

        Prometheus request histograms are emitted only when the underlying
        vLLM request is finally removed. A Lumine session intentionally keeps
        that request alive across turns, so expose the same intervals directly
        on each streamed turn instead.
        """
        stats = getattr(req_state, "stats", None)
        if stats is None:
            return None
        now_wall = time.time()
        scheduled = stats.scheduled_ts
        queued = stats.queued_ts
        first = stats.first_token_ts or stats.last_token_ts
        if first <= 0.0:
            return None
        last = stats.last_token_ts or first
        return {
            "queue_ms": round(max(0.0, scheduled - queued) * 1000.0, 3),
            "prefill_ms": round(max(0.0, first - scheduled) * 1000.0, 3),
            "inference_ms": round(max(0.0, last - scheduled) * 1000.0, 3),
            "decode_ms": round(max(0.0, last - first) * 1000.0, 3),
            "e2e_ms": round(max(0.0, now_wall - stats.arrival_time) * 1000.0, 3),
            "ttft_ms": round(max(0.0, stats.first_token_latency) * 1000.0, 3),
            "prompt_tokens": req_state.prompt_len,
            "generation_tokens": stats.num_generation_tokens,
            "prompt_tokens_cached": req_state.num_cached_tokens,
            # Prefix-cache hits exclude KV retained by a resumable session.
            # Count actual prefill work, including chunks without output.
            "prefill_kv_computed_tokens": (
                req_state.timing_prefill_scheduled_tokens
                if req_state.timing_prefill_steps
                else max(0, req_state.prompt_len - req_state.num_cached_tokens)
            ),
            "prefill_token_count_source": (
                "scheduled_prefill_events"
                if req_state.timing_prefill_steps
                else "prompt_minus_prefix_cache_estimate"
            ),
            "forward_steps": req_state.timing_event_index,
            "prefill_forward_steps": req_state.timing_prefill_steps,
            "decode_forward_steps": req_state.timing_decode_steps,
            "scheduled_tokens": req_state.timing_scheduled_tokens,
            "generated_tokens": req_state.timing_generated_tokens,
            "accepted_tokens": req_state.timing_accepted_tokens,
            "accepted_draft_tokens": req_state.timing_accepted_draft_tokens,
            "rejected_tokens": req_state.timing_rejected_tokens,
            "request_success": int(engine_core_output.finish_reason is not None),
            "source": "engine_request_metrics",
        }

    def propagate_error(self, e: Exception):
        """Propagate error to all generate() tasks."""

        for _, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)

    def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
        """Abort a list of requests.

        The request_ids may be either external request IDs (those passed to
        InputProcessor.process_inputs()) or internal request IDs (those randomly
        generated when creating the EngineCoreRequest).

        If an external request ID is provided, and that external request ID
        was used for multiple requests, all requests associated with that external
        request ID are aborted.

        In the case of parallel sampling, a request ID may be used to identify
        a parent request, in which case the associated child requests are aborted
        also.
        """
        internal_req_ids = []
        for request_id in request_ids:
            if internal:
                # Internal ID - this may be a parent request
                internal_req_ids.append(request_id)

                # Remove internal ID from the external->internal mapping
                if req_state := self.request_states.get(request_id):
                    external_req_id = req_state.external_req_id
                    internal_ids = self.external_req_ids[external_req_id]
                    internal_ids.remove(request_id)
                    if not internal_ids:
                        del self.external_req_ids[external_req_id]
            elif internal_ids := self.external_req_ids.pop(request_id, []):
                # External ID - abort all requests in the external->internal mapping
                internal_req_ids.extend(internal_ids)

        request_ids_to_abort = []
        for request_id in internal_req_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                self.lora_states.request_finished(request_id, req_state.lora_name)
                request_ids_to_abort.append(request_id)
                # Produce final abort output.
                if req_state.queue is not None and (
                    request_output := req_state.make_request_output(
                        new_token_ids=[],
                        # Set pooling_output is not None to
                        # correctly enter the abort pooling branch
                        pooling_output=EMPTY_CPU_TENSOR
                        if req_state.detokenizer is None
                        else None,
                        finish_reason=FinishReason.ABORT,
                        stop_reason=None,
                        kv_transfer_params=None,
                    )
                ):
                    req_state.queue.put(request_output)
            elif parent := self.parent_requests.get(request_id):
                # Abort children prior to removing the parent.
                if parent.child_requests:
                    child_reqs = list(parent.child_requests)
                    child_reqs = self.abort_requests(child_reqs, internal=True)
                    request_ids_to_abort.extend(child_reqs)
                self.parent_requests.pop(request_id, None)
        return request_ids_to_abort

    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        request_id = request.request_id
        req_state = self.request_states.get(request_id)
        if req_state is not None:
            self._update_streaming_request_state(req_state, request, prompt)
            return

        req_state = RequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
        )
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        # Track the external_req_id -> [internal_req_id, ...] mapping
        self.external_req_ids[req_state.external_req_id].append(request_id)

    def _update_streaming_request_state(
        self, req_state: RequestState, request: EngineCoreRequest, prompt: str | None
    ) -> None:
        """Queue a streaming update instead of immediately applying it."""
        if not request.resumable:
            # Final request - just mark completion, don't add its dummy tokens.
            if req_state.input_chunk_queue is None:
                # Engine already finished - emit final output and clean up.
                self._finish_request(req_state)
                if req_state.queue is not None:
                    # Emit a final output with finished=True
                    # to unblock the generate() loop.
                    req_state.queue.put(STREAM_FINISHED)
            elif req_state.input_chunk_queue:
                req_state.input_chunk_queue[-1].final = True
            else:
                req_state.streaming_input = False
            return

        update = StreamingUpdate(
            request=request,
            prompt=prompt,
        )

        # Apply request updates now if the last input already completed.
        if req_state.input_chunk_queue is None:
            req_state.apply_streaming_update(update, self.tokenizer)
            req_state.input_chunk_queue = deque()
        else:
            # Queue the streaming update otherwise.
            req_state.input_chunk_queue.append(update)

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If there is a queue (for usage with AsyncLLM),
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.

            * If there is no queue (for usage with LLMEngine),
              return a list of RequestOutput objects.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, do it from
        within the loop below.
        """

        request_outputs: list[RequestOutput | PoolingRequestOutput] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            output_process_started = (
                time.monotonic() if engine_core_output.timing is not None else 0.0
            )
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue
            engine_core_turn_id = engine_core_output.streaming_turn_id
            last_completed_turn_id = getattr(
                req_state,
                "last_completed_streaming_turn_id",
                None,
            )
            if (
                engine_core_turn_id is not None
                and last_completed_turn_id is not None
                and engine_core_turn_id <= last_completed_turn_id
            ):
                # The Scheduler's authoritative stop already completed this
                # turn. A late speculative/control output must not mutate the
                # detokenizer or logprobs state of the following turn.
                continue

            # The request state is authoritative here: token-count heuristics
            # cannot reliably distinguish a one-token/chunked prefill from
            # decode.  Persist this label with every streamed engine step.
            if engine_core_output.timing is not None:
                for attr in (
                    "timing_event_index",
                    "timing_prefill_steps",
                    "timing_prefill_scheduled_tokens",
                    "timing_decode_steps",
                    "timing_scheduled_tokens",
                    "timing_generated_tokens",
                    "timing_accepted_tokens",
                    "timing_accepted_draft_tokens",
                    "timing_rejected_tokens",
                ):
                    if not hasattr(req_state, attr):
                        setattr(req_state, attr, 0)

                counter_events = engine_core_output.timing_events or [
                    engine_core_output.timing
                ]
                for event in counter_events:
                    timing_phase = event.get("phase")
                    if timing_phase not in ("prefill", "decode"):
                        timing_phase = (
                            "prefill" if req_state.is_prefilling else "decode"
                        )
                        event["phase"] = timing_phase
                    req_state.timing_event_index += 1
                    if timing_phase == "prefill":
                        req_state.timing_prefill_steps += 1
                        scheduled_tokens = event.get("scheduled_tokens", 0)
                        if isinstance(scheduled_tokens, (int, float)):
                            req_state.timing_prefill_scheduled_tokens += int(
                                scheduled_tokens
                            )
                    else:
                        req_state.timing_decode_steps += 1
                    for field, attr in (
                        ("scheduled_tokens", "timing_scheduled_tokens"),
                        ("generated_tokens", "timing_generated_tokens"),
                        ("accepted_tokens", "timing_accepted_tokens"),
                        ("accepted_draft_tokens", "timing_accepted_draft_tokens"),
                        ("rejected_tokens", "timing_rejected_tokens"),
                    ):
                        value = event.get(field, 0)
                        if isinstance(value, (int, float)):
                            setattr(req_state, attr, getattr(req_state, attr) + int(value))
                    event.update(
                        {
                            "request_event_index": req_state.timing_event_index,
                            "prefill_forward_steps": req_state.timing_prefill_steps,
                            "decode_forward_steps": req_state.timing_decode_steps,
                            "scheduled_tokens_total": req_state.timing_scheduled_tokens,
                            "generated_tokens_total": req_state.timing_generated_tokens,
                            "accepted_tokens_total": req_state.timing_accepted_tokens,
                            "accepted_draft_tokens_total": req_state.timing_accepted_draft_tokens,
                            "rejected_tokens_total": req_state.timing_rejected_tokens,
                        }
                    )
                engine_core_output.timing.update(counter_events[-1])
                vision_by_request = engine_core_output.timing.pop(
                    "vision_by_request", None
                )
                if isinstance(vision_by_request, dict):
                    vision_stats = vision_by_request.get(req_id)
                    if isinstance(vision_stats, dict):
                        encoder_secs = vision_stats.get("encoder_forward_secs")
                        if isinstance(encoder_secs, (int, float)):
                            engine_core_output.timing["vision_encoder_ms"] = round(
                                float(encoder_secs) * 1000.0, 3
                            )
                        calls = vision_stats.get("num_encoder_calls")
                        if isinstance(calls, (int, float)):
                            engine_core_output.timing["vision_encoder_calls"] = int(
                                calls
                            )
                        cache_hit = vision_stats.get("encoder_cache_hit")
                        if isinstance(cache_hit, bool):
                            engine_core_output.timing["vision_encoder_cache_hit"] = (
                                cache_hit
                            )
                        local_work = vision_stats.get("encoder_work_local")
                        if isinstance(local_work, bool):
                            engine_core_output.timing["vision_encoder_work_local"] = (
                                local_work
                            )
                        any_rank_work = vision_stats.get("encoder_work_any_rank")
                        if isinstance(any_rank_work, bool):
                            engine_core_output.timing[
                                "vision_encoder_work_any_rank"
                            ] = any_rank_work
                        embedding_tokens = vision_stats.get(
                            "encoder_embedding_tokens"
                        )
                        if isinstance(embedding_tokens, (int, float)):
                            engine_core_output.timing[
                                "vision_encoder_embedding_tokens"
                            ] = int(embedding_tokens)
                        attribution_weight = vision_stats.get(
                            "encoder_attribution_weight"
                        )
                        if isinstance(attribution_weight, (int, float)):
                            engine_core_output.timing[
                                "vision_encoder_attribution_weight"
                            ] = float(attribution_weight)
                        attribution_method = vision_stats.get(
                            "encoder_attribution_method"
                        )
                        if isinstance(attribution_method, str):
                            engine_core_output.timing[
                                "vision_encoder_attribution_method"
                            ] = attribution_method
                if req_state.request_preprocess is not None:
                    engine_core_output.timing["request_preprocess"] = dict(
                        req_state.request_preprocess
                    )

            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            finish_reason = engine_core_output.finish_reason
            engine_core_turn_finished = finish_reason is not None
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            routed_experts = engine_core_output.routed_experts

            # Scheduler timing-only carrier for chunked prefill. It exists
            # solely to transport a forward-step event; do not update
            # detokenizer, prefill state, or request-level output metrics.
            if (
                not new_token_ids
                and pooling_output is None
                and finish_reason is None
                and engine_core_output.timing_events
            ):
                if hasattr(req_state, "is_prefilling"):
                    req_state.is_prefilling = True
                add_pending_events = getattr(
                    req_state, "add_pending_timing_events", None
                )
                if add_pending_events is not None:
                    add_pending_events(engine_core_output.timing_events)
                continue

            # 1) Compute stats for visible token/pooling outputs only.
            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            req_state.apply_streaming_session_stats(
                engine_core_output.streaming_session_stats,
                self.tokenizer,
            )

            if req_state.is_prefilling:
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                req_state.is_prefilling = False

            if pooling_output is None:
                assert req_state.detokenizer is not None
                assert req_state.logprobs_processor is not None
                # 2) Detokenize the token ids into text and perform stop checks.
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP
                )
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string

                # 3) Compute sample and prompt logprobs for request,
                # if required.
                req_state.logprobs_processor.update_from_output(engine_core_output)

            # 4) Create and handle RequestOutput objects. A timing event may
            # precede a visible output (chunked prefill, FINAL_ONLY, or stream
            # interval filtering), so carry those events into the next output.
            take_pending_events = getattr(
                req_state, "take_pending_timing_events", lambda: []
            )
            output_timing_events = [
                *take_pending_events(),
                *(
                    engine_core_output.timing_events
                    if engine_core_output.timing_events
                    else ()
                ),
            ]
            request_metrics = self._request_metrics_snapshot(
                req_state, engine_core_output
            )

            request_output = req_state.make_request_output(
                new_token_ids,
                pooling_output,
                finish_reason,
                stop_reason,
                kv_transfer_params,
                routed_experts,
                engine_core_output.timing,
                output_timing_events,
                request_metrics,
            )
            was_streaming_input = bool(req_state.streaming_input)
            completed_turn_id = (
                engine_core_turn_id
                if engine_core_turn_id is not None
                else getattr(req_state, "streaming_turn_id", None)
            )
            if request_output:
                if engine_core_output.timing is not None:
                    engine_core_output.timing["output_process_ms"] = round(
                        (time.monotonic() - output_process_started) * 1000.0, 3
                    )
                if was_streaming_input:
                    request_output.finished = False
                    request_output.streaming_turn_id = completed_turn_id
            else:
                # FINAL_ONLY/stream-interval filtering suppressed this
                # otherwise visible output. Restore every event so the next
                # emitted RequestOutput carries the complete step history.
                add_pending_events = getattr(
                    req_state, "add_pending_timing_events", None
                )
                if add_pending_events is not None:
                    add_pending_events(output_timing_events)

            # Free completed requests.
            if finish_reason is not None:
                if req_state.streaming_input:
                    # A Lumine continuation is represented as one resumable
                    # vLLM request whose frontend state survives across turns.
                    # It still has a complete per-turn latency interval, so
                    # record it before apply_streaming_update resets the
                    # RequestStateStats timestamps for the next turn.  Without
                    # this, Prometheus only observes TTFT (an iteration metric)
                    # and never emits request_success/prefill/inference/e2e
                    # histograms for Lumine turns.
                    self._update_stats_from_finished(
                        req_state, finish_reason, iteration_stats
                    )
                    if req_state.input_chunk_queue:
                        update = req_state.input_chunk_queue.popleft()
                        req_state.apply_streaming_update(update, self.tokenizer)
                    else:
                        req_state.input_chunk_queue = None
                else:
                    self._finish_request(req_state)
                    if not engine_core_output.finished:
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        reqs_to_abort.append(req_id)

                    # Track per-request stats
                    self._update_stats_from_finished(
                        req_state, finish_reason, iteration_stats
                    )
                    if self.tracing_enabled:
                        self.do_tracing(engine_core_output, req_state, iteration_stats)

            if request_output:
                if was_streaming_input and engine_core_turn_finished:
                    assert isinstance(request_output, RequestOutput)
                    # Carry the authoritative EngineCore stop through the
                    # collector. AsyncLLM converts it into a separate ordered
                    # boundary only after this ordinary completion has been
                    # yielded to the consumer.
                    request_output.streaming_turn_complete = True
                    request_output.streaming_turn_id = completed_turn_id
                    if completed_turn_id is not None:
                        req_state.last_completed_streaming_turn_id = (
                            completed_turn_id
                        )
                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    def _finish_request(self, req_state: RequestState) -> None:
        req_id = req_state.request_id
        self.request_states.pop(req_id)

        internal_ids = self.external_req_ids[req_state.external_req_id]
        internal_ids.remove(req_id)
        if not internal_ids:
            del self.external_req_ids[req_state.external_req_id]

        # Remove parent request if applicable.
        parent_req = req_state.parent_req
        if parent_req and not parent_req.child_requests:
            self.parent_requests.pop(parent_req.request_id, None)

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        self.lora_states.update_scheduler_stats(scheduler_stats)

    def do_tracing(
        self,
        engine_core_output: EngineCoreOutput,
        req_state: RequestState,
        iteration_stats: IterationStats | None,
    ) -> None:
        assert req_state.stats is not None
        assert iteration_stats is not None

        metrics = req_state.stats
        arrival_time_ns = int(metrics.arrival_time * 1e9)
        trace_context = extract_trace_context(engine_core_output.trace_headers)
        prompt_length = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Calculate timing metrics
        e2e_time = iteration_stats.iteration_timestamp - metrics.arrival_time
        queued_time = metrics.scheduled_ts - metrics.queued_ts
        prefill_time = metrics.first_token_ts - metrics.scheduled_ts
        decode_time = metrics.last_token_ts - metrics.first_token_ts
        inference_time = metrics.last_token_ts - metrics.scheduled_ts

        # Build attributes dict
        attributes: dict[str, Any] = {
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: (
                metrics.first_token_latency
            ),
            SpanAttributes.GEN_AI_LATENCY_E2E: e2e_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: queued_time,
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: prompt_length,
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: (
                metrics.num_generation_tokens
            ),
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: prefill_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: decode_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: inference_time,
            SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,
        }

        # Add optional request parameters
        if req_state.top_p:
            attributes[SpanAttributes.GEN_AI_REQUEST_TOP_P] = req_state.top_p
        if req_state.max_tokens_param:
            attributes[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = (
                req_state.max_tokens_param
            )
        if req_state.temperature:
            attributes[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = (
                req_state.temperature
            )
        if req_state.n:
            attributes[SpanAttributes.GEN_AI_REQUEST_N] = req_state.n

        instrument_manual(
            span_name="llm_request",
            start_time=arrival_time_ns,
            attributes=attributes,
            context=trace_context,
            kind=SpanKind.SERVER,
        )

    def _update_stats_from_output(
        self,
        req_state: RequestState,
        engine_core_output: EngineCoreOutput,
        engine_core_timestamp: float | None,
        iteration_stats: IterationStats | None,
    ):
        if iteration_stats is None:
            return

        assert engine_core_timestamp is not None
        assert req_state.stats is not None
        iteration_stats.update_from_output(
            engine_core_output,
            engine_core_timestamp,
            req_state.is_prefilling,
            req_state.stats,
            self.lora_states,
            req_state.lora_name,
        )

    def _update_stats_from_finished(
        self,
        req_state: RequestState,
        finish_reason: FinishReason | None,
        iteration_stats: IterationStats | None,
    ):
        if iteration_stats is None:
            return

        assert finish_reason is not None
        assert req_state.stats is not None
        iteration_stats.update_from_finished_request(
            finish_reason=finish_reason,
            request_id=req_state.external_req_id,
            num_prompt_tokens=req_state.prompt_len,
            max_tokens_param=req_state.max_tokens_param,
            req_stats=req_state.stats,
            num_cached_tokens=req_state.num_cached_tokens,
        )
        self.lora_states.request_finished(req_state.request_id, req_state.lora_name)

        ParentRequest.observe_finished_request(
            req_state.parent_req, iteration_stats, req_state.stats.num_generation_tokens
        )

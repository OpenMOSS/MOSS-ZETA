# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from collections.abc import Sequence as GenericSequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Final

import msgspec
import partial_json_parser
import regex as re
from fastapi import Request
from partial_json_parser.core.options import Allow

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    get_history_tool_calls_cnt,
    get_tool_call_id_type,
    make_tool_call_id,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.stream_harmony import (
    TokenState,
    extract_harmony_streaming_delta,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    FunctionCall,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    GenerationError,
    OpenAIServing,
    clamp_prompt_logprobs,
)
from vllm.entrypoints.openai.lumine_ocr import (
    OCR_ARCHIVE_HEADERS,
    coalesce_ocr_history,
    inline_ocr_blocks,
    inline_ocr_message,
    inline_ocr_token_ranges,
)
from vllm.entrypoints.openai.lumine_session import (
    LumineSessionError,
    LumineSessionManager,
    aggregate_delta_outputs,
    prompt_fingerprint,
    sampling_params_payload,
    slice_lumine_continuation,
)
from vllm.entrypoints.openai.lumine_training import (
    TrainingBusyError,
    TrainingSessions,
    TrainingTurn,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant,
    parse_chat_output,
)
from vllm.entrypoints.openai.utils import maybe_filter_parallel_tool_calls
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams
from vllm.sampling_params import BeamSearchParams, RequestOutputKind, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.utils import partial_json_loads
from vllm.utils.collection_utils import as_list
from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser

if TYPE_CHECKING:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

logger = init_logger(__name__)

@dataclass(frozen=True)
class _LumineObservationTurn:
    start: int
    image: int
    end: int
    ocr: int | None
    retain_ocr: bool
    completed: bool


@dataclass(frozen=True)
class _LumineRenderLayout:
    messages: list[dict[str, Any]]
    turns: list[_LumineObservationTurn]


def _lumine_message_text(message: Mapping[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping) or part.get("type") not in (
            "text",
            "input_text",
        ):
            return None
        text = part.get("text")
        if not isinstance(text, str):
            return None
        parts.append(text)
    return "".join(parts)


def _lumine_is_ocr_message(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    text = _lumine_message_text(message)
    if text is None:
        return False
    normalized = text.strip()
    for header in OCR_ARCHIVE_HEADERS:
        normalized = normalized.removeprefix(header).strip()
    return normalized.startswith("<screen_ocr>") and normalized.endswith(
        "</screen_ocr>"
    )


def _lumine_retain_ocr(message: Mapping[str, Any]) -> bool:
    return _lumine_is_ocr_message(message) and bool(inline_ocr_blocks(message))


def _lumine_is_current_ocr_message(message: Mapping[str, Any]) -> bool:
    """Distinguish a legacy OCR/image pair from the training history archive.

    The archive belongs to the cold-start sink. Re-sending it with one current
    image in an existing session must not append the historical events again.
    """
    text = _lumine_message_text(message)
    return (
        _lumine_is_ocr_message(message)
        and text is not None
        and not text.strip().startswith(OCR_ARCHIVE_HEADERS)
    )


def _lumine_has_image(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if part_type in ("image", "image_url", "input_image"):
            return True
        if "image_url" in part:
            return True
    return False


def _lumine_observation_turns(
    messages: list[dict[str, Any]],
) -> list[_LumineObservationTurn]:
    partial_turns: list[tuple[int, int, int | None, bool]] = []
    for image_index, message in enumerate(messages):
        if not _lumine_has_image(message):
            continue
        ocr_index = image_index - 1
        if ocr_index < 0 or not _lumine_is_current_ocr_message(messages[ocr_index]):
            ocr_index = None
        start = image_index if ocr_index is None else ocr_index
        retain_ocr = (
            ocr_index is not None and _lumine_retain_ocr(messages[ocr_index])
        )
        partial_turns.append((start, image_index, ocr_index, retain_ocr))

    turns: list[_LumineObservationTurn] = []
    for index, (start, image, ocr, retain_ocr) in enumerate(partial_turns):
        end = (
            partial_turns[index + 1][0]
            if index + 1 < len(partial_turns)
            else len(messages)
        )
        completed = any(
            message.get("role") == "assistant" for message in messages[image + 1 : end]
        )
        turns.append(
            _LumineObservationTurn(
                start=start,
                image=image,
                end=end,
                ocr=ocr,
                retain_ocr=retain_ocr,
                completed=completed,
            )
        )
    return turns


def _serialize_stream_chunk(
    chunk: ChatCompletionStreamResponse,
    *,
    exclude_none: bool = False,
) -> tuple[str, dict[str, Any]]:
    def dump() -> str:
        return chunk.model_dump_json(
            exclude_unset=True,
            exclude_none=exclude_none,
        )

    started = time.perf_counter()
    data = dump()
    baseline_ms = round((time.perf_counter() - started) * 1000.0, 3)
    baseline_bytes = len(data.encode("utf-8"))
    if envs.VLLM_ENABLE_REQUEST_TIMING:
        # Measure the response-object serialization without the diagnostic
        # field.  Then solve the self-referential byte count by a tiny bounded
        # fixed-point iteration; the final value is the exact JSON payload
        # that is yielded by this helper.
        timing = {
            "scope": "server_chunk_serialization",
            "server_chunk_serialize_ms": baseline_ms,
            "server_chunk_payload_bytes": baseline_bytes,
            "server_chunk_bytes": 0,
        }
        for _ in range(4):
            chunk.server_output_timing = timing
            data = dump()
            final_bytes = len(data.encode("utf-8"))
            if final_bytes == timing["server_chunk_bytes"]:
                break
            timing["server_chunk_bytes"] = final_bytes
        chunk.server_output_timing = timing
        data = dump()
    return data, {
        "scope": "server_chunk_serialization",
        "server_chunk_serialize_ms": baseline_ms,
        "server_chunk_bytes": len(data.encode("utf-8")),
    }


def _instrument_nonstream_response(response: ChatCompletionResponse) -> ChatCompletionResponse:
    """Attach an in-handler serialization diagnostic to a full response.

    FastAPI/Starlette serializes the returned model after this handler exits,
    so that framework/socket work cannot be measured here.  The diagnostic is
    intentionally named and scoped as an *instrumented* response-object dump;
    it is useful for comparing response construction/serialization but is not
    a wire-level egress measurement.
    """
    if not envs.VLLM_ENABLE_REQUEST_TIMING:
        return response

    started = time.perf_counter()
    payload = response.model_dump_json(exclude_unset=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.server_output_timing = {
        "scope": "server_nonstream_response_serialization",
        "server_response_serialize_ms": round(elapsed_ms, 3),
        # This is the payload size before adding the diagnostic itself.  A
        # second dump below records the final instrumented object size.
        "server_response_payload_bytes": len(payload.encode("utf-8")),
        "server_response_bytes": 0,
    }
    for _ in range(4):
        final_payload = response.model_dump_json(exclude_unset=True)
        final_bytes = len(final_payload.encode("utf-8"))
        if final_bytes == response.server_output_timing["server_response_bytes"]:
            break
        response.server_output_timing["server_response_bytes"] = final_bytes
    return response


class OpenAIServingChat(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        openai_serving_render: "OpenAIServingRender",
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        enable_log_deltas: bool = True,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        lumine_session_manager: LumineSessionManager | None = None,
        lumine_context_format: str | None = None,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.openai_serving_render = openai_serving_render
        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.enable_log_outputs = enable_log_outputs
        self.enable_log_deltas = enable_log_deltas
        self.lumine_session_manager = lumine_session_manager
        self.lumine_training = None
        context_format = lumine_context_format if lumine_context_format is not None else os.getenv("VLLM_LUMINE_CONTEXT_FORMAT", "legacy")
        if context_format not in ("legacy", "v0930_v8_9"):
            raise ValueError("Unknown VLLM_LUMINE_CONTEXT_FORMAT")
        if context_format == "v0930_v8_9":
            if lumine_session_manager is None:
                raise ValueError("v8.9 requires Lumine sessions")
            # The frontend rebuilds the canonical aggregate before round 6.
            # Prevent the engine's legacy 5/3 trigger from evicting at round 5.
            lumine_session_manager.high_watermark = 6
            lumine_session_manager.window_size = 6
            lumine_session_manager.retain_turns = 4
            self.lumine_training = TrainingSessions(
                idle_timeout_s=lumine_session_manager.idle_timeout_s
            )

        # set up reasoning parser
        self.reasoning_parser_cls = ParserManager.get_reasoning_parser(
            reasoning_parser_name=reasoning_parser
        )
        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        self.tool_parser = ParserManager.get_tool_parser(
            tool_parser_name=tool_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
        )
        self.parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
        )
        if (
            is_mistral_tool_parser(self.tool_parser)
            and self.reasoning_parser_cls is not None
        ):
            from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

            MistralToolParser.model_can_reason = True

        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
        self.use_harmony = self.model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        self.tool_call_id_type = get_tool_call_id_type(self.model_config)

        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

    def warmup(self) -> None:
        self.renderer.warmup(
            ChatParams(
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                chat_template_kwargs=self.default_chat_template_kwargs,
            )
        )

    def _effective_chat_template_kwargs(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """
        Validate the model and preprocess a chat completion request.

        Delegates preprocessing logic to OpenAIServingRender, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A tuple of (conversation, engine_inputs) on success,
            or an ErrorResponse on failure.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        return await self.openai_serving_render.render_chat(request)

    async def create_chat_completion(self, request, raw_request=None):
        from lumine_protocol import COMPACT
        from lumine_protocol.http import complete_body, complete_stream
        training = getattr(self, "lumine_training", None)
        profile = getattr(self, "lumine_eval_profile", None) or (training.protocol.render_profile if training else None)
        if not profile or profile["version"] != COMPACT:
            return await OpenAIServingChat._create_protocol_completion(self, request, raw_request)
        if request.n not in (None, 1) or request.tools or request.echo or request.continue_final_message or request.add_generation_prompt is False:
            return self.create_error_response("Compact Policy protocol requires n=1, no tools/echo/continue_final_message")
        if self.reasoning_parser_cls is not None:
            return self.create_error_response("Compact protocol requires raw assistant output; disable reasoning parser")
        enabled = (request.chat_template_kwargs or {}).get("enable_thinking", True)
        if type(enabled) is not bool:
            return self.create_error_response("enable_thinking must be boolean")
        request = request.model_copy(update={"return_token_ids": True, "skip_special_tokens": False,
                                             "spaces_between_special_tokens": False})
        result = await OpenAIServingChat._create_protocol_completion(self, request, raw_request)
        if isinstance(result, ErrorResponse):
            return result
        if isinstance(result, ChatCompletionResponse):
            return ChatCompletionResponse.model_validate(complete_body(result.model_dump(mode="json"), profile, enabled))
        return complete_stream(result, profile, enabled)

    async def _create_protocol_completion(self, request, raw_request=None):

        training = getattr(self, "lumine_training", None)
        headers = raw_request.headers if raw_request else {}
        if training is None or headers.get("x-lumine-session-mode", "session").strip().lower() != "session":
            result = await OpenAIServingChat._create_training_completion(self, request, raw_request)
            if training is not None and isinstance(result, ChatCompletionResponse):
                from vllm.entrypoints.openai.lumine_rendering import public_profile
                result.lumine_rendering = public_profile(training.protocol.render_profile)
            if training is not None and not isinstance(result, (ChatCompletionResponse, ErrorResponse)):
                from vllm.entrypoints.openai.lumine_rendering import public_profile
                async def metadata_stream():
                    async for chunk in result:
                        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                            import json
                            payload = json.loads(chunk[6:])
                            payload["lumine_rendering"] = public_profile(training.protocol.render_profile)
                            chunk = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                        yield chunk
                return metadata_stream()
            return result
        from vllm.entrypoints.openai.lumine_requests import RequestConflict
        sid = headers.get("x-lumine-session-id", "").strip()
        rid = headers.get("x-lumine-request-id", "").strip()
        ledger = training.requests
        try:
            cached = ledger.begin(sid, rid, request.model_dump(mode="json", exclude_unset=True),
                                  initial=request.lumine_state is not None,
                                  live=sid in training.busy or (sid in training.contexts and
                                      (training.idle_timeout_s <= 0 or __import__('time').monotonic() - training.contexts[sid].updated_at <= training.idle_timeout_s)))
        except RequestConflict as error:
            return self.create_error_response(str(error), err_type="LumineRequestConflict",
                                              status_code=HTTPStatus.CONFLICT)

        async def replay(chunks):
            for chunk in chunks:
                yield chunk

        if cached is not None:
            if cached["kind"] == "stream":
                return replay(cached["chunks"])
            cls = ErrorResponse if cached["kind"] == "error" else ChatCompletionResponse
            return cls.model_validate(cached["body"])
        try:
            result = await self._create_training_completion(request, raw_request)
            if isinstance(result, (ChatCompletionResponse, ErrorResponse)):
                payload = result.model_dump(mode="json")
                if isinstance(result, ChatCompletionResponse):
                    payload["lumine_session_id"] = sid
                    payload["lumine_request_id"] = rid
                    from vllm.entrypoints.openai.lumine_rendering import public_profile
                    payload["lumine_rendering"] = public_profile(training.protocol.render_profile)
                    result = ChatCompletionResponse.model_validate(payload)
                ledger.finish(sid, rid, dict(kind="error" if isinstance(result, ErrorResponse) else "response", body=payload))
                return result
            async def committed_stream():
                chunks = []
                try:
                    async for chunk in result:
                        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                            import json
                            payload = json.loads(chunk[6:])
                            from vllm.entrypoints.openai.lumine_rendering import public_profile
                            payload.update(lumine_session_id=sid, lumine_request_id=rid, lumine_rendering=public_profile(training.protocol.render_profile))
                            chunk = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                        chunks.append(chunk)
                        # Completion is not acknowledged before history and
                        # durable result are both committed. Token deltas stream.
                        if not chunk.startswith("data: [DONE]"):
                            yield chunk
                    ledger.finish(sid, rid, dict(kind="stream", chunks=chunks))
                    for chunk in chunks:
                        if chunk.startswith("data: [DONE]"):
                            yield chunk
                except BaseException:
                    ledger.fail(sid, rid)
                    raise
                finally:
                    await result.aclose()
            return committed_stream()
        except BaseException:
            ledger.fail(sid, rid)
            raise

    async def _create_training_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        training = getattr(self, "lumine_training", None)
        mode = (
            raw_request.headers.get("x-lumine-session-mode", "session")
            if raw_request
            else "stateless"
        )
        if training is None or mode.strip().lower() != "session":
            if (
                request.lumine_state is not None
                or request.lumine_ocr is not None
                or request.lumine_state_reset
            ):
                return self.create_error_response(
                    "lumine_state/lumine_ocr require the v0930_v8_9 session profile",
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                )
            if training is not None:
                if request.chat_template is not None and request.chat_template != training.protocol.render_profile["template"]:
                    return self.create_error_response("This profile fixes the template for session and stateless requests")
                request = request.model_copy(update={"chat_template_kwargs": {"enable_thinking": (request.chat_template_kwargs or {}).get("enable_thinking", True)}})
            return await self._create_chat_completion_impl(request, raw_request)
        session_id = raw_request.headers.get("x-lumine-session-id", "").strip()
        if not session_id:
            return self.create_error_response("v8.9 requires X-Lumine-Session-ID")
        if request.chat_template is not None or request.add_generation_prompt is False:
            return self.create_error_response(
                "v8.9 uses the server's fixed chat template and generation prefix"
            )
        try:
            turn = training.begin(
                session_id,
                request.messages,
                state=request.lumine_state,
                raw_ocr=request.lumine_ocr,
                force_reset=request.lumine_state_reset,
            )
        except (ValueError, TypeError, KeyError) as error:
            return self.create_error_response(
                str(error),
                err_type="LumineTrainingContextError",
                status_code=HTTPStatus.CONFLICT
                if isinstance(error, TrainingBusyError)
                else HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        turn.enable_thinking = (request.chat_template_kwargs or {}).get("enable_thinking", True)
        manager = self.lumine_session_manager
        assert manager is not None
        handed_off = False

        async def finish(commit):
            try:
                if not commit:
                    await manager.reset(session_id)
            finally:
                training.end(session_id, turn, commit=commit)

        try:
            exists = await manager.has_session(session_id)
            if turn.reason or not exists:
                turn.reason = turn.reason or "engine_cache_miss"
                await manager.reset(session_id)
                incremental = False
            else:
                incremental = True
            request = request.model_copy(
                update={
                    "messages": turn.messages(incremental=False),
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                    "chat_template_kwargs": {"enable_thinking": (request.chat_template_kwargs or {}).get("enable_thinking", True)},
                }
            )
            result = await self._create_chat_completion_impl(
                request, raw_request, training_turn=turn
            )
            if isinstance(result, (ChatCompletionResponse, ErrorResponse)):
                await finish(
                    isinstance(result, ChatCompletionResponse) and turn.completed
                )
                handed_off = True
                return result

            async def stream():
                succeeded = False
                try:
                    async for chunk in result:
                        yield chunk
                    succeeded = turn.completed
                finally:
                    try:
                        await result.aclose()
                    finally:
                        await finish(succeeded)

            handed_off = True
            return stream()
        except (ValueError, TypeError, KeyError) as error:
            return self.create_error_response(
                str(error),
                err_type="LumineTrainingContextError",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        finally:
            if not handed_off:
                await finish(False)

    async def _create_chat_completion_impl(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
        *,
        training_turn: TrainingTurn | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        request_timing_started = time.perf_counter()
        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        lumine_session_id = None
        lumine_render_layout: _LumineRenderLayout | None = None
        if raw_request is not None:
            session_mode = (
                raw_request.headers.get("x-lumine-session-mode", "session")
                .strip()
                .lower()
            )
            if session_mode not in ("session", "stateless"):
                return self.create_error_response(
                    "X-Lumine-Session-Mode must be 'session' or 'stateless'.",
                    err_type="LumineRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                    param="X-Lumine-Session-Mode",
                )
            if session_mode == "session":
                lumine_session_id = (
                    raw_request.headers.get("x-lumine-session-id", "").strip()
                    or "default-single-agent"
                )

        render_request = request
        if lumine_session_id and self.lumine_session_manager is not None:
            try:
                has_session = getattr(self.lumine_session_manager, "has_session", None)
                session_exists = (
                    await has_session(lumine_session_id)
                    if callable(has_session)
                    else True
                )
                if training_turn is not None:
                    rendered_messages = request.messages
                    # The fixed aggregate is always a sink message, even when
                    # empty; it must never be paired with the first image.
                    image_indices = [
                        i
                        for i, m in enumerate(rendered_messages)
                        if _lumine_has_image(m)
                    ]
                    turns = [
                        _LumineObservationTurn(
                            start=i,
                            image=i,
                            end=image_indices[k + 1]
                            if k + 1 < len(image_indices)
                            else len(rendered_messages),
                            ocr=None,
                            retain_ocr=False,
                            completed=k + 1 < len(image_indices),
                        )
                        for k, i in enumerate(image_indices)
                    ]
                else:
                    rendered_messages = self._lumine_render_messages(
                        request,
                        existing_session=session_exists,
                        high_watermark=getattr(
                            self.lumine_session_manager, "high_watermark", 5
                        ),
                        retain_turns=getattr(
                            self.lumine_session_manager, "retain_turns", 3
                        ),
                    )
                    turns = _lumine_observation_turns(rendered_messages)
                lumine_render_layout = _LumineRenderLayout(
                    rendered_messages,
                    turns,
                )
                render_request = request.model_copy(
                    update={"messages": rendered_messages}
                )
            except ValueError as error:
                return self.create_error_response(
                    str(error),
                    err_type="LumineUnsupportedPrompt",
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                )
        reasoning_parser: ReasoningParser | None = None
        if self.reasoning_parser_cls:
            reasoning_parser = self.reasoning_parser_cls(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
            )
        render_started = time.perf_counter()
        result = await self.render_chat_request(render_request)
        render_request_ms = round((time.perf_counter() - render_started) * 1000.0, 3)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_inputs = result
        profile = getattr(self, "lumine_eval_profile", None)
        training = getattr(self, "lumine_training", None)
        profile = profile or (training.protocol.render_profile if training else None)
        if profile and profile["version"] == "compact-think-v1":
            from lumine_protocol import ROLE_HEAD, prefill
            expected = tokenizer.encode(ROLE_HEAD + prefill(profile, (request.chat_template_kwargs or {}).get("enable_thinking", True)), add_special_tokens=False)
            for engine_input in engine_inputs:
                actual = list(self._extract_prompt_components(engine_input).token_ids)
                if actual[-len(expected):] != expected:
                    return self.create_error_response("Actual inference tokens disagree with protocol prefill")
        request_preprocess = (
            engine_inputs[0].get("request_preprocess")
            if len(engine_inputs) == 1
            else None
        )
        if request_preprocess is not None:
            request_preprocess = dict(request_preprocess)
            request_preprocess.update(
                {
                    "server_render_request_ms": render_request_ms,
                    "server_preprocess_to_engine_ms": round(
                        (time.perf_counter() - request_timing_started) * 1000.0,
                        3,
                    ),
                }
            )

        if training_turn is not None:
            request_preprocess = dict(request_preprocess or {})
            request_preprocess.update(
                {
                    "training_context_format": "v0930_v8_9",
                    "training_context_rebuild_reason": training_turn.reason or None,
                    "training_context_incremental": not bool(training_turn.reason),
                    "training_context_visible_images": len(training_turn.context.turns)
                    + 1,
                }
            )

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)
        model_name = self.models.model_name(lora_request)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)
        trace_headers = (
            None
            if raw_request is None
            else await self._get_trace_headers(raw_request.headers)
        )

        if lumine_session_id and self.lumine_session_manager is not None:
            return await self._create_lumine_chat_completion(
                request=request,
                raw_request=raw_request,
                conversation=conversation,
                engine_inputs=engine_inputs,
                request_id=f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}",
                model_name=model_name,
                chat_template_kwargs=chat_template_kwargs,
                reasoning_parser=reasoning_parser,
                session_id=lumine_session_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                data_parallel_rank=data_parallel_rank,
                request_preprocess=request_preprocess,
                lumine_render_layout=lumine_render_layout,
                training_turn=training_turn,
            )

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, engine_input in enumerate(engine_inputs):
            prompt_token_ids = self._extract_prompt_components(engine_input).token_ids

            # If we are creating sub requests for multiple prompts, ensure that they
            # have unique request ids.
            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
            )

            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )

            self._log_inputs(
                sub_request_id,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_input,
                    request_id=sub_request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                if not request.include_reasoning:
                    reasoning_ended = True
                elif request._grammar_from_tool_parser:
                    # The Mistral grammar already includes an optional
                    # `think?` rule that handles both reasoning and
                    # non-reasoning outputs.
                    reasoning_ended = True
                elif reasoning_parser:
                    reasoning_ended = reasoning_parser.is_reasoning_end(
                        prompt_token_ids or []
                    )
                else:
                    reasoning_ended = None

                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=reasoning_ended,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": chat_template_kwargs,
                    }
                    if reasoning_parser
                    else None,
                )

            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
                chat_template_kwargs=chat_template_kwargs,
                request_preprocess=request_preprocess,
            )

        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            reasoning_parser,
            request_preprocess=request_preprocess,
        )

    async def _create_lumine_chat_completion(
        self,
        *,
        request: ChatCompletionRequest,
        raw_request: Request | None,
        conversation: list[ConversationMessage],
        engine_inputs: list[EngineInput],
        request_id: str,
        model_name: str,
        chat_template_kwargs: dict[str, Any],
        reasoning_parser: ReasoningParser | None,
        session_id: str,
        lora_request: LoRARequest | None,
        trace_headers: Mapping[str, str] | None,
        data_parallel_rank: int | None,
        request_preprocess: dict[str, Any] | None,
        lumine_render_layout: _LumineRenderLayout | None,
        training_turn: TrainingTurn | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """Submit one OpenAI chat request to a persistent Lumine session."""
        if len(engine_inputs) != 1:
            return self.create_error_response(
                "Lumine sessions support exactly one prompt per request.",
                err_type="LumineUnsupportedRequest",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        if request.n not in (None, 1):
            return self.create_error_response(
                "Lumine sessions support n=1 only.",
                err_type="LumineUnsupportedRequest",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                param="n",
            )
        if request.use_beam_search or request.stop:
            return self.create_error_response(
                "Lumine sessions do not support beam search or stop strings.",
                err_type="LumineUnsupportedRequest",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        if (
            request.tools
            or request.response_format is not None
            or request.structured_outputs is not None
        ):
            return self.create_error_response(
                "Lumine sessions currently support plain text chat only; "
                "tools and structured outputs are not supported.",
                err_type="LumineUnsupportedRequest",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        engine_input = engine_inputs[0]
        prompt_token_ids = self._extract_prompt_components(engine_input).token_ids
        max_tokens = get_max_tokens(
            self.model_config.max_model_len,
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens,
            self._extract_prompt_len(engine_input),
            self.default_sampling_params,
            self.override_max_tokens,
        )
        sampling_params = request.to_sampling_params(
            max_tokens,
            self.default_sampling_params,
        )
        if not isinstance(sampling_params, SamplingParams):
            return self.create_error_response(
                "Lumine sessions require sampling parameters.",
                err_type="LumineUnsupportedRequest",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        sampling_params.output_kind = RequestOutputKind.DELTA
        if sampling_params.stop:
            return self.create_error_response(
                "Lumine sessions do not support stop strings.",
                err_type="LumineUnsupportedRequest",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                param="stop",
            )

        tokenizer = self.renderer.tokenizer
        if tokenizer is None:
            return self.create_error_response(
                "Lumine sessions require a tokenizer.",
                err_type="LumineUnsupportedRequest",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        try:
            prompt_ids = list(prompt_token_ids)
            if training_turn is not None:
                generation_prefix = tokenizer.encode(
                    "<|im_start|>assistant\n" + __import__("lumine_protocol").prefill(training_turn.render_profile, training_turn.enable_thinking), add_special_tokens=False
                )
                if prompt_ids[-len(generation_prefix):] != generation_prefix:
                    raise ValueError("Generation prefix disagrees with the configured rendering profile")
                training_turn.tokenizer = tokenizer
                training_turn.prefix_ids = generation_prefix
            message_starts: dict[int, int] = {}
            cursor = 0
            user_role_prefix = tokenizer.encode(
                "<|im_start|>user\n", add_special_tokens=False
            )
            if lumine_render_layout is not None:
                for index, message in enumerate(lumine_render_layout.messages):
                    if message.get("role") != "user":
                        continue
                    message_start = self._find_subsequence(
                        prompt_ids, user_role_prefix, cursor
                    )
                    if message_start is None:
                        raise ValueError(
                            "expected chat-template message prefix was not found"
                        )
                    message_starts[index] = message_start
                    cursor = message_start + len(user_role_prefix)

            turns = (
                lumine_render_layout.turns
                if lumine_render_layout is not None
                else []
            )
            if turns:
                if turns[-1].completed:
                    raise ValueError(
                        "the latest Lumine observation must not already have an "
                        "assistant response"
                    )
                if any(not turn.completed for turn in turns[:-1]):
                    raise ValueError(
                        "historical Lumine observations must have assistant responses"
                    )
                sink_token_count = message_starts[turns[0].start]
                initial_turn_spans = [
                    (
                        message_starts[turn.start],
                        message_starts[turns[index + 1].start],
                        (
                            message_starts[turn.image]
                            if turn.retain_ocr
                            else message_starts[turn.start]
                        ),
                    )
                    for index, turn in enumerate(turns[:-1])
                ]
                active_turn = turns[-1]
                active_turn_start = message_starts[active_turn.start]
                active_retainable_prefix_end = (
                    message_starts[active_turn.image]
                    if active_turn.retain_ocr
                    else active_turn_start
                )
            else:
                active_turn_start = self._find_token_subsequence(
                    prompt_ids, user_role_prefix
                )
                sink_token_count = active_turn_start
                active_retainable_prefix_end = active_turn_start
                initial_turn_spans = []
            boundary_token_ids = tokenizer.encode(
                "<|im_end|>\n", add_special_tokens=False
            )
            retained_token_ranges: list[tuple[int, int]] = []
            if lumine_render_layout is not None and training_turn is None:
                for turn in turns:
                    message = lumine_render_layout.messages[turn.image]
                    blocks = inline_ocr_blocks(message)
                    if blocks:
                        retained_token_ranges.extend(
                            inline_ocr_token_ranges(
                                prompt_ids,
                                message_starts[turn.image],
                                message_starts.get(turn.end, len(prompt_ids)),
                                tokenizer,
                                blocks,
                            )
                        )
            continuation_prompt = slice_lumine_continuation(
                engine_input, active_turn_start
            )
        except Exception as error:
            return self.create_error_response(
                f"Unable to prepare Lumine continuation prompt: {error}",
                err_type="LumineUnsupportedPrompt",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        config_fingerprint = prompt_fingerprint(
            {
                "model": model_name,
                "render_profile": training_turn.render_profile if training_turn is not None else None,
                "chat_template": request.chat_template,
                "chat_template_kwargs": chat_template_kwargs,
                "media_io_kwargs": request.media_io_kwargs,
                "mm_processor_kwargs": request.mm_processor_kwargs,
                "sampling_params": sampling_params_payload(sampling_params),
                "logprobs": request.logprobs,
                "top_logprobs": request.top_logprobs,
                "include_reasoning": request.include_reasoning,
                "reasoning_effort": request.reasoning_effort,
                "cache_salt": request.cache_salt,
                "lora_request": self._lumine_lora_payload(lora_request),
                "priority": request.priority,
                "data_parallel_rank": data_parallel_rank,
            }
        )
        try:
            assert self.lumine_session_manager is not None
            if training_turn is not None:
                from lumine_protocol import can_continue, media_identity
                c = training_turn.context
                current_media = media_identity(engine_input)
                reusable = (can_continue(prompt_ids, current_media, c.engine_tokens, c.engine_media)
                            and c.engine_fingerprint == config_fingerprint)
                if await self.lumine_session_manager.has_session(session_id) and not reusable:
                    await self.lumine_session_manager.reset(session_id)
                    training_turn.reason = training_turn.reason or "token_media_or_config_mismatch"
                training_turn.engine_tokens = prompt_ids
                training_turn.engine_media = current_media
                training_turn.engine_fingerprint = config_fingerprint
            session = await self.lumine_session_manager.get_or_create(
                session_id=session_id,
                request_id=f"lumine-{session_id}",
                first_prompt=engine_input,
                initial_turn_spans=initial_turn_spans,
                first_retainable_token_ranges=retained_token_ranges,
                active_turn_start=active_turn_start,
                active_retainable_prefix_end=active_retainable_prefix_end,
                continuation_boundary_tokens=boundary_token_ids,
                sink_token_count=sink_token_count,
                sampling_params=sampling_params,
                config_fingerprint=config_fingerprint,
                cache_salt=request.cache_salt,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
                data_parallel_rank=data_parallel_rank,
            )
            turn = await session.reserve_turn(
                continuation_prompt=continuation_prompt,
                sampling_params=sampling_params,
                config_fingerprint=config_fingerprint,
                retainable_prefix_tokens=(
                    active_retainable_prefix_end - active_turn_start
                ),
                retainable_token_ranges=[
                    (start - active_turn_start, end - active_turn_start)
                    for start, end in retained_token_ranges
                    if start >= active_turn_start
                ],
            )
        except LumineSessionError as error:
            return self.create_error_response(
                str(error),
                err_type=error.err_type,
                status_code=error.status_code,
                param=error.param,
            )

        async def released_outputs() -> AsyncIterator[RequestOutput]:
            try:
                async for output in turn.outputs():
                    if training_turn is not None:
                        training_turn.observe_output(output)
                    yield output
            finally:
                await session.release_turn(turn)

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata
        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                released_outputs(),
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
                chat_template_kwargs=chat_template_kwargs,
                request_preprocess=(
                    {**(turn.request_preprocess or {}), **(request_preprocess or {})}
                    or None
                ),
            )
        return await self.chat_completion_full_generator(
            request,
            aggregate_delta_outputs(released_outputs()),
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            reasoning_parser,
            request_preprocess=(
                {**(turn.request_preprocess or {}), **(request_preprocess or {})}
                or None
            ),
        )

    @staticmethod
    def _lumine_instruction_messages(
        request: ChatCompletionRequest,
    ) -> list[dict[str, Any]]:
        """Return stable instructions while ignoring the caller's history window."""

        return [
            dict(message)
            for message in request.messages
            if message.get("role") in ("system", "developer")
        ]

    @classmethod
    def _lumine_render_messages(
        cls,
        request: ChatCompletionRequest,
        *,
        existing_session: bool = True,
        high_watermark: int = 5,
        retain_turns: int = 3,
    ) -> list[dict[str, Any]]:
        """Normalize full or incremental caller history for a Lumine session."""
        if existing_session:
            latest_image_entry = next(
                (
                    (index, request.messages[index])
                    for index in range(len(request.messages) - 1, -1, -1)
                    if _lumine_has_image(request.messages[index])
                ),
                None,
            )
            if latest_image_entry is not None:
                image_index, latest_image = latest_image_entry
                latest_ocr = image_index - 1
                current_messages = [latest_image]
                if latest_ocr >= 0 and _lumine_is_current_ocr_message(
                    request.messages[latest_ocr]
                ):
                    current_messages.insert(0, request.messages[latest_ocr])
                return [
                    *cls._lumine_instruction_messages(request),
                    *(dict(message) for message in current_messages),
                ]
            last_user = next(
                (
                    dict(message)
                    for message in reversed(request.messages)
                    if message.get("role") == "user"
                ),
                None,
            )
            if last_user is not None:
                return [*cls._lumine_instruction_messages(request), last_user]

        messages = [dict(message) for message in request.messages]
        turns = _lumine_observation_turns(messages)
        if not turns and not any(
            message.get("role") == "user" for message in messages
        ):
            raise ValueError("a Lumine turn must contain a user message")
        if len(turns) < high_watermark:
            return messages
        if not 1 <= retain_turns < high_watermark:
            raise ValueError(
                "Lumine window configuration must satisfy "
                "1 <= retain_turns < high_watermark"
            )

        first_retained = turns[-retain_turns].start
        retained_ocr = []
        for message in messages[:first_retained]:
            if _lumine_retain_ocr(message):
                retained_ocr.append(message)
            elif _lumine_has_image(message) and (
                archived := inline_ocr_message(message)
            ):
                retained_ocr.append(archived)
        retained_history = messages[first_retained:]
        return [
            *cls._lumine_instruction_messages(request),
            *coalesce_ocr_history(retained_ocr),
            *[
                message
                for message in retained_history
                if message.get("role") not in ("system", "developer")
            ],
        ]

    @staticmethod
    def _lumine_lora_payload(
        lora_request: LoRARequest | None,
    ) -> dict[str, Any] | None:
        if lora_request is None:
            return None
        return {
            "lora_name": lora_request.lora_name,
            "lora_int_id": lora_request.lora_int_id,
            "lora_path": lora_request.lora_path,
            "base_model_name": lora_request.base_model_name,
        }

    @staticmethod
    def _find_subsequence(
        values: list[int], needle: list[int], start: int = 0
    ) -> int | None:
        if not needle:
            return None
        for index in range(start, len(values) - len(needle) + 1):
            if values[index : index + len(needle)] == needle:
                return index
        return None

    @classmethod
    def _find_token_subsequence(cls, values: list[int], needle: list[int]) -> int:
        index = cls._find_subsequence(values, needle)
        if index is None:
            raise ValueError("expected chat-template token prefix was not found")
        return index

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    @staticmethod
    def _bracket_level(s: str, opening="{", closing="}") -> int:
        """
        Calculate the current level of nested brackets in a given string.
        """
        level = 0
        for char in s:
            if char == opening:
                level += 1
            elif char == closing:
                level -= 1
        return level

    @staticmethod
    def _filter_delta_text(delta_text: str, previous_text: str) -> tuple[str, bool]:
        # remove last '},' of the tool definition stemming from the
        # "name"/"parameters" outer object or closing ']' of the tool list
        # count occurrences of opening and closing curly braces and
        # once level 0 is reached stop outputting text
        # if 0 is reached while parsing the delta_text we know the current
        # tool will finish in this current iteration
        bracket_level = OpenAIServingChat._bracket_level(previous_text)
        updated_delta, passed_zero = "", False
        for c in delta_text:
            if c == "{":
                bracket_level += 1
                passed_zero = bracket_level == 0
            elif c == "}":
                bracket_level -= 1
                passed_zero = bracket_level == 0

            if bracket_level != 0:
                updated_delta += c
            else:
                # if a comma is reached at level 0 we can stop
                if c == ",":
                    break
        return updated_delta, passed_zero

    def extract_tool_call_required_streaming(
        self,
        previous_text: str,
        current_text: str | None,
        delta_text: str,
        function_name_returned: bool,
        tool_call_idx: int | None = None,
    ) -> tuple[DeltaMessage | None, bool]:
        if current_text is None or current_text == "":
            # if the current text is empty, we cannot parse it
            return None, function_name_returned
        try:
            flags = Allow.ALL
            obj, _ = partial_json_loads(current_text, flags)
        except (
            partial_json_parser.core.exceptions.MalformedJSON,
            json.JSONDecodeError,
        ):
            logger.debug("not enough tokens to parse into JSON yet")
            obj = None

        # check if the current text is a valid array
        # containing a partial tool calling object
        # if not repeat
        if obj is None or not isinstance(obj, list) or not len(obj) > 0:
            function_name_returned = False
            delta_message = None
        else:
            _, finishes_previous_tool = OpenAIServingChat._filter_delta_text(
                delta_text, previous_text
            )
            # take the last tool call from the generated list
            current_tool_call = obj[-1]

            # once parameters have been generated the name is complete as well
            if not finishes_previous_tool and (
                "name" not in current_tool_call or "parameters" not in current_tool_call
            ):
                function_name_returned = False
                delta_message = None
            else:
                if not function_name_returned:
                    # get partly generated arguments from the latest tool call
                    param_match = re.search(
                        r'.*"parameters":\s*(.*)', current_text, re.DOTALL
                    )
                    arguments = param_match.group(1) if param_match else ""
                    arguments, _ = OpenAIServingChat._filter_delta_text(
                        arguments, previous_text
                    )

                    # if this iteration finishes a previous tool call but a
                    # new incomplete tool is already generated, take the
                    # previous from the list
                    if finishes_previous_tool and "parameters" not in current_tool_call:
                        current_tool_call = obj[-2]

                    function_name_returned = True
                    tool_call_id = make_tool_call_id(
                        id_type=self.tool_call_id_type,
                        func_name=current_tool_call["name"],
                        idx=tool_call_idx,
                    )
                    delta_message = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                id=tool_call_id,
                                function=DeltaFunctionCall(
                                    name=current_tool_call["name"], arguments=arguments
                                ),
                                index=len(obj) - 1,
                                type="function",
                            )
                        ]
                    )

                else:
                    delta_text, _ = OpenAIServingChat._filter_delta_text(
                        delta_text, previous_text
                    )

                    if delta_text != "":
                        delta_message = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    function=DeltaFunctionCall(
                                        # OpenAI API returns None
                                        # instead of name every time
                                        name=None,
                                        arguments=delta_text,
                                    ),
                                    index=len(obj) - 1,
                                )
                            ]
                        )
                    else:
                        delta_message = None

        return delta_message, function_name_returned

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        request_preprocess: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        if self.use_harmony:
            harmony_parsers = [
                get_streamable_parser_for_assistant() for _ in range(num_choices)
            ]
            harmony_tools_streamed = [False] * num_choices
        tools_streamed = [False] * num_choices

        is_mistral_grammar_path = request._grammar_from_tool_parser

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request)
        )

        # Determine whether required/named tool_choice should fall back to
        # the auto tool_parser path instead of the standard JSON-based parsing.
        # This happens when the parser declares supports_required_and_named=False
        # (e.g. GLM models that output XML instead of JSON).
        tool_choice_uses_parser = (
            self.tool_parser is not None
            and not self.tool_parser.supports_required_and_named
            and request.tools
            and (
                request.tool_choice == "required"
                or isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
            )
        )

        all_previous_token_ids: list[list[int]] | None
        function_name_returned = [False] * num_choices
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Always track previous_texts for comprehensive output logging
        previous_texts = [""] * num_choices

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if (
            is_mistral_grammar_path
            or tool_choice_auto
            or tool_choice_uses_parser
            or reasoning_parser
        ):
            # These are only required in "auto" tool choice case
            all_previous_token_ids = [[] for _ in range(num_choices)]
            reasoning_end_arr = [False] * num_choices
            prompt_is_reasoning_end_arr: list[bool | None] = [None] * num_choices
        else:
            all_previous_token_ids = None

        try:
            if self.parser_cls is not None:
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer not available when `skip_tokenizer_init=True`"
                    )
                parsers: list[Parser | None] = [
                    self.parser_cls(
                        tokenizer,
                        request.tools,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    for _ in range(num_choices)
                ]
            else:
                parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in parser creation.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(
                                res.prompt_token_ids
                                if request.return_token_ids
                                else None
                            ),
                            request_preprocess=request_preprocess,
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data, _ = _serialize_stream_chunk(chunk)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )

                                data, _ = _serialize_stream_chunk(chunk)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    parser = parsers[i]
                    tool_parser = parser.tool_parser if parser is not None else None

                    if (
                        reasoning_parser
                        and res.prompt_token_ids
                        and prompt_is_reasoning_end_arr[i] is None
                    ):
                        # only check once per choice, because prompt_token_ids
                        # are the same for all deltas in that choice
                        prompt_is_reasoning_end_arr[i] = (
                            reasoning_parser.is_reasoning_end(res.prompt_token_ids)
                        )
                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    if self.use_harmony:
                        harmony_parser = harmony_parsers[i]
                        prev_recipient = harmony_parser.current_recipient

                        # Track accumulated content per token with their state
                        token_states: list[TokenState] = []
                        for token_id in output.token_ids:
                            harmony_parser.process(token_id)
                            token_delta = harmony_parser.last_content_delta or ""
                            token_states.append(
                                TokenState(
                                    harmony_parser.current_channel,
                                    harmony_parser.current_recipient,
                                    token_delta,
                                )
                            )
                        delta_text = "".join(delta for _, _, delta in token_states)
                        cur_channel = harmony_parser.current_channel

                        # handle the case where several tokens where generated at once
                        # including the final token, leading to a delta in the text
                        # but the current channel to be empty (start state)
                        if not cur_channel and delta_text:
                            cur_channel = "final"
                    else:
                        delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    # just update previous_texts and previous_token_ids
                    if (
                        is_mistral_grammar_path
                        or tool_choice_auto
                        or tool_choice_uses_parser
                        or reasoning_parser
                    ):
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        # avoid the None + list error.
                        if previous_token_ids:
                            current_token_ids = previous_token_ids + as_list(
                                output.token_ids
                            )
                        else:
                            current_token_ids = as_list(output.token_ids)

                    if self.use_harmony:
                        delta_message, tools_streamed_flag = (
                            extract_harmony_streaming_delta(
                                harmony_parser=harmony_parser,
                                token_states=token_states,
                                prev_recipient=prev_recipient,
                                include_reasoning=request.include_reasoning,
                            )
                        )
                        harmony_tools_streamed[i] |= tools_streamed_flag
                    # Mistral grammar path: combined reasoning + tool streaming
                    elif is_mistral_grammar_path:
                        from vllm.tool_parsers.mistral_tool_parser import (
                            MistralToolParser,
                        )

                        assert tool_parser is not None
                        assert isinstance(tool_parser, MistralToolParser)
                        assert reasoning_end_arr is not None
                        output_token_ids = as_list(output.token_ids)
                        result = tool_parser.extract_maybe_reasoning_and_tool_streaming(
                            reasoning_parser=reasoning_parser,
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            previous_token_ids=previous_token_ids,
                            current_token_ids=current_token_ids,
                            output_token_ids=output_token_ids,
                            reasoning_ended=reasoning_end_arr[i],
                            prompt_is_reasoning_end=(prompt_is_reasoning_end_arr[i]),
                            request=request,
                        )
                        delta_message = result.delta_message
                        reasoning_end_arr[i] = result.reasoning_ended
                        current_text = result.current_text
                        current_token_ids = result.current_token_ids
                        if result.tools_called:
                            tools_streamed[i] = True
                    # handle streaming deltas for tools with named tool_choice
                    # Skip when tool_choice_uses_parser so it falls through
                    # to the auto tool_parser branches below.
                    elif tool_choice_function_name and not tool_choice_uses_parser:
                        # When encountering think end id in prompt_token_ids
                        # i.e {"enable_thinking": False},
                        # check BEFORE calling the parser to avoid a spurious
                        # reasoning delta on the first chunk.
                        if (
                            reasoning_parser
                            and not reasoning_end_arr[i]
                            and prompt_is_reasoning_end_arr[i]
                        ):
                            reasoning_end_arr[i] = True

                        if (
                            reasoning_parser
                            and not reasoning_end_arr[i]
                            and not reasoning_parser.is_reasoning_end(
                                previous_token_ids
                            )
                        ):
                            assert reasoning_parser is not None
                            delta_message = (
                                reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                )
                            )
                            # When encountering think end id in delta_token_ids,
                            # set reasoning status to end.
                            # Only keep 'content', remove 'reasoning'.
                            if reasoning_parser.is_reasoning_end(
                                as_list(output.token_ids)
                            ):
                                reasoning_end_arr[i] = True
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                        else:
                            # Just to add remaining `content`
                            if reasoning_parser:
                                delta_text = previous_text + delta_text
                                current_text = ""

                            if function_name_returned[i]:
                                delta_tool_call = DeltaToolCall(
                                    function=DeltaFunctionCall(arguments=delta_text),
                                    index=i,
                                )
                            else:
                                # Generate ID based on tokenizer type
                                if is_mistral_tokenizer(tokenizer):
                                    from vllm.tool_parsers.mistral_tool_parser import (
                                        MistralToolCall,
                                    )

                                    tool_call_id = MistralToolCall.generate_random_id()
                                else:
                                    tool_call_id = make_tool_call_id(
                                        id_type=self.tool_call_id_type,
                                        func_name=tool_choice_function_name,
                                        idx=history_tool_call_cnt,
                                    )
                                delta_tool_call = DeltaToolCall(
                                    id=tool_call_id,
                                    type="function",
                                    function=DeltaFunctionCall(
                                        name=tool_choice_function_name,
                                        arguments=delta_text,
                                    ),
                                    index=i,
                                )
                                function_name_returned[i] = True
                                history_tool_call_cnt += 1

                            delta_message = DeltaMessage(
                                tool_calls=[
                                    delta_tool_call,
                                ]
                            )
                            tools_streamed[i] = True

                    # Skip when tool_choice_uses_parser so it falls through
                    # to the auto tool_parser branches below.
                    elif (
                        request.tool_choice == "required"
                        and not tool_choice_uses_parser
                    ):
                        assert previous_texts is not None
                        previous_text = previous_texts[i]
                        current_text = previous_text + delta_text
                        fn_name_returned = function_name_returned[i]
                        output_token_ids = as_list(output.token_ids)

                        if (
                            reasoning_parser is not None
                            and not reasoning_end_arr[i]
                            and prompt_is_reasoning_end_arr[i]
                        ):
                            reasoning_end_arr[i] = True

                        if reasoning_parser and not reasoning_end_arr[i]:
                            delta_message = (
                                reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output_token_ids,
                                )
                            )
                            if reasoning_parser.is_reasoning_end(output_token_ids):
                                reasoning_end_arr[i] = True
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    # reasoning ended
                                    current_text = ""

                        else:
                            # either finished reasoning or no reasoning at all
                            content = current_text

                            delta_message, function_name_returned[i] = (
                                self.extract_tool_call_required_streaming(
                                    previous_text=previous_text,
                                    current_text=content,
                                    delta_text=delta_text,
                                    function_name_returned=fn_name_returned,
                                    tool_call_idx=history_tool_call_cnt,
                                )
                            )
                            if (
                                delta_message
                                and delta_message.tool_calls
                                and delta_message.tool_calls[0].id is not None
                            ):
                                history_tool_call_cnt += 1
                                tools_streamed[i] = True

                    elif parser is not None:
                        delta_message = parser.parse_delta(
                            delta_text=delta_text,
                            delta_token_ids=as_list(output.token_ids),
                            request=request,
                            prompt_token_ids=res.prompt_token_ids,
                        )
                        if delta_message and delta_message.tool_calls:
                            tools_streamed[i] = True
                    # handle streaming just a content delta (no parsers)
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    if (
                        is_mistral_grammar_path
                        or tool_choice_auto
                        or tool_choice_uses_parser
                        or reasoning_parser
                    ) and not self.use_harmony:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids
                    else:
                        # Update for comprehensive logging even in simple case
                        assert previous_texts is not None
                        previous_texts[i] += delta_text

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        # NOTE: If return_token_ids is enabled, we still need to
                        # send a chunk with token_ids even if delta_message is None
                        # to ensure all tokens are included in the response
                        if (
                            output.finish_reason is None
                            and not request.return_token_ids
                        ):
                            continue
                        delta_message = DeltaMessage()

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content_parts = []
                        if delta_message.content:
                            delta_content_parts.append(delta_message.content)
                        if delta_message.reasoning:
                            reasoning = delta_message.reasoning
                            delta_content_parts.append(f"[reasoning: {reasoning}]")
                        if delta_message.tool_calls:
                            tool_args = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments
                            )
                            if tool_args:
                                delta_content_parts.append(f"[tool_calls: {tool_args}]")

                        if delta_content_parts and self.enable_log_deltas:
                            delta_content = " ".join(delta_content_parts)
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                    # if the model is finished generating
                    else:
                        # check for error finish reason and abort streaming
                        # finish_reason='error' indicates a retryable error
                        self._raise_if_error(output.finish_reason, request_id)

                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using structured outputs
                        index = 0
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                            index = (
                                len(tool_parser.prev_tool_call_arr) - 1
                                if auto_tools_called
                                else 0
                            )
                        should_check = (
                            self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output
                            )
                        )
                        # only check if there are any tool calls
                        # detected by partial parsing
                        if should_check and tool_parser and auto_tools_called:
                            latest_delta_len = 0
                            if (
                                isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall,
                                )
                            ) and isinstance(
                                delta_message.tool_calls[0].function.arguments, str
                            ):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.arguments
                                )

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON.
                            # Tool parsers (e.g. Qwen3Coder) store
                            # arguments as a JSON string in
                            # prev_tool_call_arr. Calling json.dumps()
                            # on an already-serialized string would
                            # double-serialize it (e.g. '{"k":1}' becomes
                            # '"{\\"k\\":1}"'), which then causes the
                            # replace() below to fail and append the
                            # entire double-serialized string as a
                            # spurious final delta.
                            args = tool_parser.prev_tool_call_arr[index].get(
                                "arguments", {}
                            )
                            if isinstance(args, str):
                                expected_call = args
                            else:
                                expected_call = json.dumps(args, ensure_ascii=False)

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[index]
                            if latest_delta_len > 0:
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call.replace(actual_call, "", 1)
                            # set that as a delta message
                            delta_message = self._create_remaining_args_delta(
                                delta_message, remaining_call, index
                            )

                        # Send the finish response for each request.n only once
                        # In OpenAI's API, when a tool is called, the
                        # finish_reason is:
                        # "tool_calls" for "auto" or "required" tool calls,
                        # and "stop" for named tool calls.
                        if (
                            auto_tools_called
                            or (tools_streamed[i] and not tool_choice_function_name)
                            or (self.use_harmony and harmony_tools_streamed[i])
                        ):
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                        finish_reason_sent[i] = True

                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                        **(
                            {"vllm_timing": res.timing}
                            if res.timing is not None
                            else {}
                        ),
                        **(
                            {"request_metrics": res.request_metrics}
                            if res.request_metrics is not None
                            else {}
                        ),
                        **(
                            {"timing_events": res.timing_events}
                            if getattr(res, "timing_events", None)
                            else {}
                        ),
                        **(
                            {
                                "streaming_session_stats": msgspec.to_builtins(
                                    res.streaming_session_stats
                                )
                            }
                            if res.streaming_session_stats is not None
                            else {}
                        ),
                    )

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data, _ = _serialize_stream_chunk(chunk)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens
                    )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data, _ = _serialize_stream_chunk(
                    final_usage_chunk,
                    exclude_none=True,
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts)
                        else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except GenerationError as e:
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
        except Exception as e:
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
        request_preprocess: dict[str, Any] | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None
        all_timing_events: list[dict[str, Any]] = []
        first_timing: dict[str, Any] | None = None
        first_request_metrics: dict[str, Any] | None = None

        try:
            async for res in result_generator:
                final_res = res
                if first_timing is None and res.timing is not None:
                    first_timing = dict(res.timing)
                if first_request_metrics is None and res.request_metrics is not None:
                    first_request_metrics = dict(res.request_metrics)
                if getattr(res, "timing_events", None):
                    all_timing_events.extend(
                        item
                        for item in res.timing_events
                        if isinstance(item, dict)
                    )
                    all_timing_events = all_timing_events[-512:]
                if request_preprocess is None and getattr(
                    res, "request_preprocess", None
                ) is not None:
                    request_preprocess = dict(res.request_preprocess)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        if final_res is None:
            return self.create_error_response(
                "No output received from the engine.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        choices: list[ChatCompletionResponseChoice] = []
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            # check for error finish reason and raise GenerationError
            # finish_reason='error' indicates a retryable request-level internal error
            self._raise_if_error(output.finish_reason, request_id)
            token_ids = output.token_ids
            out_logprobs = output.logprobs
            tool_call_info = None

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if self.use_harmony:
                reasoning, content, _ = parse_chat_output(token_ids)
                if not request.include_reasoning:
                    reasoning = None

                if self.tool_parser is not None:
                    if tokenizer is None:
                        raise ValueError(
                            "Tokenizer not available when `skip_tokenizer_init=True`"
                        )

                    tool_parser = self.tool_parser(tokenizer, request.tools)
                    # NOTE: We use token_ids for openai tool parser
                    tool_call_info = tool_parser.extract_tool_calls(
                        "",
                        request=request,
                        token_ids=token_ids,  # type: ignore
                    )
                    content = tool_call_info.content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_info.tool_calls,
                    )
                else:
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                    )

                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    logprobs=logprobs,
                    finish_reason=(
                        "tool_calls"
                        if (tool_call_info is not None and tool_call_info.tools_called)
                        else output.finish_reason
                        if output.finish_reason
                        else "stop"
                    ),
                    stop_reason=output.stop_reason,
                    token_ids=(
                        as_list(output.token_ids) if request.return_token_ids else None
                    ),
                )
                choices.append(choice_data)
                continue

            if reasoning_parser:
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning, content = reasoning_parser.extract_reasoning(
                    output.text, request=request
                )
                if not request.include_reasoning:
                    reasoning = None
            else:
                reasoning = None
                content = output.text

            auto_tools_called = False
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            tool_calls, content = self._parse_tool_calls_from_content(
                request=request,
                tokenizer=tokenizer,
                content=content,
                enable_auto_tools=self.enable_auto_tools,
                tool_parser_cls=self.tool_parser,
            )
            if is_mistral_tokenizer(tokenizer):
                from vllm.tool_parsers.mistral_tool_parser import MistralToolCall

                tool_call_class: type[ToolCall] = MistralToolCall
            else:
                tool_call_class = ToolCall

            use_mistral_tool_parser = request._grammar_from_tool_parser
            if use_mistral_tool_parser:
                from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

                tool_call_items = MistralToolParser.build_non_streaming_tool_calls(
                    tool_calls
                )
                if tool_call_items:
                    auto_tools_called = (
                        request.tool_choice is None or request.tool_choice == "auto"
                    )
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_call_items,
                )

            elif (not self.enable_auto_tools or not self.tool_parser) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            elif (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                assert tool_calls is not None and len(tool_calls) > 0
                tool_call_class_items = []
                for idx, tc in enumerate(tool_calls):
                    # Use native ID if available (e.g., Kimi K2),
                    # otherwise generate ID with correct id_type
                    if tc.id:
                        tool_call_class_items.append(
                            tool_call_class(id=tc.id, function=tc)
                        )
                    else:
                        # Generate ID using the correct format (kimi_k2 or random),
                        # but leave it to the class if it's Mistral to preserve
                        # 9-char IDs
                        if is_mistral_tokenizer(tokenizer):
                            tool_call_class_items.append(tool_call_class(function=tc))
                        else:
                            generated_id = make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tc.name,
                                idx=history_tool_call_cnt,
                            )
                            tool_call_class_items.append(
                                tool_call_class(id=generated_id, function=tc)
                            )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content="",
                    tool_calls=tool_call_class_items,
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class_items = []
                tool_calls = tool_calls or []
                for idx, tool_call in enumerate(tool_calls):
                    # Use native ID if available,
                    # otherwise generate ID with correct id_type
                    if tool_call.id:
                        tool_call_class_items.append(
                            tool_call_class(id=tool_call.id, function=tool_call)
                        )
                    else:
                        # Generate ID using the correct format (kimi_k2 or random),
                        # but leave it to the class if it's Mistral to preserve
                        # 9-char IDs
                        if is_mistral_tokenizer(tokenizer):
                            tool_call_class_items.append(
                                tool_call_class(function=tool_call)
                            )
                        else:
                            generated_id = make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tool_call.name,
                                idx=history_tool_call_cnt,
                            )
                            tool_call_class_items.append(
                                tool_call_class(id=generated_id, function=tool_call)
                            )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=tool_call_class_items,
                    reasoning=reasoning,
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and self.tool_parser
            ):
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_calls is not None and len(tool_calls) > 0
                if tool_calls:
                    tool_call_items = []
                    for idx, tc in enumerate(tool_calls):
                        # Use native ID if available (e.g., Kimi K2),
                        # otherwise generate ID with correct id_type
                        if tc.id:
                            tool_call_items.append(
                                tool_call_class(id=tc.id, function=tc)
                            )
                        else:
                            # Generate ID using the correct format (kimi_k2 or random),
                            # but leave it to the class if it's Mistral to preserve
                            # 9-char IDs
                            if is_mistral_tokenizer(tokenizer):
                                tool_call_items.append(tool_call_class(function=tc))
                            else:
                                generated_id = make_tool_call_id(
                                    id_type=self.tool_call_id_type,
                                    func_name=tc.name,
                                    idx=history_tool_call_cnt,
                                )
                                tool_call_items.append(
                                    tool_call_class(id=generated_id, function=tc)
                                )
                        history_tool_call_cnt += 1
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_items,
                    )

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    ret_content = content

                    # try to use content return from tool parser first,
                    # tool parser may do some modify for the content.
                    if content and len(content) > 0:
                        ret_content = content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=ret_content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = ChatMessage(role=role, reasoning=reasoning, content=content)
            # In OpenAI's API, when a tool is called, the finish_reason is:
            # "tool_calls" for "auto" or "required" tool calls,
            # and "stop" for named tool calls.
            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
                token_ids=(
                    as_list(output.token_ids) if request.return_token_ids else None
                ),
            )
            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)

            choices.append(choice_data)

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )

        request_metadata.final_usage_info = usage

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            kv_transfer_params=final_res.kv_transfer_params,
            **(
                {"vllm_timing": final_res.timing or first_timing}
                if final_res.timing is not None or first_timing is not None
                else {}
            ),
            **(
                {"request_metrics": final_res.request_metrics or first_request_metrics}
                if final_res.request_metrics is not None or first_request_metrics is not None
                else {}
            ),
            **(
                {
                    "request_preprocess": request_preprocess
                    or getattr(final_res, "request_preprocess", None)
                    or (final_res.timing or {}).get("request_preprocess")
                }
                if request_preprocess is not None
                or getattr(final_res, "request_preprocess", None) is not None
                or isinstance((final_res.timing or {}).get("request_preprocess"), dict)
                else {}
            ),
            **(
                {"timing_events": all_timing_events or final_res.timing_events}
                if all_timing_events or getattr(final_res, "timing_events", None)
                else {}
            ),
        )
        if getattr(self, "lumine_training", None) is not None or getattr(self, "lumine_eval_profile", None) is not None:
            response.lumine_engine_outputs = [dict(text=o.text, token_ids=list(o.token_ids)) for o in final_res.outputs]

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:  # type: ignore
                        function_call: FunctionCall = tc.function  # type: ignore
                        tool_call_descriptions.append(
                            f"{function_call.name}({function_call.arguments})"
                        )
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return _instrument_nonstream_response(response)

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: TokenizerLike | None,
        should_return_as_token_id: bool,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            if (top_logprobs and i < top_logprobs or top_logprobs == -1)
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        tokenizer: TokenizerLike | None,
        num_output_top_logprobs: int | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "Unable to get tokenizer because `skip_tokenizer_init=True`"
                        )

                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)

    def _should_stream_with_auto_tool_parsing(self, request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (
            request.tools
            and self.tool_parser
            and self.enable_auto_tools
            and request.tool_choice in ["auto", None]
        )

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: DeltaMessage | None,
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools
            and self.tool_parser
            and delta_message
            and delta_message.tool_calls
            and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )

    @staticmethod
    def _create_remaining_args_delta(
        delta_message: DeltaMessage,
        remaining_call: str,
        index: int,
    ) -> DeltaMessage:
        """
        Create a delta message for remaining tool arguments, preserving
        id/type/name from the original delta.
        """
        original_tc = next(
            (tc for tc in delta_message.tool_calls if tc.index == index),
            None,
        )
        original_fn = original_tc.function if original_tc else None
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    id=original_tc.id if original_tc else None,
                    type=original_tc.type if original_tc else None,
                    function=DeltaFunctionCall(
                        name=original_fn.name if original_fn else None,
                        arguments=remaining_call,
                    ),
                )
            ]
        )

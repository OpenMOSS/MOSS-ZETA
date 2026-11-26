# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.lumine_training import TrainingBusyError
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def batch_chat(request: Request) -> OpenAIServingChatBatch | None:
    return request.app.state.openai_serving_chat_batch


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/chat/completions/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_batch_chat_completion(
    request: BatchChatCompletionRequest, raw_request: Request
):
    handler = batch_chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    result = await handler.create_batch_chat_completion(request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())


@router.post(
    "/v1/lumine/sessions/{session_id}/reset",
    responses={
        HTTPStatus.OK.value: {},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
async def reset_lumine_session(session_id: str, raw_request: Request):
    """Drop the server-side KV/session state for one Lumine session."""

    manager = getattr(raw_request.app.state, "lumine_session_manager", None)
    if manager is None:
        error = ErrorResponse(
            error=ErrorInfo(
                message="Lumine sessions are not enabled.",
                type="NotImplementedError",
                code=HTTPStatus.NOT_IMPLEMENTED,
                param="session_id",
            )
        )
        return JSONResponse(
            content=error.model_dump(), status_code=HTTPStatus.NOT_IMPLEMENTED
        )

    handler = chat(raw_request)
    training = getattr(handler, "lumine_training", None)
    reset_text = False
    if training is not None:
        try:
            reset_text = training.reset(session_id)
        except TrainingBusyError as error:
            return JSONResponse(
                content={
                    "error": {
                        "message": str(error),
                        "type": "LumineSessionBusy",
                        "code": 409,
                    }
                },
                status_code=HTTPStatus.CONFLICT,
            )
    if training is not None:
        training.busy.add(session_id)
    try:
        reset = await manager.reset(session_id) or reset_text
    finally:
        if training is not None:
            training.busy.discard(session_id)
    if not reset:
        error = ErrorResponse(
            error=ErrorInfo(
                message=f"Lumine session '{session_id}' was not found.",
                type="NotFoundError",
                code=HTTPStatus.NOT_FOUND,
                param="session_id",
            )
        )
        return JSONResponse(
            content=error.model_dump(), status_code=HTTPStatus.NOT_FOUND
        )

    return JSONResponse(
        content={"session_id": session_id, "reset": True},
        status_code=HTTPStatus.OK,
    )


def attach_router(app: FastAPI):
    app.include_router(router)


@router.get("/v1/lumine/cache-stats")
async def lumine_cache_stats(raw_request: Request):
    from vllm.multimodal.media.image import image_cache_stats
    training = getattr(chat(raw_request), "lumine_training", None)
    return {"image_decode": image_cache_stats(),
            "requests": training.requests.snapshot() if training else None,
            "active_sessions": len(training.contexts) if training else 0}

# Explicit expert-history endpoint; online contract is unchanged.
from vllm.entrypoints.openai.lumine_eval_session import register as register_eval_route
register_eval_route(router)

"""Authenticated HTTP model deployment for exact-prefix preparation/generation."""
from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
import dataclasses
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import signal
import time
import uuid

from .protocol import digest, image_ids
from .wire import rendering_profile
from .request_journal import RequestJournal
from lumine_protocol import can_continue, media_identity, public_profile, reconstruct


def completion_budget(requested_tokens, prompt_tokens, max_model_len):
    """Use every remaining context token for null, without truncating input."""
    remaining = max_model_len - prompt_tokens
    if remaining < 1:
        raise ValueError(f'Full prompt has {prompt_tokens} tokens; model context limit '
                         f'{max_model_len} leaves no generation space (input was not truncated)')
    return dict(mode='context_remaining' if requested_tokens is None else 'explicit',
                requested_max_tokens=requested_tokens,
                effective_max_tokens=remaining if requested_tokens is None else min(requested_tokens, remaining),
                max_model_len=max_model_len, prompt_tokens=prompt_tokens,
                remaining_context_tokens=remaining, prompt_truncated=False)


def valid_message_tail(messages, prepare):
    """A reset prefix is exactly one nonempty system message; observations
    and retained dialogue continue to use their existing role contract.
    """
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return False
    if prepare and len(messages) == 1 and messages[0].get('role') == 'system':
        content = messages[0].get('content')
        return isinstance(content, str) and bool(content.strip())
    return messages[-1].get('role') in (('user', 'assistant') if prepare else ('user',))


def system_prepare_prompt(tokenizer, template, messages):
    """Render only the real system prefix, using the frozen chat template.

    The template requires a user query. A render-only boundary message lets us
    obtain its exact system serialization; its whole serialized suffix is removed
    before tokenization for the engine. No dummy observation enters model input,
    session history or the request journal.
    """
    if not (valid_message_tail(messages, True) and len(messages) == 1
            and messages[0]['role'] == 'system'):
        raise ValueError('System preparation requires one nonempty system message')
    boundary = {'role': 'user', 'content': 'system-prefix-render-boundary'}
    kw = dict(chat_template=template, tokenize=False, add_generation_prompt=False,
              continue_final_message=False, tools=None, enable_thinking=True)
    full = tokenizer.apply_chat_template(messages + [boundary], **kw)
    suffix = tokenizer.apply_chat_template([boundary], **kw)
    if not suffix or not full.endswith(suffix) or len(full) <= len(suffix):
        raise ValueError('Frozen template does not expose an exact system prefix')
    prefix = full[:-len(suffix)]
    tokens = list(tokenizer.encode(prefix, add_special_tokens=False))
    full_tokens = list(tokenizer.encode(full, add_special_tokens=False))
    if not tokens or full_tokens[:len(tokens)] != tokens:
        raise ValueError('System preparation token boundary differs from full chat')
    return {'prompt': prefix, 'prompt_token_ids': tokens}


async def serve(config):
    import torch
    from aiohttp import web
    from PIL import Image
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.renderers.params import ChatParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    dest = Path(config['output_dir'])
    dest.mkdir(parents=True, exist_ok=True)
    key = Path(config['api_key_file']).read_text().strip()
    if len(key) < 20:
        raise ValueError('A nonempty strong API key is required')
    kwargs = dict(config['engine'])
    if kwargs['tensor_parallel_size'] != torch.cuda.device_count():
        raise ValueError('Visible GPU count does not match TP')
    if not all('H200' in torch.cuda.get_device_name(i) and torch.cuda.get_device_capability(i) == (9, 0) for i in range(torch.cuda.device_count())):
        raise ValueError('This deployment recipe requires SM90 H200 GPUs')
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    if config.get('fast_mm_check', True):
        import fast_mm_check
        fast_mm_check.install()
    renderer = engine.renderer
    template = Path(config['chat_template']).read_text()
    compatible = config.get('workstation_compat', False)
    profile = None
    if compatible:
        if config['format_version'] != 'think_memory_minimap_compact_v1':
            raise ValueError('Workstation compatibility requires a compact checkpoint')
        model_dir = Path(kwargs['model'])
        tokenizer_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(model_dir.glob('*.json'))
                           if p.name.startswith('tokenizer') or 'processor' in p.name}
        profile = rendering_profile(config['chat_template'], digest(tokenizer_files),
            dict(processor_files=tokenizer_files, image_mode='RGB',
                 mm_processor_kwargs=kwargs.get('mm_processor_kwargs', {})))
    params = ChatParams(chat_template=template, chat_template_content_format='openai', chat_template_kwargs={
        'add_generation_prompt': True, 'continue_final_message': False,
        'tools': None, 'enable_thinking': True})
    prep_params = dataclasses.replace(params, chat_template_kwargs={
        **params.chat_template_kwargs, 'add_generation_prompt': False})
    tokenize = renderer.default_chat_tok_params.with_kwargs(max_length=kwargs['max_model_len']-512)
    processor = renderer.get_mm_processor()
    original_apply = processor.apply
    def apply(*args, **kw):
        old = torch.get_num_threads()
        torch.set_num_threads(config.get('preprocess_threads', 16))
        try:
            return original_apply(*args, **kw)
        finally:
            torch.set_num_threads(old)
    processor.apply = apply
    from .fast_image_prompt import install as install_fast_image_prompt
    fast_prompt_counters = (install_fast_image_prompt(processor, verify_first=200)
                            if config.get('fast_image_prompt', True) else {})
    from .render_profile import install as install_render_profile
    render_stages = install_render_profile(processor, dest)
    identity = dict(model_revision=config['model_revision'], format_version=config['format_version'],
                    chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                    tokenizer_sha256=hashlib.sha256((Path(kwargs['model'])/'tokenizer.json').read_bytes()).hexdigest(),
                    quantization=kwargs.get('quantization'), tp=kwargs['tensor_parallel_size'],
                    kv_cache_dtype=kwargs.get('kv_cache_dtype', 'auto'), mtp=False)
    if profile:
        profile['deployment_identity'] = identity
    metadata = dict(identity=identity, pid=os.getpid(), worker_id=config['worker_id'],
                    gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    engine_instance=uuid.uuid4().hex, torch=torch.__version__)
    _sc = engine.vllm_config.scheduler_config
    metadata['fairness_budgets'] = dict(batch=_sc.max_num_batched_tokens,
        encoder_compute=_sc.max_num_encoder_input_tokens, encoder_cache=_sc.encoder_cache_size)
    if metadata['fairness_budgets']['encoder_cache'] != config['encoder_cache_tokens']:
        raise ValueError('Encoder cache override did not reach the actual engine')
    if profile:
        metadata['lumine_rendering'] = public_profile(profile)
        metadata['capabilities'] = ['think-memory-minimap-compact-v1', 'sampling', 'release', 'completed-action-prefix-v1', 'policy-state-stream-v1', 'generated-dialogue-v1', 'context-limited-generation-v1']
    (dest/'metadata.json').write_text(json.dumps(metadata, indent=2))
    journal = RequestJournal(dest/'requests.jsonl')
    images = OrderedDict()
    image_bytes = 0
    prepared = OrderedDict()
    ledger = OrderedDict()
    lock = asyncio.Lock()

    @web.middleware
    async def auth_errors(request, handler):
        if not hmac.compare_digest(request.headers.get('Authorization', ''), 'Bearer '+key):
            return web.json_response({'error': 'unauthorized'}, status=401)
        try:
            return await handler(request)
        except (ValueError, KeyError, TypeError) as error:
            return web.json_response({'error': type(error).__name__, 'detail': str(error)}, status=400)

    async def health(request):
        await engine.check_health()
        try:
            journal.check()
        except RuntimeError as error:
            return web.json_response(dict(metadata, request_journal=journal.snapshot(),
                                          error=str(error)), status=503)
        return web.json_response(dict(metadata, request_journal=journal.snapshot()))

    async def upload(request):
        nonlocal image_bytes
        sha = request.match_info['sha']
        if not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('Invalid SHA256')
        data = await request.read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError('Image bytes do not match SHA256')
        if sha not in images:
            with Image.open(io.BytesIO(data)) as im:
                if im.width*im.height > 16_000_000:
                    raise ValueError('Image too large')
                pixels = im.convert('RGB').copy()
            size = pixels.width*pixels.height*3
            images[sha] = (pixels, size)
            image_bytes += size
            while image_bytes > config.get('image_cache_bytes', 512*1024*1024):
                _, (_, old_size) = images.popitem(last=False)
                image_bytes -= old_size
        images.move_to_end(sha)
        return web.json_response({'image_id': sha})

    async def infer(request):
        journal.check()
        received = time.perf_counter()
        body = await request.json()
        prep = request.path == '/prepare'
        fingerprint = digest([request.path, body])
        request_id = body['request_id']
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 256:
            raise ValueError('Invalid request_id')
        async with lock:
            if request_id in ledger:
                old_hash, old_result = ledger[request_id]
                if old_hash != fingerprint:
                    return web.json_response({'error': 'request_id_conflict'}, status=409)
                return web.json_response(old_result)
            cache_key = body['cache_key']
            if not isinstance(cache_key, str) or not 1 <= len(cache_key) <= 256:
                raise ValueError('Invalid cache_key')
            messages = body['messages']
            if not valid_message_tail(messages, prep):
                raise ValueError('Expected current user observation or confirmed-history preparation')
            ids = image_ids(messages)
            if not (0 if prep else 1) <= len(ids) <= config['engine']['limit_mm_per_prompt']['image']:
                raise ValueError('Image references exceed the configured engine limit')
            if any(not re.fullmatch('[0-9a-f]{64}', x) for x in ids):
                raise ValueError('Only content-addressed cache:// images are accepted')
            missing = [x for x in ids if x not in images]
            if missing:
                return web.json_response({'error': 'missing_images', 'image_ids': missing}, status=409)
            converted = []
            for message in messages:
                content = message['content']
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part['type'] == 'image_url':
                            sha = part['image_url']['url'].removeprefix('cache://')
                            images.move_to_end(sha)
                            parts.append({'type': 'image_pil', 'image_pil': images[sha][0], 'uuid': sha})
                        elif part['type'] == 'text':
                            parts.append(part)
                        else:
                            raise ValueError('Unsupported content part')
                    content = parts
                else:
                    # The frozen training template trims string messages. vLLM
                    # converts them to text-part lists before applying it.
                    content = content.strip()
                converted.append({'role': message['role'], 'content': content})
            started = time.perf_counter()
            generation = body.get('generation', {}) if compatible else {}
            enabled = generation.get('enable_thinking', True)
            count = 1 if prep else generation.get('max_tokens', body.get('max_tokens', 256))
            unlimited = compatible and not prep and count is None
            if type(enabled) is not bool or (not unlimited and (type(count) is not int or not 1 <= count <= (4096 if compatible else 512))):
                raise ValueError('Invalid thinking/token limit')
            request_params = dataclasses.replace(params, chat_template_kwargs={
                **params.chat_template_kwargs, 'enable_thinking': enabled})
            request_tokenize = renderer.default_chat_tok_params.with_kwargs(
                max_length=kwargs['max_model_len']-(1 if unlimited else count), truncate_prompt_tokens=None)
            if prep and len(converted) == 1 and converted[0]['role'] == 'system':
                # Generation and retained-history preparation keep their normal
                # renderer. This path sends only the exact system token prefix.
                system_prompt = system_prepare_prompt(renderer.get_tokenizer(), template, converted)
                inputs = await renderer.render_cmpl_async([system_prompt], request_tokenize)
                if list(inputs[0]['prompt_token_ids']) != system_prompt['prompt_token_ids']:
                    raise ValueError('System prefix tokens changed during engine rendering')
            else:
                _, inputs = await renderer.render_chat_async([converted], prep_params if prep else request_params, request_tokenize)
            inp = inputs[0]
            tokens = list(inp['prompt_token_ids'])
            token_budget = completion_budget(count, len(tokens), kwargs['max_model_len'])
            count = token_budget['effective_max_tokens']
            media = media_identity(inp)
            matched = False
            fallback = None
            if prep:
                if cache_key in prepared:
                    raise ValueError('Preparation cache_key must be unique')
            elif body.get('use_prepared'):
                saved = prepared.get(cache_key)
                matched = bool(saved and can_continue(tokens, media, saved[0], saved[2])
                               and ids[:len(saved[1])] == saved[1])
                if not matched:
                    fallback = 'missing_or_mismatched_prepared_prefix'
                    cache_key = uuid.uuid4().hex
            from .cache_policy import engine_cache_salt
            inp['cache_salt'] = engine_cache_salt(cache_key, config.get('prefix_cache_scope', 'request'))
            rendered = time.perf_counter()
            if not compatible and (not 1 <= count <= 512 or body.get('temperature', 0) != 0):
                raise ValueError('Supported sampling: temperature=0, max_tokens=1..512')
            sample_kw = {k: v for k, v in generation.items() if k not in ('enable_thinking', 'max_tokens', 'stop')}
            if prep or not compatible:
                sample_kw = dict(temperature=0, top_p=1, seed=20260728)
            checkpoint = 0
            if (prep and config.get('ocr_prefix_checkpoint', False)
                    and len(body['messages']) >= 2
                    and body['messages'][1].get('role') == 'user'
                    and isinstance(body['messages'][1].get('content'), str)
                    and '<screen_ocr>' in body['messages'][1]['content']):
                # Only a real archived OCR message supplies this checkpoint.
                # Before the first eviction, message 2 is a visual observation.
                # Leave the
                # mutable closing envelope out of the candidate checkpoint.
                # A poor boundary can only miss the cache: vLLM still verifies
                # every token/media hash before reusing any state.
                end_id = renderer.get_tokenizer().convert_tokens_to_ids('<|im_end|>')
                endings = [i for i, token in enumerate(tokens) if token == end_id]
                if len(endings) >= 2:
                    checkpoint = max(0, endings[1] - 32)
                    sample_kw['extra_args'] = {'ocr_prefix_checkpoint': checkpoint}
            sampling = SamplingParams(**sample_kw,
                max_tokens=count, ignore_eos=prep, skip_special_tokens=False,
                stop=None if prep else generation.get('stop', ['</a>']), include_stop_str_in_output=True,
                output_kind=RequestOutputKind.DELTA)
            submitted = time.perf_counter()
            first = None
            text = []
            generated = []
            cached = 0
            finish = None
            stop_reason = None
            engine_request_id = uuid.uuid4().hex
            streaming = bool(body.get('stream')) and not prep
            response = None
            delta_times = []
            async def emit(event):
                await response.write(('data: '+json.dumps(event, ensure_ascii=False)+'\n\n').encode())
            if streaming:
                response = web.StreamResponse(headers={'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
            stream_started = False
            try:
                async for event in engine.generate(inp, sampling, engine_request_id):
                    if first is None and any(output.token_ids for output in event.outputs):
                        first = time.perf_counter()
                    # Rendering has already reserved shared image-cache reads.
                    # A disconnected HTTP peer must not prevent those inputs
                    # from reaching the engine. Delay every socket write until
                    # the first model event, after the readers consumed them.
                    # Keep these writes inside abort cleanup as well.
                    if streaming and not stream_started:
                        await response.prepare(request)
                        await emit(dict(type='start', request_id=request_id, lumine_rendering=public_profile(profile),
                                        worker_id=config['worker_id'], identity=identity))
                        stream_started = True
                    cached = getattr(event, 'num_cached_tokens', cached) or 0
                    for output in event.outputs:
                        if output.token_ids and first is None:
                            first = time.perf_counter()
                        text.append(output.text)
                        generated.extend(output.token_ids)
                        if streaming and (output.text or output.token_ids):
                            elapsed = (time.perf_counter()-received)*1000
                            delta_times.append(dict(elapsed_ms=elapsed, token_count=len(generated), text_length=sum(map(len,text))))
                            await emit(dict(type='delta', text=output.text, token_ids=list(output.token_ids), worker_elapsed_ms=elapsed))
                        if output.finish_reason is not None:
                            finish = output.finish_reason
                            stop_reason = output.stop_reason
            except BaseException:
                await engine.abort(engine_request_id)
                raise
            ended = time.perf_counter()
            first = first or ended
            raw = ''.join(text)
            prefix = ''
            full = prefix + raw.removesuffix('<|im_end|>')
            timing = dict(queue_ms=(started-received)*1000, frontend_ms=(rendered-started)*1000,
                          engine_ttft_ms=(first-submitted)*1000, decode_ms=(ended-first)*1000,
                          total_ms=(ended-received)*1000, prompt_sha256=digest(tokens),
                          image_ids=ids, num_cached_tokens=cached, prompt_tokens=len(tokens),
                          exact_prefix_verified=matched, fallback=fallback,
                          prepared_prefix_tokens=len(prepared[body['cache_key']][0]) if matched else 0,
                          completion_token_ids=generated, worker_id=config['worker_id'])
            timing['fast_prompt_counters'] = dict(fast_prompt_counters)
            timing['render_stages_ms'] = dict(render_stages)
            timing['generation_budget'] = token_budget
            timing['stream_deltas'] = delta_times
            timing['prefix_cache_scope'] = config.get('prefix_cache_scope', 'request')
            timing['ocr_prefix_checkpoint'] = checkpoint
            if prep:
                prepared[cache_key] = (tokens, ids, media)
                while len(prepared) > 128:
                    prepared.popitem(last=False)
                result = dict(prepared=True, cache_key=cache_key, timing=timing)
            else:
                result = dict(id=request_id, object='chat.completion', model=config['model_revision'],
                    choices=[dict(index=0, message=dict(role='assistant', content=full), finish_reason=finish)],
                    usage=dict(prompt_tokens=len(tokens), completion_tokens=len(generated), total_tokens=len(tokens)+len(generated)),
                    raw_generated_text=raw, timing=timing)
                if profile:
                    record = reconstruct(raw, generated, finish, stop_reason, profile, enabled)
                    result['choices'][0]['message']['content'] = record['full_content']
                    result['choices'][0]['stop_reason'] = stop_reason
                    result['lumine_rendering'] = public_profile(profile)
                    result['lumine_outputs'] = [record]
            journal.append(dict(request_id=request_id, prepare=prep, request_sha256=fingerprint,
                                messages_sha256=digest(messages), prompt_token_ids=tokens,
                                execution_history=body.get('execution_history'), result=result))
            ledger[request_id] = (fingerprint, result)
            while len(ledger) > 128:
                ledger.popitem(last=False)
            if streaming:
                await emit(dict(type='result', result=result))
                await response.write_eof()
                return response
            return web.json_response(result)

    async def release(request):
        namespace = (await request.json()).get('namespace', '')
        if not re.fullmatch('[0-9a-f]{32}', namespace):
            raise ValueError('Invalid namespace')
        async with lock:
            for mapping in (prepared, ledger):
                for k in list(mapping):
                    if k.startswith(namespace + ':'):
                        del mapping[k]
        # vLLM's exact, salted APC blocks remain evictable global cache entries.
        # They can never be addressed by a new session's random namespace.
        return web.json_response({'released': True})

    async def bench_reset(request):
        body = await request.json()
        async with lock:
            if body.get('all', True):
                await engine.reset_mm_cache()
                await engine.reset_encoder_cache()
            reset = await engine.reset_prefix_cache()
            prepared.clear()
            ledger.clear()
        return web.json_response({'prefix_reset': reset, 'all': body.get('all', True)})

    async def bench_profile(request):
        body = await request.json()
        if body['action'] == 'start':
            await engine.start_profile(profile_prefix='infra-hotspot')
        else:
            await engine.stop_profile()
        return web.json_response({'ok': True})

    app = web.Application(middlewares=[auth_errors], client_max_size=16*1024*1024)
    app.router.add_get('/health', health)
    app.router.add_get('/v1/models', health)
    app.router.add_put('/images/{sha}', upload)
    app.router.add_post('/prepare', infer)
    app.router.add_post('/v1/chat/completions', infer)
    app.router.add_post('/release', release)
    app.router.add_post('/bench/reset', bench_reset)
    app.router.add_post('/bench/profile', bench_profile)
    # aiohttp normally leaves a handler running on peer disconnect. Make that
    # contract explicit: cancelling render_chat_async can strand SHM writer
    # reservations before engine admission. The handler drains to the first
    # engine event and only then observes a failed socket write and aborts.
    # Application shutdown may cancel handlers; the engine/cache shut down too.
    runner = web.AppRunner(app, keepalive_timeout=300, handler_cancellation=False)
    stopping = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stopping.set)
    try:
        await runner.setup()
        await web.TCPSite(runner, '0.0.0.0', config.get('port', 8000)).start()
        (dest/'READY').write_text(str(time.time()))
        await stopping.wait()
    finally:
        try:
            await runner.cleanup()
        finally:
            try:
                await asyncio.to_thread(journal.close)
            finally:
                engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(serve(json.loads(args.config.read_text())))


if __name__ == '__main__':
    main()

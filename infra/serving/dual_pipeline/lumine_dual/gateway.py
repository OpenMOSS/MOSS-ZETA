"""Optional CPU HTTP adapter using the existing Lumine request field names."""
import argparse
import asyncio
import base64
import hashlib
import hmac
import json
from pathlib import Path
import re

from aiohttp import web

from .client import DualPipeline, InvalidOutput
from .protocol import Context, parse_system_state, system_prompt, validate_block


def decode_observation(message):
    if message.get('role') != 'user' or not isinstance(message.get('content'), list):
        raise ValueError('Expected a user content list with one JPEG/PNG data URL and encoded OCR')
    texts, pictures = [], []
    for part in message['content']:
        if part.get('type') == 'text':
            texts.append(part['text'])
        elif part.get('type') == 'image_url':
            url = part['image_url']['url']
            if not re.match(r'^data:image/(jpeg|png);base64,', url):
                raise ValueError('Upload image bytes using a JPEG/PNG data URL')
            pictures.append(base64.b64decode(url.split(',', 1)[1], validate=True))
        else:
            raise ValueError('Unsupported observation part')
    if len(pictures) != 1:
        raise ValueError('Exactly one current observation image is required')
    block = ''.join(texts).strip()
    validate_block(block)
    return pictures[0], block


def create_app(config, *, pipeline_factory=DualPipeline):
    auth = Path(config['gateway_api_key_file']).read_text().strip()
    if len(auth) < 20:
        raise ValueError('A strong gateway API key is required')
    keys = [Path(p).read_text().strip() for p in config['api_key_files']]
    template = Path(config['system_template']).read_text().strip()
    version = config.get('format_version', 'training_v89')
    sessions = {}
    creation_lock = asyncio.Lock()

    @web.middleware
    async def authenticate(request, handler):
        if not hmac.compare_digest(request.headers.get('Authorization', ''), 'Bearer '+auth):
            return web.json_response({'error': 'unauthorized'}, status=401)
        try:
            return await handler(request)
        except InvalidOutput as error:
            return web.json_response({'error': 'invalid_model_output', 'committed': False,
                                      'result': error.result}, status=422)
        except (ValueError, KeyError, TypeError) as error:
            return web.json_response({'error': str(error)}, status=400)

    async def complete(request):
        body = await request.json()
        session_id = body['lumine_session_id']
        request_id = body['lumine_request_id']
        if not isinstance(session_id, str) or not re.fullmatch(r'[a-zA-Z0-9_.:-]{1,128}', session_id):
            raise ValueError('Invalid lumine_session_id')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError('Invalid lumine_request_id')
        if body.get('stream', False):
            raise ValueError('This action-completion adapter supports stream=false')
        if body.get('temperature', 0) != 0 or body.get('top_p', 1) != 1:
            raise ValueError('This deployment uses temperature=0, top_p=1')
        count = body.get('max_tokens', 256)
        if not isinstance(count, int) or not 1 <= count <= 512:
            raise ValueError('max_tokens must be 1..512; truncated turns are not committed')
        if body.get('lumine_format_version', version) != version:
            raise ValueError('Request format differs from deployed checkpoint format')
        messages = body['messages']
        state = body.get('lumine_state')
        if len(messages) == 2 and messages[0]['role'] == 'system':
            parsed = parse_system_state(messages[0]['content'])
            if state is not None and state != parsed:
                raise ValueError('lumine_state differs from system message')
            if system_prompt(template, parsed, version) != messages[0]['content']:
                raise ValueError('System does not match the training template')
            state = parsed
            messages = messages[1:]
        if len(messages) != 1:
            raise ValueError('Stateful HTTP adapter accepts one current observation per request')
        pixels, block = decode_observation(messages[0])
        async with creation_lock:
            if session_id not in sessions:
                if state is None:
                    raise ValueError('New session requires complete lumine_state or canonical system')
                if len(sessions) >= config.get('max_sessions', 8):
                    return web.json_response({'error': 'session_capacity; DELETE an idle session'}, status=429)
                p = pipeline_factory(config['endpoints'], keys,
                    Context(system_prompt(template, state, version), version),
                    system_template=template, session_id=session_id)
                await p.start()
                sessions[session_id] = p
            pipeline = sessions[session_id]
        if pipeline.lock.locked():
            return web.json_response({'error': 'session_busy; wait for the preceding causal turn'}, status=409)
        result = await pipeline.step(pixels, block, state=state,
            reset_state=body.get('lumine_state_reset', False), request_id=request_id, max_tokens=count)
        result['lumine_session_id'] = session_id
        result['lumine_request_id'] = request_id
        return web.json_response(result)

    async def remove(request):
        async with creation_lock:
            key = request.match_info['session']
            p = sessions.get(key)
            if p is None:
                return web.json_response({'deleted': False}, status=404)
            if p.lock.locked():
                return web.json_response({'error': 'session_busy'}, status=409)
            del sessions[key]
            await p.close()
        return web.json_response({'deleted': True})

    async def health(request):
        return web.json_response({'adapter': 'lumine_dual', 'format_version': version,
                                  'sessions': len(sessions), 'routes': 2})

    async def cleanup(app):
        await asyncio.gather(*(p.close() for p in sessions.values()))

    app = web.Application(middlewares=[authenticate], client_max_size=16*1024*1024)
    app.router.add_get('/health', health)
    app.router.add_post('/v1/chat/completions', complete)
    app.router.add_delete('/v1/lumine/sessions/{session}', remove)
    app.on_cleanup.append(cleanup)
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=18805)
    args = p.parse_args()
    web.run_app(create_app(json.loads(args.config.read_text())), host=args.host, port=args.port)


if __name__ == '__main__':
    main()

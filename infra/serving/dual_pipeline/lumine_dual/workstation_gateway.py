"""Server-side adapter for the unmodified Windows PolicyTransport/llmClient.

One API worker owns sessions. SQLite retains retired IDs across restarts; a
missing context is an explicit 409, never an implicit second execution.
"""
import argparse
import asyncio
from collections import OrderedDict
from copy import deepcopy
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid

from aiohttp import web
import httpx

from .client import DualPipeline, InvalidOutput
from .execution_history import MODE as EXECUTION_MODE
from .gateway import decode_observation
from .protocol import Context, digest, system_prompt
from .wire import generation_options
from lumine_protocol import load_profile

HISTORY_RECOVERY_VERSION = 'committed-generated-history-v1'

class WireError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def create_app(config, *, pipeline_factory=DualPipeline, task_factory=asyncio.create_task):
    auth = Path(config['gateway_api_key_file']).read_text().strip()
    if len(auth) < 20:
        raise ValueError('A strong gateway API key is required')
    keys = [Path(p).read_text().strip() for p in config['api_key_files']]
    template = Path(config['system_template']).read_text().strip()
    local_profile = load_profile()
    model = config['served_model_name']
    db_path = Path(config['session_db'])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    os.chmod(db_path, 0o600)
    db.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)')
    db.commit()
    sessions = {}
    creation_lock = asyncio.Lock()
    stateless_slots = asyncio.Semaphore(config.get('max_stateless', 2))
    tasks = set()
    stats = dict(completed=0, replayed=0, invalid=0, unavailable=0)

    def track(coro):
        task = task_factory(coro)
        tasks.add(task)
        def finished(t):
            tasks.discard(t)
            if not t.cancelled():
                t.exception()  # Retrieve a failure even after HTTP disconnect.
        task.add_done_callback(finished)
        return task

    @web.middleware
    async def authenticate(request, handler):
        if not hmac.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + auth):
            return web.json_response({'error': {'message': 'unauthorized'}}, status=401)
        try:
            return await handler(request)
        except WireError as e:
            return web.json_response({'error': {'message': e.message}}, status=e.status)
        except (ValueError, KeyError, TypeError) as e:
            return web.json_response({'error': {'message': str(e)}}, status=400)
        except httpx.HTTPError:
            return web.json_response({'error': {'message': 'Backend unavailable'}}, status=503)

    def check_body(body):
        if not isinstance(body, dict):
            raise ValueError('Expected a JSON object')
        generation = generation_options(body)
        execution_mode = body.get('lumine_execution_mode')
        if execution_mode not in (None, EXECUTION_MODE):
            raise ValueError('Unsupported lumine_execution_mode')
        if 'lumine_execution_receipts' in body and execution_mode != EXECUTION_MODE:
            raise ValueError('Execution receipts require completed_ack_v1')
        if execution_mode and not isinstance(body.get('lumine_execution_receipts', []), list):
            raise ValueError('lumine_execution_receipts must be a list')
        if body.get('model', model) != model:
            raise ValueError('Unknown model; consult /v1/models')
        if body.get('lumine_format_version', 'think_memory_minimap_compact_v1') not in ('think_memory_minimap_compact_v1', 'think-memory-minimap-compact-v1'):
            raise ValueError('This service requires think-memory-minimap-compact-v1')
        if 'chat_template' in body and body['chat_template'] != local_profile['template']:
            raise ValueError('chat_template differs from the deployed compact template')
        return generation

    def window_for(body):
        window = body.get('lumine_window', config.get('window', {'max': 5, 'keep': 3}))
        if not isinstance(window, dict) or set(window) != {'max', 'keep'}:
            raise ValueError('lumine_window requires max and keep')
        Context('', window_max=window['max'], window_keep=window['keep'])
        return dict(window)

    def context_for(state, window):
        return Context(system_prompt(template, state, 'think_memory_minimap_compact_v1'), 'think_memory_minimap_compact_v1',
            window_max=window['max'], window_keep=window['keep'])

    async def start_pipeline(state, sid, window, execution_mode=None):
        p = pipeline_factory(config['endpoints'], keys,
            context_for(state, window),
            system_template=template, session_id=sid,
            **({'execution_mode': execution_mode} if execution_mode else {}))
        try:
            await p.start()
            profiles = [h.get('lumine_rendering') for h in p.health]
            if not profiles[0] or profiles[0] != profiles[1]:
                raise ValueError('Both workers must expose the same rendering identity')
            if execution_mode and any('completed-action-prefix-v1' not in h.get('capabilities', []) for h in p.health):
                raise ValueError('Acknowledged history requires completed-action-prefix-v1 on both workers')
            if any('think-memory-minimap-compact-v1' not in h.get('capabilities', []) for h in p.health):
                raise ValueError('Workers require the workstation compatibility recipe')
            if profiles[0]['template_sha256'] != local_profile['template_sha256'] or profiles[0]['version'] != 'think-memory-minimap-compact-v1':
                raise ValueError('Worker template does not match the workstation contract')
            return p
        except BaseException:
            await p.close()
            raise

    async def close_pipeline(p):
        await p.close()

    def response_body(p, result, sid=None, rid=None):
        result = deepcopy(result)
        if result.get('lumine_rendering') != p.health[0]['lumine_rendering']:
            raise ValueError('Backend identity changed; session unavailable')
        if not result.get('lumine_outputs'):
            raise ValueError('Backend omitted raw output diagnostics')
        result['model'] = model
        result['history_cancel_version'] = 'latest-unstarted-turn-v1'
        if sid is not None:
            result.update(lumine_session_id=sid, lumine_request_id=rid)
        return result

    async def run_session(entry, body, pixels, block, generation, sid, rid, fingerprint, on_delta=None):
        started = time.perf_counter()
        try:
            entry['active_phase'] = (rid, 'initializing')
            if entry['pipeline'] is None:
                entry['pipeline'] = await start_pipeline(entry['state'], sid, entry['window'], entry['execution_mode'])
            initialized = time.perf_counter()
            p = entry['pipeline']
            try:
                entry['active_phase'] = (rid, 'step_started')
                result = await p.step(pixels, block, request_id=rid,
                    await_next_prepare=body.get('await_next_prepare', False),
                    max_tokens=generation['max_tokens'], generation=generation, on_delta=on_delta,
                    **({'execution_receipts': body.get('lumine_execution_receipts', [])}
                       if entry['execution_mode'] else {}))
                status = 200
            except InvalidOutput as e:
                stats['invalid'] += 1
                result = e.result
                truncated = any(x.get('truncated') for x in result.get('lumine_outputs', []))
                # Let the unchanged PolicyActor run its truncation recovery.
                status = 200 if truncated else 422
                entry['poisoned'] = True
                if status == 422:
                    result['error'] = {'message': 'Invalid model output; session unavailable; create a new ID'}
            result = response_body(p, result, sid, rid)
            if entry.get('update_version') is not None:
                result['lumine_update_version'] = entry['update_version']
                result['lumine_state_sha256'] = digest(entry['state'])
                if 'next_route_preparation' in result:
                    result['next_route_preparation']['update_version'] = entry['update_version']
            result['gateway_session_timing'] = dict(init_ms=(initialized-started)*1000,
                total_ms=(time.perf_counter()-started)*1000)
            entry['seen'][rid] = fingerprint
            entry['results'][rid] = (status, result)
            while len(entry['results']) > config.get('result_cache_size', 64):
                entry['results'].popitem(last=False)
            stats['completed'] += 1
            return status, result
        except Exception:
            # The remote worker may already have generated. No automatic second
            # inference is safe without the committed local context and result.
            entry['poisoned'] = True
            stats['unavailable'] += 1
            raise WireError(409, 'Session unavailable after backend failure; create a new session ID')
        finally:
            entry['active'] = None

    async def complete(request):
        arrived = time.perf_counter()
        body = await request.json()
        read_at = time.perf_counter()
        generation = check_body(body)
        window = window_for(body)
        mode = request.headers.get('X-Lumine-Session-Mode',
            'session' if body.get('lumine_session_id') else 'stateless')
        if mode == 'stateless':
            raise ValueError('think-memory-minimap-compact-v1 currently requires lumine_session; single-frame stateless payloads do not match training')
            if any(k in body for k in ('lumine_state', 'lumine_session_id', 'lumine_request_id')) or request.headers.get('X-Lumine-Session-ID'):
                raise ValueError('stateless requests cannot carry Session state or IDs')
            status, result = await asyncio.shield(track(run_stateless(body, generation)))
            return web.json_response(result, status=status)
        if mode != 'session':
            raise ValueError('Unknown X-Lumine-Session-Mode')
        ids = []
        for header, field in [('X-Lumine-Session-ID', 'lumine_session_id'),
                              ('X-Lumine-Request-ID', 'lumine_request_id')]:
            h, b = request.headers.get(header), body.get(field)
            if h is not None and b is not None and h != b:
                raise ValueError('Conflicting header/body ID')
            value = h if h is not None else b
            if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_.:-]{1,128}', value):
                raise ValueError('Missing or invalid ' + field)
            ids.append(value)
        sid, rid = ids
        messages = body['messages']
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError('Session requires exactly one current observation')
        pixels, block = decode_observation(messages[0])
        decoded_at = time.perf_counter()
        # IDs may be carried in either envelope; the normalized complete body
        # binds model, template, pixels, state and sampling to the logical ID.
        fingerprint = digest({k: v for k, v in body.items() if k not in ('lumine_session_id', 'lumine_request_id')})
        async with creation_lock:
            entry = sessions.get(sid)
            if entry is None:
                if db.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone():
                    raise WireError(409, 'Unknown session or expired session; create a new ID with full state')
                if len(sessions) >= config.get('max_sessions', 8):
                    raise WireError(429, 'Session capacity reached; reset an idle session')
                # Validate before reserving the durable ID.
                state = deepcopy(body.get('lumine_state', {}))
                c = context_for(state, window)
                c.stage([hashlib.sha256(pixels).hexdigest(), 'minimap-validation-placeholder'], block)
                db.execute('INSERT INTO sessions VALUES (?)', (sid,)); db.commit()
                entry = dict(state=state, window=window, generation_identity=generation['enable_thinking'],
                    execution_mode=body.get('lumine_execution_mode'), history_mode=body.get('lumine_history_mode'),
                    pipeline=None, active=None, active_phase=None, prepare_task=None, update_version=None, prepare_receipt=None, seen={}, admitted=set(), results=OrderedDict(), cancelled={}, poisoned=False)
                sessions[sid] = entry
            if entry.get('recovering'):
                raise WireError(409, 'Session history recovery in progress')
            if entry.get('recovery_unavailable'):
                raise WireError(409, 'Session history recovery failed; recovery is required')
            if entry.get('prepare_task') and not entry['prepare_task'].done():
                raise WireError(409, 'Session preparation in progress')
            if entry.get('update_version') is not None:
                if (not entry['seen'] and body.get('update_version') != entry['update_version']) or ('update_version' in body and body['update_version'] != entry['update_version']):
                    raise WireError(409, 'Prepared session update_version mismatch')
                if not (entry.get('prepare_receipt') or {}).get('ready'):
                    raise WireError(409, 'Prepared session is not ready')
            if rid in entry['seen']:
                if entry['seen'][rid] != fingerprint:
                    raise WireError(409, 'Request ID conflict: payload differs')
                if rid not in entry['results']:
                    raise WireError(409, 'Request result expired; session unavailable; create a new ID')
                stats['replayed'] += 1
                status, result = entry['results'][rid]
                return web.json_response(result, status=status)
            if rid in entry['cancelled'] or rid in entry.get('recovery_fenced', ()):
                raise WireError(409, 'Request was cancelled and cannot be replayed')
            if entry['active'] is not None:
                if entry['active'][:2] != (rid, fingerprint) and entry['active'][0] == rid:
                    raise WireError(409, 'Request ID conflict: payload differs')
                raise WireError(409, 'Session request in progress; retry the same logical request')
            if entry['poisoned']:
                raise WireError(409, 'Session unavailable; create a new session ID')
            if 'lumine_state' in body and body['lumine_state'] != entry['state']:
                raise WireError(409, 'Session state is immutable; create a new session ID')
            if body.get('lumine_execution_mode') != entry['execution_mode']:
                raise WireError(409, 'Session execution mode is immutable; create a new session ID')
            if body.get('lumine_history_mode') != entry['history_mode']:
                raise WireError(409, 'Session history mode is immutable')
            if window != entry['window']:
                raise WireError(409, 'Session window is immutable; create a new session ID')
            if generation['enable_thinking'] != entry['generation_identity']:
                raise WireError(409, 'Session thinking configuration changed; create a new session ID')
            if len(entry['seen']) + len(entry['cancelled']) >= config.get('max_requests_per_session', 100000):
                raise WireError(409, 'Session request ledger expired; create a new session ID')
            stream = bool(body.get('stream'))
            response = None
            if stream:
                response = web.StreamResponse(headers={'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
                await response.prepare(request)
            async def emit(event):
                await response.write(('data: '+json.dumps({**event, 'lumine_session_id': sid, 'lumine_request_id': rid}, ensure_ascii=False)+'\n\n').encode())
            # A successful result is not the only proof of admission: failed
            # initialization/step requests must never be called never-admitted.
            entry['admitted'].add(rid)
            task = track(run_session(entry, body, pixels, block, generation, sid, rid, fingerprint, on_delta=emit if stream else None))
            entry['active'] = (rid, fingerprint, task)
            entry['active_phase'] = (rid, 'admitted')
        try:
            status, result = await asyncio.shield(task)
        except Exception as error:
            if not stream: raise
            try:
                await emit(dict(type='error', error=type(error).__name__, detail=str(error)))
                await response.write_eof()
            except (ConnectionError, RuntimeError): pass
            return response
        result = dict(result)
        result['gateway_timing'] = dict(body_read_ms=(read_at-arrived)*1000,
            validation_decode_ms=(decoded_at-read_at)*1000, total_ms=(time.perf_counter()-arrived)*1000)
        if stream:
            await emit(dict(type='result', result=result, status=status))
            await response.write_eof()
            return response
        return web.json_response(result, status=status)

    async def run_stateless(body, generation):
        async with stateless_slots:
            from .protocol import parse_system_state
            p = await start_pipeline(parse_system_state(body['messages'][0]['content']), uuid.uuid4().hex, window_for(body))
            try:
                messages = deepcopy(body['messages'])
                if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'user':
                    raise ValueError('Expected full messages ending at the current observation')
                for m in messages:
                    if isinstance(m.get('content'), list):
                        for part in m['content']:
                            if part.get('type') == 'image_url':
                                pixels, _ = decode_observation({'role': 'user', 'content': [
                                    {'type': 'text', 'text': '<screen_ocr></screen_ocr>'}, part]})
                                part['image_url']['url'] = 'cache://' + p.add_image(pixels)
                result = await p._run(0, '/v1/chat/completions', dict(messages=messages,
                    cache_key=p.engine_namespace + ':' + uuid.uuid4().hex, use_prepared=False,
                    request_id=p.engine_namespace + ':' + uuid.uuid4().hex, generation=generation))
                return 200, response_body(p, result)
            finally:
                await close_pipeline(p)

    async def prepare_session(request):
        sid = request.match_info['session']
        if not re.fullmatch(r'[a-zA-Z0-9_.:-]{1,128}', sid):
            raise ValueError('Invalid session ID')
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError('Expected JSON object')
        allowed = {'model', 'lumine_state', 'lumine_format_version', 'lumine_history_mode',
                   'lumine_window', 'chat_template_kwargs', 'update_version'}
        if set(body) - allowed:
            raise ValueError('Preparation accepts only state/settings, never messages or images')
        generation = check_body(body)
        if body.get('lumine_history_mode') != 'generated_dialogue_v1':
            raise ValueError('Preparation requires generated_dialogue_v1')
        if 'update_version' not in body:
            raise ValueError('Preparation requires update_version')
        state = deepcopy(body.get('lumine_state', {}))
        window = window_for(body)
        context_for(state, window)
        identity = digest(body)
        async with creation_lock:
            entry = sessions.get(sid)
            if entry:
                if entry.get('prepare_identity') != identity:
                    raise WireError(409, 'Preparation session ID conflict')
                if entry['active'] or entry['seen'] or entry['cancelled']:
                    raise WireError(409, 'Cannot prepare a session after generation starts')
                task = entry['prepare_task']
            else:
                if db.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone():
                    raise WireError(409, 'Unknown or retired session ID')
                if len(sessions) >= config.get('max_sessions', 8):
                    raise WireError(429, 'Session capacity reached')
                db.execute('INSERT INTO sessions VALUES (?)', (sid,)); db.commit()
                entry = dict(state=state, window=window, generation_identity=generation['enable_thinking'],
                    execution_mode=None, history_mode='generated_dialogue_v1', pipeline=None, active=None,
                    active_phase=None, seen={}, admitted=set(), results=OrderedDict(), cancelled={}, poisoned=False,
                    update_version=body['update_version'], prepare_identity=identity, prepare_receipt=None)
                sessions[sid] = entry
                async def run_prepare():
                    try:
                        p = await start_pipeline(state, sid, window)
                        entry['pipeline'] = p
                        result = await p.prepare_session()
                        receipt = dict(result, prepare_version='state-prefix-v1', session_id=sid,
                            update_version=entry['update_version'], state_sha256=digest(state))
                        entry['prepare_receipt'] = receipt
                        return receipt
                    except BaseException:
                        entry['poisoned'] = True
                        raise
                task = track(run_prepare())
                entry['prepare_task'] = task
        receipt = await asyncio.shield(task)
        return web.json_response(receipt, status=200 if receipt['ready'] else 503)

    async def reset(request):
        sid = request.match_info['session']
        async with creation_lock:
            entry = sessions.get(sid)
            if entry and entry.get('recovering'):
                raise WireError(409, 'Session history recovery in progress; reset after it completes')
            if entry and entry['active']:
                raise WireError(409, 'Session request in progress; reset after it completes')
            if entry and entry.get('prepare_task') and not entry['prepare_task'].done():
                # Preparation owns no generated dialogue. Superseded staging
                # sessions can be retired without waiting for a warm receipt.
                entry['prepare_task'].cancel()
                await asyncio.gather(entry['prepare_task'], return_exceptions=True)
            if entry:
                del sessions[sid]
        if entry and entry['pipeline']:
            await close_pipeline(entry['pipeline'])
        return web.json_response({'reset': True, 'deleted': entry is not None})

    async def cancel_turn(request):
        sid, rid = request.match_info['session'], request.match_info['request']
        async with creation_lock:
            entry = sessions.get(sid)
            if entry is None:
                raise WireError(409, 'Unknown session; cancellation was not acknowledged')
            if entry.get('recovering') or entry.get('recovery_unavailable'):
                raise WireError(409, 'Session history recovery in progress; cancellation was not acknowledged')
            if entry['execution_mode'] or entry['history_mode'] != 'generated_dialogue_v1':
                raise WireError(409, 'Cancellation requires generated dialogue mode')
            if rid in entry['cancelled']:
                return web.json_response(entry['cancelled'][rid])
            active = entry['active']
            if active and active[0] != rid:
                raise WireError(409, 'Cannot cancel a turn after a dependent request started')
            p = entry['pipeline']
            point = p.rollback_point if p else None
            if (not active and rid not in entry['admitted'] and rid not in entry['seen']
                    and (not point or point['request_id'] != rid)):
                # HTTP dispatch does not imply admission: a small cancellation
                # can overtake a chat POST still reading/validating its body.
                # Under the SAME lock as admission, fence this exact ID before
                # acknowledging unchanged history. This is not a rollback and
                # must not heal a poisoned session or disturb prior preparation.
                if len(entry['seen']) + len(entry['cancelled']) >= config.get('max_requests_per_session', 100000):
                    raise WireError(409, 'Session request ledger expired; cancellation was not acknowledged')
                receipt = dict(cancelled=True, request_id=rid, not_admitted=True,
                    entered_step=False, cancellation_kind='admission_fenced',
                    request_admission_blocked=True, history_changed=False,
                    turn=p.turn if p else 0, history_mode='generated_dialogue_v1',
                    history_sha256=digest(p.context.messages() if p else context_for(entry['state'], entry['window']).messages()),
                    lumine_session_id=sid, history_cancel_version='latest-unstarted-turn-v1')
                entry['cancelled'][rid] = receipt
                return web.json_response(receipt)
            phase = entry.get('active_phase')
            never_entered_step = bool(active and phase in ((rid, 'admitted'), (rid, 'initializing')))
            if active:
                active[2].cancel()
                await asyncio.gather(active[2], return_exceptions=True)
                entry['active'] = None
            p = entry['pipeline']
            if p is None or never_entered_step:
                if not active:
                    raise WireError(409, 'Unknown request; cancellation was not acknowledged')
                receipt = dict(cancelled=True, request_id=rid, turn=p.turn if p else 0,
                    entered_step=False, history_mode='generated_dialogue_v1',
                    history_sha256=digest(p.context.messages() if p else context_for(entry['state'], entry['window']).messages()))
            else:
                try:
                    receipt = await p.cancel_turn(rid)
                except ValueError as error:
                    raise WireError(409, str(error))
            entry['seen'].pop(rid, None)
            entry['results'].pop(rid, None)
            entry['poisoned'] = False
            receipt.update(lumine_session_id=sid, history_cancel_version='latest-unstarted-turn-v1')
            entry['cancelled'][rid] = receipt
            return web.json_response(receipt)

    async def recover_session(request):
        sid = request.match_info['session']
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {'recovery_id', 'request_id', 'lumine_state', 'lumine_window'}:
            raise ValueError('Recovery requires recovery_id, request_id, lumine_state and lumine_window only')
        recovery_id, rid = body['recovery_id'], body['request_id']
        for value, name in ((recovery_id, 'recovery_id'), (rid, 'request_id')):
            if value is None and name == 'request_id':
                continue
            if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_.:-]{1,128}', value):
                raise ValueError('Invalid ' + name)
        window = window_for(body)
        context_for(body['lumine_state'], window)
        identity = digest(body)
        async with creation_lock:
            entry = sessions.get(sid)
            if entry is None:
                raise WireError(409, 'Unknown session; committed history recovery was not acknowledged')
            if entry['execution_mode'] or entry['history_mode'] != 'generated_dialogue_v1':
                raise WireError(409, 'Recovery requires generated dialogue mode')
            if body['lumine_state'] != entry['state'] or window != entry['window']:
                raise WireError(409, 'Recovery state/window differs from the admitted session')
            recoveries = entry.setdefault('recoveries', {})
            previous = recoveries.get(recovery_id)
            if previous:
                if previous[0] != identity:
                    raise WireError(409, 'Recovery ID conflict: payload differs')
                task = previous[1]
            else:
                if entry.get('recovering'):
                    raise WireError(409, 'Another session history recovery is in progress')
                if entry.get('prepare_task') and not entry['prepare_task'].done():
                    raise WireError(409, 'Initial session preparation is still in progress')
                active = entry['active']
                if active and active[0] != rid:
                    raise WireError(409, 'Recovery request_id differs from the active request')
                # Cancellation is synchronous under the admission lock. No new
                # request can be admitted while the canceled task is joined.
                entry['recovering'] = True
                entry['recovery_unavailable'] = True
                entry['poisoned'] = True
                if rid:
                    entry.setdefault('recovery_fenced', set()).add(rid)
                if active:
                    active[2].cancel()

                async def run_recovery():
                    try:
                        if active:
                            await asyncio.gather(active[2], return_exceptions=True)
                        entry['active'] = None
                        if entry['pipeline'] is None:
                            # Existing admitted ID, failed before initialization.
                            # No history ever existed; unknown IDs never reach here.
                            entry['pipeline'] = await start_pipeline(entry['state'], sid, entry['window'])
                        p = entry['pipeline']
                        committed = bool(rid and rid in p.ledger and p.ledger[rid][1].get('committed'))
                        retained = next((key for key, (_, result) in reversed(p.ledger.items())
                                         if result.get('committed')), None)
                        cancelled_id = rid if rid and not committed else None
                        if cancelled_id:
                            # Fence an overtaken POST too: the canceled request
                            # cannot reenter after the recovery block is lifted.
                            entry['cancelled'][rid] = dict(cancelled=True, request_id=rid,
                                lumine_session_id=sid, history_changed=False,
                                cancellation_kind='recovery_admission_fenced',
                                request_admission_blocked=True,
                                history_cancel_version='latest-unstarted-turn-v1')
                            entry['seen'].pop(rid, None)
                            entry['results'].pop(rid, None)
                        result = await p.recover_generated_history()
                        receipt = dict(result, recovery_version=HISTORY_RECOVERY_VERSION,
                            session_id=sid, recovery_id=recovery_id,
                            state_sha256=digest(entry['state']), retained_request_id=retained,
                            cancelled_request_id=cancelled_id)
                        if receipt['ready']:
                            entry['poisoned'] = False
                            entry['recovery_unavailable'] = False
                        return 200 if receipt['ready'] else 503, receipt
                    except Exception as error:
                        # Preserve the original context and keep admissions shut.
                        # The same recovery ID receives this same failure receipt.
                        return 503, dict(recovery_version=HISTORY_RECOVERY_VERSION,
                            session_id=sid, recovery_id=recovery_id, ready=False,
                            state_sha256=digest(entry['state']),
                            error={'message': 'History recovery failed: ' + type(error).__name__ + ': ' + str(error)})
                    finally:
                        entry['recovering'] = False
                task = track(run_recovery())
                recoveries[recovery_id] = (identity, task)
        status, receipt = await asyncio.shield(task)
        return web.json_response(receipt, status=status)

    async def models(request):
        return web.json_response({'object': 'list', 'data': [{'id': model, 'object': 'model', 'owned_by': 'lumine'}]})

    async def cache_stats(request):
        if request.path == '/health':
            async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
                for endpoint,key in zip(config['endpoints'], keys):
                    r = await client.get(endpoint.rstrip('/')+'/health', headers={'Authorization':'Bearer '+key})
                    r.raise_for_status()
                    if r.json()['identity']['model_revision'] != model:
                        raise ValueError('Unexpected backend model')
        return web.json_response({**stats, 'pipeline_revision': 'hot-swap-generated-stream-v1', 'session_prepare_version': 'state-prefix-v1', 'history_cancel_version': 'latest-unstarted-turn-v1', 'history_recovery_version': HISTORY_RECOVERY_VERSION, 'execution_modes': [EXECUTION_MODE], 'history_modes': ['generated_dialogue_v1'], 'stream_versions': ['policy-state-stream-v1'], 'sessions': len(sessions), 'routes': 2,
            'active': sum(e['active'] is not None for e in sessions.values())})

    async def eval_unsupported(request):
        raise WireError(501, 'teacher_forced evaluation uses the existing eval service; this gateway supports online Session and HTTP stateless')

    async def cleanup(app):
        if tasks:
            await asyncio.gather(*list(tasks), return_exceptions=True)
        await asyncio.gather(*(close_pipeline(e['pipeline']) for e in sessions.values() if e['pipeline']), return_exceptions=True)
        db.close()

    app = web.Application(middlewares=[authenticate], client_max_size=32 * 1024 * 1024)
    app.router.add_post('/v1/chat/completions', complete)
    app.router.add_post('/v1/lumine/sessions/{session}/prepare', prepare_session)
    app.router.add_post('/v1/lumine/sessions/{session}/recover', recover_session)
    app.router.add_post('/v1/lumine/sessions/{session}/reset', reset)
    app.router.add_post('/v1/lumine/sessions/{session}/requests/{request}/cancel', cancel_turn)
    app.router.add_delete('/v1/lumine/sessions/{session}', reset)
    app.router.add_get('/v1/models', models)
    app.router.add_get('/v1/lumine/cache-stats', cache_stats)
    app.router.add_get('/health', cache_stats)
    app.router.add_post('/v1/eval/chat/completions', eval_unsupported)
    app.on_cleanup.append(cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18806)
    args = parser.parse_args()
    web.run_app(create_app(json.loads(args.config.read_text())), host=args.host,
                port=args.port, handler_cancellation=False)


if __name__ == '__main__':
    main()

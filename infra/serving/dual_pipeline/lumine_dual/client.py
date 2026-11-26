"""Long-lived HTTPS clients and causal, transactional two-route scheduling."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time
import uuid
import ssl

import httpx
import certifi

from .protocol import Context, digest, image_ids, parse_action, system_prompt
from .execution_history import ExecutionHistory, MODE as EXECUTION_MODE

# HTTPX otherwise reloads the CA bundle twice for every session, even for HTTP
# loopback routes. Reuse the immutable verified trust context, not session state.
_TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class InvalidOutput(RuntimeError):
    def __init__(self, result):
        super().__init__('Incomplete/invalid model output; history was not committed')
        self.result = result


@dataclass
class Preparation:
    task: asyncio.Task
    key: str
    prefix: list


class DualPipeline:
    """One causal game session; each route is an independent TP4 deployment.

    State changes are explicit workstation acknowledgements. In particular,
    generated model_thought is not automatically installed before actions run.
    """

    def __init__(self, endpoints, api_keys, context, *, system_template=None,
                 policy='prefetch', max_tokens=256, session_id=None, execution_mode=None):
        if len(endpoints) != 2 or len(api_keys) != 2:
            raise ValueError('Exactly two endpoints/keys required')
        if policy not in ('prefetch', 'serial'):
            raise ValueError('Unknown scheduling policy')
        self.clients = [httpx.AsyncClient(
            base_url=url.rstrip('/'), headers={'Authorization': 'Bearer ' + key},
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4, keepalive_expiry=None),
            timeout=httpx.Timeout(180, connect=10), trust_env=False,
            verify=_TLS_CONTEXT,
        ) for url, key in zip(endpoints, api_keys)]
        if execution_mode not in (None, EXECUTION_MODE):
            raise ValueError('Unknown execution history mode')
        self.execution_mode = execution_mode
        self.execution = ExecutionHistory() if execution_mode else None
        self.context = context
        self.system_template = system_template
        self.policy = policy
        self.max_tokens = max_tokens
        self.session_id = session_id or uuid.uuid4().hex
        self.engine_namespace = uuid.uuid4().hex
        self.turn = 0
        self.pending = None
        self.images = OrderedDict()
        self.uploaded = [set(), set()]
        self.ledger = OrderedDict()
        self.lock = asyncio.Lock()
        self.health = None
        self.closed = False
        self.rollback_point = None
        self.initial_preparations = {}

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def start(self):
        try:
            self.health = await asyncio.gather(*(self._json(i, 'GET', '/health') for i in range(2)))
            identities = [h['identity'] for h in self.health]
            if identities[0] != identities[1] or identities[0]['format_version'] != self.context.format_version:
                raise ValueError('Route model/template/format identities do not match')
        except BaseException:
            await self.close()
            raise
        return self.health

    async def _json(self, route, method, path, **kwargs):
        r = await self.clients[route].request(method, path, **kwargs)
        r.raise_for_status()
        return r.json()

    def add_image(self, pixels):
        key = hashlib.sha256(pixels).hexdigest()
        self.images[key] = pixels
        self.images.move_to_end(key)
        return key

    def _prune_images(self, additional=()):
        needed = set(image_ids(self.context.messages())) | set(additional)
        if self.rollback_point:
            needed.update(image_ids(self.rollback_point['context'].messages()))
        if self.execution is not None:
            needed.update(self.execution.pinned_images())
        if self.pending is not None:
            needed.update(image_ids(self.pending.prefix))
        for image_id in list(self.images):
            if image_id not in needed:
                del self.images[image_id]
                for uploaded in self.uploaded:
                    uploaded.discard(image_id)

    async def upload(self, route, ids, *, force=False):
        for key in dict.fromkeys(ids):
            if force or key not in self.uploaded[route]:
                await self._json(route, 'PUT', '/images/' + key,
                                 content=self.images[key], headers={'Content-Type': 'application/octet-stream'})
                self.uploaded[route].add(key)

    async def _run(self, route, path, body, on_delta=None):
        started = time.perf_counter()
        ids = image_ids(body['messages'])
        upload_count = sum(key not in self.uploaded[route] for key in dict.fromkeys(ids))
        await self.upload(route, ids)
        uploaded = time.perf_counter()
        repair = False
        if on_delta is not None:
            async def read_stream():
                async with self.clients[route].stream('POST', path, json={**body, 'stream': True}) as response:
                    if response.status_code == 409:
                        payload = json.loads(await response.aread())
                        if payload.get('error') == 'missing_images':
                            return None, payload['image_ids']
                    response.raise_for_status()
                    if 'text/event-stream' not in response.headers.get('content-type', ''):
                        return json.loads(await response.aread()), None
                    final = None
                    async for line in response.aiter_lines():
                        if not line.startswith('data: '): continue
                        event = json.loads(line[6:])
                        if event['type'] == 'result': final = event['result']
                        elif event['type'] in ('start', 'delta'):
                            await on_delta({**event, 'route': route})
                        else: raise ValueError('Unexpected worker stream event')
                    if final is None: raise ValueError('Worker stream ended without result')
                    return final, None
            result, missing = await read_stream()
            if missing is not None:
                repair = True
                await self.upload(route, missing, force=True)
                result, missing = await read_stream()
                if missing is not None: raise ValueError('Repeated worker image cache miss')
            result['transport_timing'] = dict(image_upload_ms=(uploaded-started)*1000,
                backend_http_ms=(time.perf_counter()-uploaded)*1000, response_decode_ms=0,
                image_upload_count=upload_count, missing_image_repair=repair, streamed=True)
            return result
        r = await self.clients[route].post(path, json=body)
        # The server reports misses BEFORE invoking the engine; one repair is safe.
        if r.status_code == 409 and r.json().get('error') == 'missing_images':
            repair = True
            await self.upload(route, r.json()['image_ids'], force=True)
            r = await self.clients[route].post(path, json=body)
        r.raise_for_status()
        received = time.perf_counter()
        result = r.json()
        result['transport_timing'] = dict(image_upload_ms=(uploaded-started)*1000,
            backend_http_ms=(received-uploaded)*1000, response_decode_ms=(time.perf_counter()-received)*1000,
            image_upload_count=upload_count, missing_image_repair=repair)
        return result

    async def _prepare(self, route, prefix, key):
        result = await self._run(route, '/prepare', dict(messages=prefix, cache_key=key, request_id=key + ':prepare'))
        result['client_ready_at'] = time.perf_counter()
        return result

    async def _drain(self, pending):
        if pending is None:
            return None
        try:
            return await pending.task
        except Exception as error:
            # A failed optimization must not substitute stale state for full prefill.
            return {'error': type(error).__name__ + ': ' + str(error)}

    async def prepare_session(self):
        """Warm an empty state prefix on both routes, without an observation/turn."""
        async with self.lock:
            if self.turn or self.context.turns or self.context.archive or self.images or self.ledger or self.rollback_point:
                raise ValueError('Session preparation requires zero history and zero images')
            prefix = self.context.header()
            for route in range(2):
                if route not in self.initial_preparations:
                    key = self.engine_namespace + ':state:' + str(route) + ':' + uuid.uuid4().hex
                    self.initial_preparations[route] = Preparation(
                        asyncio.create_task(self._prepare(route, prefix, key)), key, prefix)
            prepared = await asyncio.gather(*(self._drain(self.initial_preparations[i]) for i in range(2)))
            routes = [dict(route=i, prepared=bool(r and r.get('prepared') is True and not r.get('error')),
                           prefix_sha256=digest(prefix), image_count=0,
                           timing=r.get('timing') if r else None) for i, r in enumerate(prepared)]
            return dict(ready=all(r['prepared'] for r in routes), routes=routes,
                        history_turns=self.turn, image_count=len(self.images), prefix_sha256=digest(prefix))

    async def recover_generated_history(self):
        """Caller fences admissions and joins the old step before entering.

        Context is authoritative: a completed generated turn stays even when
        no action was executed. A staged/partially decoded turn never entered
        it. Do not use cancel_turn(), which intentionally rolls back a turn.
        """
        async with self.lock:
            if self.execution is not None:
                raise ValueError('Recovery requires generated dialogue history')
            preparations = list(self.initial_preparations.values())
            if self.pending is not None:
                preparations.append(self.pending)
            for preparation in preparations:
                preparation.task.cancel()
            await asyncio.gather(*(p.task for p in preparations), return_exceptions=True)
            self.pending = None
            self.initial_preparations.clear()
            health = await asyncio.gather(*(self._json(i, 'GET', '/health', timeout=10) for i in range(2)))
            if not self.health or any(
                    h.get('identity') != self.health[i].get('identity') or
                    h.get('lumine_rendering') != self.health[i].get('lumine_rendering')
                    for i, h in enumerate(health)):
                raise ValueError('Worker model/rendering identity changed during recovery')
            # release takes each worker's inference lock. Acknowledgement is
            # proof the disconnected old request can no longer own its lane.
            # Failure leaves the gateway unavailable and its history intact.
            if not all('release' in h.get('capabilities', []) for h in health):
                raise ValueError('Recovery requires release on both workers')
            released = await asyncio.gather(*(self._json(i, 'POST', '/release',
                json={'namespace': self.engine_namespace}, timeout=10) for i in range(2)))
            if not all(r.get('released') is True for r in released):
                raise ValueError('Worker did not acknowledge old namespace release')
            self.engine_namespace = uuid.uuid4().hex
            # This is exactly the normal NEXT stage trim, performed once while
            # paused so the warmed prefix is the one the next image will use.
            restored = deepcopy(self.context)
            before = len(restored.turns)
            restored.trim()
            prefix = restored.messages()
            prefix_hash = digest(prefix)
            needed = set(image_ids(prefix))
            if not needed.issubset(self.images):
                raise ValueError('Committed history images are missing')
            for route in range(2):
                key = self.engine_namespace + ':recover:' + str(route) + ':' + uuid.uuid4().hex
                self.initial_preparations[route] = Preparation(
                    asyncio.create_task(self._prepare(route, prefix, key)), key, prefix)
            prepared = await asyncio.gather(*(self._drain(self.initial_preparations[i]) for i in range(2)))
            routes = [dict(route=i, prepared=bool(r and r.get('prepared') is True and not r.get('error')),
                prefix_sha256=prefix_hash, image_count=len(needed),
                error=r.get('error') if r else 'missing preparation') for i, r in enumerate(prepared)]
            ready = all(r['prepared'] for r in routes)
            if ready:
                self.context = restored
                # A delayed legacy cancel must not withdraw preserved history.
                self.rollback_point = None
                self._prune_images()
            return dict(ready=ready, routes=routes, turn=self.turn,
                history_turns=len(self.context.turns), history_turns_before_trim=before,
                trimmed_for_next_stage=ready and before != len(restored.turns),
                trimmed_turns=before-len(restored.turns) if ready else 0,
                prefix_history_turns=len(restored.turns), prefix_image_count=len(needed),
                history_sha256=digest(self.context.messages()), prefix_sha256=prefix_hash,
                image_count=len(set(image_ids(self.context.messages()))),
                ocr_state=deepcopy(self.context.delivered),
                window_max=self.context.window_max, window_keep=self.context.window_keep)

    async def step(self, jpeg_bytes, ocr_block, *, request_id=None, state=None,
                   reset_state=False, max_tokens=None, generation=None, execution_receipts=None, on_delta=None, await_next_prepare=False):
        arrival = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex
        input_hash = digest([hashlib.sha256(jpeg_bytes).hexdigest(), ocr_block, state,
                             reset_state, max_tokens, generation, execution_receipts, await_next_prepare])
        async with self.lock:
            if request_id in self.ledger:
                previous_hash, result = self.ledger[request_id]
                if previous_hash != input_hash:
                    raise ValueError('request_id reused for different input')
                return result
            self.rollback_point = dict(request_id=request_id, context=self.context, turn=self.turn)
            if self.execution is not None:
                if state is not None or reset_state:
                    raise ValueError('Acknowledged execution state is immutable; start a new session')
                self.context, applied_receipts = self.execution.apply(self.context, [] if execution_receipts is None else execution_receipts)
            elif execution_receipts:
                raise ValueError('Execution receipts require completed_ack_v1')
            context = self.context
            if state is not None:
                if self.system_template is None:
                    raise ValueError('system_template required for explicit state updates')
                system = system_prompt(self.system_template, state, context.format_version)
                if system != context.system or reset_state:
                    context = Context(system, context.format_version, window_max=context.window_max, window_keep=context.window_keep)
            elif reset_state:
                context = Context(context.system, context.format_version, window_max=context.window_max, window_keep=context.window_keep)
            # The crop is derived from the exact decoded full-frame pixels,
            # matching the frozen training adapter, never a later screenshot.
            import io
            from PIL import Image
            image_started = time.perf_counter()
            with Image.open(io.BytesIO(jpeg_bytes)) as image:
                full=image.convert('RGB')
                if full.size!=(1280,720): raise ValueError('Training input requires a 1280x720 full frame')
                image_decoded = time.perf_counter()
                crop=full.crop((0,528,256,720)).resize((512,384),Image.Resampling.BICUBIC)
                crop_resized = time.perf_counter()
                from .lossless_png import encode_rgb
                output=io.BytesIO(encode_rgb(crop))
                image_encoded = time.perf_counter()
            key = [self.add_image(jpeg_bytes), self.add_image(output.getvalue())]
            try:
                if self.execution is None:
                    candidate, observation, messages, prefix = context.stage(key, ocr_block)
                else:
                    candidate, observation, messages, prefix, visible_ids, ocr_audit = self.execution.stage(context, key, ocr_block)
                    execution_audit = self.execution.summary(applied_receipts, visible_ids, ocr_audit)
            except BaseException:
                self._prune_images()
                raise
            current = self.turn % 2 if self.policy == 'prefetch' else 0
            pending, self.pending = self.pending, None
            if pending is None:
                pending = self.initial_preparations.get(current)
            # Both warmed routes are ready at recovery; the normal alternating
            # preparation owns subsequent turns after this first continuation.
            self.initial_preparations.clear()
            next_pending = None
            if self.policy == 'prefetch':
                next_key = self.engine_namespace + ':' + uuid.uuid4().hex
                next_pending = Preparation(asyncio.create_task(self._prepare(1-current, prefix, next_key)), next_key, prefix)
            try:
                # Uploads do not read/modify prepared KV state. Overlap them with
                # prior preparation, but join both before exact-prefix inference.
                wait_started = time.perf_counter()
                from .overlap_upload import join_preparation_upload
                prepared, upload_overlap = await join_preparation_upload(
                    self._drain(pending), self.upload(current, image_ids(messages)))
                waited = time.perf_counter()
                matches = bool(context is self.context and pending and prepared and not prepared.get('error')
                               and messages[:len(pending.prefix)] == pending.prefix)
                cache_key = pending.key if matches else self.engine_namespace + ':' + uuid.uuid4().hex
                result = await self._run(current, '/v1/chat/completions', dict(
                    messages=messages, cache_key=cache_key, use_prepared=matches,
                    request_id=self.engine_namespace + ':' + hashlib.sha256(request_id.encode()).hexdigest(),
                    max_tokens=(generation['max_tokens'] if generation is not None and 'max_tokens' in generation
                                else self.max_tokens if max_tokens is None else max_tokens), temperature=0,
                    **({'generation': generation} if generation is not None else {}),
                    **({'execution_history': execution_audit} if self.execution is not None else {})), on_delta=on_delta)
                if self.health[current].get('lumine_rendering') is not None and result.get('lumine_rendering') != self.health[current]['lumine_rendering']:
                    raise ValueError('Backend rendering identity changed; session unavailable')
                content = result['choices'][0]['message']['content']
                try:
                    if result['choices'][0]['finish_reason'] != 'stop' or any(not x['protocol_valid'] for x in result.get('lumine_outputs', [])):
                        raise ValueError('truncated')
                    action = parse_action(content, context.format_version)
                except ValueError:
                    result['committed'] = False
                    raise InvalidOutput(result)
                next_wait_started = time.perf_counter()
                next_prepared = await self._drain(next_pending) if await_next_prepare else None
                next_wait_ms = (time.perf_counter()-next_wait_started)*1000
                next_ready = bool(next_prepared and next_prepared.get('prepared') is True and not next_prepared.get('error'))
                self.turn += 1
                if self.execution is None:
                    candidate.turns.append((observation, content))
                    self.context = candidate
                else:
                    self.execution.remember(request_id, key, content, self.turn)
                self.pending = next_pending
                completed = time.perf_counter()
                result.update(committed=self.execution is None, turn=self.turn, route=current, parsed=action,
                              client_total_ms=(completed-arrival)*1000,
                              prepare_wait_ms=(waited-wait_started)*1000,
                              prepare_upload_overlap=upload_overlap,
                              client_context_ms=(wait_started-arrival)*1000,
                              client_context_breakdown_ms=dict(image_decode=(image_decoded-image_started)*1000, crop_resize=(crop_resized-image_decoded)*1000, lossless_png=(image_encoded-crop_resized)*1000, stage_after_image=(wait_started-image_encoded)*1000),
                              pipeline_revision='hot-swap-generated-stream-v1',
                              history_mode='completed_ack_v1' if self.execution else 'generated_dialogue_v1',
                              dialogue_audit=dict(messages_sha256=digest(messages), roles=[m['role'] for m in messages], image_ids=image_ids(messages), next_prefix_sha256=digest(prefix)),
                              window_max=context.window_max, window_keep=context.window_keep,
                              context_images=len(image_ids(messages)),
                              prepared_message_prefix_matched=matches,
                              preparation=prepared,
                              next_route_preparation=dict(ready=next_ready, awaited=bool(await_next_prepare),
                                  route=1-current if next_pending else None,
                                  prefix_sha256=digest(prefix) if next_pending else None,
                                  history_turns=self.turn, wait_ms=next_wait_ms,
                                  error=next_prepared.get('error') if next_prepared else None),
                              request_id=request_id)
                if self.execution is not None:
                    result['execution_history'] = self.execution.summary(applied_receipts, visible_ids, ocr_audit)
                self.ledger[request_id] = (input_hash, result)
                while len(self.ledger) > 64:
                    self.ledger.popitem(last=False)
                # Retain only images that may be needed by history/background repair.
                self._prune_images(image_ids(prefix))
                return result
            except BaseException:
                if next_pending is not None:
                    next_pending.task.cancel()
                    await asyncio.gather(next_pending.task, return_exceptions=True)
                self._prune_images()
                raise

    async def cancel_turn(self, request_id):
        """Caller must cancel/join the active step before entering this method.

        Only the newest turn can be withdrawn: a dependent request makes the
        earlier turn immutable. Workers never own the authoritative dialogue.
        """
        async with self.lock:
            point = self.rollback_point
            if not point or point['request_id'] != request_id:
                raise ValueError('Cannot withdraw a turn with unknown/dependent history')
            if self.pending:
                self.pending.task.cancel()
                await asyncio.gather(self.pending.task, return_exceptions=True)
            self.pending = None
            self.context, self.turn = point['context'], point['turn']
            self.ledger.pop(request_id, None)
            self.rollback_point = None
            self._prune_images()
            return dict(cancelled=True, request_id=request_id, turn=self.turn,
                        history_sha256=digest(self.context.messages()), history_mode='generated_dialogue_v1')

    async def close(self):
        if self.closed:
            return
        await asyncio.gather(self._drain(self.pending), *(self._drain(p) for p in self.initial_preparations.values()), return_exceptions=True)
        self.pending = None
        self.initial_preparations.clear()
        if self.health and all('release' in h.get('capabilities', []) for h in self.health):
            await asyncio.gather(*(self._json(i, 'POST', '/release',
                json={'namespace': self.engine_namespace}, timeout=10) for i in range(2)),
                return_exceptions=True)
        await asyncio.gather(*(c.aclose() for c in self.clients))
        self.closed = True

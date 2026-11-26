"""Evaluation-only route: full expert input is authoritative, game APIs unchanged."""
import asyncio,copy,hashlib,json,time,uuid
from dataclasses import dataclass,field
from contextvars import ContextVar
from typing import Optional,Any
from fastapi import HTTPException,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field


def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()

from lumine_protocol import media_identity, can_continue

class EvalRequest(BaseModel):
    mode:str='teacher_forced'
    session_id:str=Field(min_length=1,max_length=200)
    target_id:str=Field(min_length=1,max_length=200)
    session_version:Optional[str]=None
    start_ms:float
    round_ms:float=Field(gt=0)
    request:dict

@dataclass
class State:
    lock:Any=field(default_factory=asyncio.Lock)
    tokens:list=field(default_factory=list)
    media:list=field(default_factory=list)
    generation:int=0
    version:Optional[str]=None
    last_start:Optional[float]=None
    last_round:Optional[float]=None
    fingerprint:Optional[str]=None
    engine_id:Optional[str]=None
    replay_key:Any=None
    replay_response:Any=None
    last_active:float=field(default_factory=time.monotonic)

class EvaluationEngine:
    def __init__(self,engine,tokenizer):
        from vllm.entrypoints.openai.lumine_session import LumineSessionManager
        self.engine=engine;self.tokenizer=tokenizer;self.epoch=uuid.uuid4().hex;self.states={};self.context=ContextVar('eval_turn')
        self.manager=LumineSessionManager(engine,high_watermark=1000000,retain_turns=999999,eviction_policy='rebuild_on_evict',idle_timeout_s=1800)
    def __getattr__(self,name):return getattr(self.engine,name)
    async def reset(self,sid,state):
        await self.manager.reset(sid)
        if state.engine_id:await self.engine.abort(state.engine_id)
        state.tokens=[];state.media=[];state.generation+=1;state.engine_id=None
    async def generate(self,prompt,sampling,request_id,**kwargs):
        from vllm.sampling_params import RequestOutputKind
        from vllm.entrypoints.openai.lumine_session import slice_lumine_continuation,aggregate_delta_outputs,sampling_params_payload
        envelope,state,audit=self.context.get();sid='eval:'+envelope.session_id
        tokens=list(prompt['prompt_token_ids']);media=media_identity(prompt);params=copy.deepcopy(sampling);params.output_kind=RequestOutputKind.DELTA
        fp=identity({'sampling':sampling_params_payload(params),'profile':audit['rendering_identity']});contiguous=state.last_start is not None and abs(envelope.start_ms-state.last_start-state.last_round)<1e-6
        present=await self.manager.has_session(sid)
        matches=can_continue(tokens,media,state.tokens,state.media,contiguous)
        first_diff=next((i for i,(a,b) in enumerate(zip(tokens,state.tokens)) if a!=b),None)
        audit['cache_check']={'contiguous':contiguous,'prefix_tokens_equal':tokens[:len(state.tokens)]==state.tokens,'media_prefix_equal':[m for m in media if m[1]<len(state.tokens)]==state.media,'sampling_equal':fp==state.fingerprint,'engine_session_present':present,'previous_tokens':len(state.tokens),'first_token_difference':first_diff,'desired_near_difference':tokens[max(0,(first_diff or 0)-3):(first_diff or 0)+5],'cached_near_difference':state.tokens[max(0,(first_diff or 0)-3):(first_diff or 0)+5]}
        reuse=matches and fp==state.fingerprint and present
        if not reuse:await self.reset(sid,state)
        state.version=f'{self.epoch}:{state.generation}';state.engine_id=state.engine_id or f'{sid}:{self.epoch}:{state.generation}'
        user=self.tokenizer.encode('<|im_start|>user\n',add_special_tokens=False)
        sink=next((i for i in range(len(tokens)) if tokens[i:i+len(user)]==user),1)
        session=await self.manager.get_or_create(session_id=sid,request_id=state.engine_id,first_prompt=prompt,continuation_boundary_tokens=[],sink_token_count=max(sink,1),active_turn_start=max(sink,1),active_retainable_prefix_end=max(sink,1),sampling_params=params,config_fingerprint=fp,cache_salt=None,lora_request=kwargs.get('lora_request'),trace_headers=kwargs.get('trace_headers'),priority=kwargs.get('priority',0),data_parallel_rank=kwargs.get('data_parallel_rank'))
        suffix=slice_lumine_continuation(prompt,len(state.tokens)) if reuse else None
        turn=await session.reserve_turn(continuation_prompt=suffix,sampling_params=params,config_fingerprint=fp)
        began=time.perf_counter();first=None;last=None;completed=False
        try:
            async def observed():
                nonlocal first,last
                async for o in turn.outputs():
                    if first is None:first=time.perf_counter()
                    last=time.perf_counter();yield o
            async for output in aggregate_delta_outputs(observed()):
                c=output.outputs[0];generated=list(c.token_ids);cached=list(generated)
                # EOS is a stopping boundary, not a committed context token.
                eos=self.tokenizer.eos_token_id
                if c.finish_reason=='stop' and cached and cached[-1]==eos:cached.pop()
                state.tokens=tokens+cached;state.media=media;state.fingerprint=fp;state.last_start=envelope.start_ms;state.last_round=envelope.round_ms
                audit.update(session_version=state.version,cache_action='reuse' if reuse else 'rebuild',prompt_token_digest=identity(tokens),prompt_tokens=len(tokens),media=media,generated_token_ids=generated,
                             prefill_seconds=None if first is None else first-began,decode_seconds=None if first is None else last-first,
                             timing_scope='stream first/last output boundaries; may aggregate multiple decode tokens')
                # The HTTP layer must see only this target, not persistent history output.
                output=copy.copy(output);output.prompt_token_ids=tokens;output.finished=True;completed=True;yield output
        finally:
            await session.release_turn(turn)
            if not completed:await self.reset(sid,state)


def register(router):
    @router.post('/v1/eval/chat/completions')
    async def evaluate(envelope:EvalRequest,raw:Request):
        if envelope.mode!='teacher_forced':raise HTTPException(400,'Explicit teacher_forced mode required')
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.openai.lumine_rendering import profile,public_profile
        request=ChatCompletionRequest.model_validate(envelope.request)
        if request.stream or request.n not in (None,1):raise HTTPException(400,'Evaluation supports nonstream n=1 only')
        original=raw.app.state.openai_serving_chat
        existing=getattr(raw.app.state,'lumine_eval_handler',None)
        training=getattr(original,'lumine_training',None)
        p=existing.lumine_eval_profile if existing is not None else (training.protocol.render_profile if training else profile())
        if request.chat_template is not None and hashlib.sha256(request.chat_template.encode()).hexdigest()!=p['template_sha256']:raise HTTPException(409,'Template identity differs')
        # Identity checked above; use the immutable server template, not request overrides.
        request=request.model_copy(update={'chat_template':None})
        if not hasattr(raw.app.state,'lumine_eval_handler'):
            original=raw.app.state.openai_serving_chat;handler=copy.copy(original)
            proxy=EvaluationEngine(original.engine_client,original.renderer.tokenizer)
            handler.engine_client=proxy;handler.lumine_training=None;handler.lumine_session_manager=None;handler.lumine_eval_profile=p
            raw.app.state.lumine_eval_handler=handler
        handler=raw.app.state.lumine_eval_handler;proxy=handler.engine_client
        # Fixed capacity avoids racing an eviction against an in-flight request.
        # 71 video sessions fit; process restart/resume begins a fresh registry.
        if envelope.session_id not in proxy.states and len(proxy.states)>=128:
            raise HTTPException(503,'Evaluation session capacity reached')
        state=proxy.states.setdefault(envelope.session_id,State())
        async with state.lock:
            context_hash=identity({'request':envelope.request,'start_ms':envelope.start_ms,'round_ms':envelope.round_ms,'profile':public_profile(p)});key=(envelope.target_id,context_hash,envelope.session_version)
            if state.replay_key==key and state.replay_response is not None:return JSONResponse(state.replay_response)
            if state.version is not None and envelope.session_version!=state.version:raise HTTPException(409,'Stale evaluation session version')
            scope=dict(raw.scope);scope['headers']=[(k,v) for k,v in raw.scope['headers'] if not k.lower().startswith(b'x-lumine-')]+[(b'x-lumine-session-mode',b'stateless')]
            request_raw=Request(scope,raw.receive);audit={'rendering_identity':public_profile(p)};ctx=proxy.context.set((envelope,state,audit));began=time.perf_counter()
            try:
                result=await handler.create_chat_completion(request,request_raw)
                body=result.model_dump()
                if 'error' in body:return JSONResponse(body,status_code=body['error'].get('code',500) if isinstance(body['error'].get('code'),int) else 500)
                if not audit or 'choices' not in body:raise HTTPException(500,'Evaluation generation failed')
                audit.update(total_seconds=time.perf_counter()-began,context_digest=context_hash,target_id=envelope.target_id)
                body['lumine_eval']=audit;body['lumine_rendering']=public_profile(p)
                state.replay_key=key;state.replay_response=body;state.last_active=time.monotonic();return JSONResponse(body)
            finally:proxy.context.reset(ctx)

"""Training-matched think, memory, OCR and same-frame minimap rolling window."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib,json
from pretrain import training_ocr_input as ocr
from pretrain.action_component_decay import parse_action as validate_action

FORMATS=('think_memory_minimap_compact_v1',)
VERSION=FORMATS[0]

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def parse_action(content,format_version):
    if format_version != VERSION: raise ValueError('Expected think_memory_minimap_compact_v1')
    import re
    from pretrain.action_component_decay import ACTION, BUTTONS, BUTTON_TOKEN
    match=re.fullmatch(r'<think>([\s\S]*?)</think>(<a>[^<>]*</a>)',content)
    if not match: raise ValueError('Expected complete think and action envelope')
    thought,action=match.groups();a=ACTION.fullmatch(action)
    if '\n' in thought or '\r' in thought: raise ValueError('Compact think cannot contain literal newlines')
    if not a or '-0' in a[1]: raise ValueError('Invalid action coordinates')
    chunks=a[2].split(';')
    if len(chunks)!=6 or chunks[:3]!=[chunks[0]]*3 or chunks[3:]!=[chunks[3]]*3: raise ValueError('Expected two ordered 100ms button states')
    for chunk in chunks:
        tokens=BUTTON_TOKEN.findall(chunk)
        if ''.join(tokens)!=chunk or ''.join(b for b in BUTTONS if b in tokens)!=chunk: raise ValueError('Invalid buttons')
    marker='<|call_brain|>';call=marker in thought
    instruction=small_think=None
    if call:
        if thought.count(marker)!=1: raise ValueError('Repeated call marker')
        reason,payload=thought.split(marker)
        target=json.loads(payload.strip())
        if not isinstance(target,dict) or set(target)!={'instruction','small_think'}: raise ValueError('Invalid call payload keys')
        if not all(isinstance(target[k],str) for k in target) or not target['instruction'].strip(): raise ValueError('Invalid call values')
        instruction,small_think=target['instruction'],target['small_think']
        judgment=reason.strip()
    else: judgment=thought.strip()
    return dict(action=action[3:-4],judgment=judgment,call_brain=call,
                instruction=instruction,small_think=small_think)

def system_prompt(template,state,format_version):
    if format_version != VERSION: raise ValueError('Wrong format')
    if not isinstance(state,dict) or set(state)!={'instruction','memory','model_thought'}: raise ValueError('New session requires full instruction/memory/model_thought')
    if any(not isinstance(v,str) for v in state.values()): raise ValueError('State values must be strings')
    for key,value in state.items(): template=template.replace('{'+key+'}',value)
    return template.strip()

def parse_system_state(text):
    import re
    m=re.search(r'\n\n当前有效目标：([\s\S]*?)\n\nMemory：([\s\S]*?)\n\n模型思考：([\s\S]*)$',text)
    if not m: raise ValueError('Missing canonical state')
    return dict(zip(['instruction','memory','model_thought'],m.groups()))

def image_ids(messages):
    return [p['image_url']['url'].removeprefix('cache://') for m in messages
            if isinstance(m['content'],list) for p in m['content'] if p.get('type')=='image_url']

def observation(image_id,block):
    if not isinstance(image_id,list) or len(image_id)!=2: raise ValueError('Full frame plus same-frame minimap required')
    return dict(role='user',content=[dict(type='text',text=block+'\n完整游戏画面：'),
        dict(type='image_url',image_url={'url':'cache://'+image_id[0]}),
        dict(type='text',text='\n同帧小地图放大图：'),
        dict(type='image_url',image_url={'url':'cache://'+image_id[1]})])

def block_of(user):
    return user['content'][0]['text'].removesuffix('\n完整游戏画面：').strip()

def validate_block(block):
    events=ocr.parse(block)
    if len(events)>1:raise ValueError('At most one delivered S/D event per observation')
    for event in events:
        if any(ocr.normalize_value(v)!=v for v in event['changes'].values()):
            raise ValueError('OCR must use the training normalization')
    return events

def snapshot(state):
    return ocr.render('S',{k:state[k] for k in ocr.FIELDS if state[k]})

@dataclass
class Context:
    system:str
    format_version:str='think_memory_minimap_compact_v1'
    archive:list=field(default_factory=list)
    turns:list=field(default_factory=list)
    delivered:dict|None=None
    turn_ocr:list=field(default_factory=list)
    window_max:int=5
    window_keep:int=3

    def __post_init__(self):
        if self.format_version not in FORMATS:raise ValueError('Use think_memory_minimap_compact_v1')
        if type(self.window_max) is not int or type(self.window_keep) is not int or not 1 <= self.window_keep < self.window_max <= 40:
            raise ValueError('Window requires integer 1 <= keep < max <= 40')

    def trim(self):
        if len(self.turns)>=self.window_max:
            drop=len(self.turns)-self.window_keep
            for block in self.turn_ocr[:drop]:
                if ocr.parse(block):self.archive.append({'block':block})
            del self.turns[:drop];del self.turn_ocr[:drop]

    def header(self):
        # The h16/k8 training exporter omits the archive message until a
        # changed OCR event has actually left the image window.
        result = [dict(role='system', content=self.system)]
        if self.archive:
            result.append(dict(role='user', content=ocr.archive(self.archive)))
        return result

    def messages(self):
        result=self.header()
        for user,assistant in self.turns:
            result.extend([deepcopy(user),dict(role='assistant',content=assistant)])
        return result

    def stage(self,image_id,block):
        candidate=deepcopy(self);candidate.trim()
        events=validate_block(block)
        if candidate.delivered is None and (not events or events[0]['mode']!='S'):
            raise ValueError('First observation requires delivered OCR S snapshot; use empty S if unavailable')
        candidate.delivered=ocr.replay(block,candidate.delivered)
        # Preserve delivered S/D/empty events and causal delivery times.
        user=observation(image_id,block)
        messages=candidate.messages()+[user]
        candidate.turn_ocr.append(block)
        next_context=deepcopy(candidate);next_context.turns.append((user,None));next_context.trim()
        prefix=next_context.header()
        for past_user,assistant in next_context.turns:
            prefix.append(deepcopy(past_user))
            if assistant is not None:prefix.append(dict(role='assistant',content=assistant))
        return candidate,user,messages,prefix

    @classmethod
    def from_prefix(cls,messages,format_version,*,window_max=5,window_keep=3):
        if not messages or messages[0]['role']!='system':
            raise ValueError('Expected system and optional OCR aggregate')
        c=cls(messages[0]['content'],format_version,window_max=window_max,window_keep=window_keep)
        rest=messages[1:]
        if rest and rest[0]['role']=='user' and isinstance(rest[0]['content'],str):
            aggregate=rest[0]['content'].removeprefix(ocr.HEADER)
            events=ocr.parse(aggregate)
            if not events:
                raise ValueError('Empty OCR archive is not part of the training prefix')
            for event in events:c.archive.append({'block':ocr.render(event['mode'],event['changes'])})
            c.delivered=ocr.replay(aggregate)
            rest=rest[1:]
        if len(rest)%2 or len(rest)>2*window_max:raise ValueError('Too many completed turns')
        for user,assistant in zip(rest[::2],rest[1::2]):
            if user['role']!='user' or assistant['role']!='assistant' or len(image_ids([user]))!=2:
                raise ValueError('Invalid history')
            parse_action(assistant['content'],format_version)
            block=block_of(user);validate_block(block)
            c.delivered=ocr.replay(block,c.delivered)
            c.turns.append((deepcopy(user),assistant['content']));c.turn_ocr.append(block)
        return c

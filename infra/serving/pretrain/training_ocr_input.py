"""Causal P/B/C/K OCR inputs without model-visible frame numbers.

Normalize formatting only: no character substitutions, fuzzy edit-distance
merges, temporal debounce, or lookahead. Source frame IDs remain in metadata.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pretrain import joint_think_compact_ocr as legacy

VERSION = 'pbck_no_frame_format_normalized_v2'
FIELDS = ('P','B','C','K')
HEADER = '此前已移除图片和动作的 OCR 历史（按出现顺序）：\n'
OPEN, CLOSE = '<screen_ocr>\n', '\n</screen_ocr>'
EMPTY_BLOCK = '<screen_ocr></screen_ocr>'
CJK = r'\u3400-\u9fff'


def unescape(value):
    return re.sub(r'\\([\\|])',lambda m:m.group(1),value)


def escape(value):
    return value.replace('\\','\\\\').replace('|','\\|')


@lru_cache(maxsize=32768)
def normalize_value(value):
    lines=[]
    for encoded in legacy.split_lines(value):
        text=unicodedata.normalize('NFKC',unescape(encoded))
        text=re.sub(r'\s+',' ',text).strip()
        # Ellipsis length and repeated punctuation fluctuate without changing words.
        text=re.sub(r'\.{2,}','…',text)
        text=re.sub(r'([!?。])\1+',r'\1',text)
        text=re.sub(fr'(?<=[{CJK}]) +(?=[{CJK}A-Za-z0-9])','',text)
        text=re.sub(fr'(?<=[A-Za-z0-9]) +(?=[{CJK}])','',text)
        text=re.sub(r' +(?=[,;:!?。…])','',text)
        text=re.sub(fr'(?<=[,;:!?。…]) +(?=[{CJK}])','',text)
        if text:
            # Preserve line order and distinct repeated lines; only normalize glyphs.
            lines.append(escape(text))
    return '|'.join(lines)


def render(mode,changes):
    return OPEN+'\n'.join([mode]+[f'{key}={changes[key]}' for key in FIELDS if key in changes])+CLOSE


def observation_ocr(text):
    """Read the OCR payload with either supported image/OCR ordering."""
    if text.count('<image>') != 1:
        raise ValueError('Expected one observation image')
    if text.startswith('<image>'):
        return text.removeprefix('<image>').strip()
    if text.endswith('<image>'):
        return text.removesuffix('<image>').strip()
    raise ValueError('Image must be at an observation boundary')


def parse(block):
    if re.fullmatch(r'<screen_ocr>\s*</screen_ocr>', block):
        return []
    if not block.startswith(OPEN) or not block.endswith(CLOSE):
        raise ValueError('Invalid OCR envelope')
    result=[];event=None
    for line in block[len(OPEN):-len(CLOSE)].splitlines():
        if line in ('S','D'):
            event=dict(mode=line,changes={});result.append(event)
        else:
            key,sep,value=line.partition('=')
            if not event or not sep or key not in FIELDS or key in event['changes']:
                raise ValueError('Unexpected OCR header or field')
            event['changes'][key]=value
    if not result:raise ValueError('Missing OCR event')
    return result


def replay(block,state=None):
    events = parse(block)
    if not events and state is None:
        raise ValueError('Empty OCR before initial snapshot')
    for event in events:
        if event['mode']=='S':state={k:'' for k in FIELDS}
        if state is None:raise ValueError('Delta before snapshot')
        state={**state,**event['changes']}
    return state


class Encoder:
    def __init__(self):self.source=None;self.output=None;self.last_frame=0

    def convert(self,block):
        result=[]
        for event in source_events(block):
            if event['frame']<=self.last_frame:raise ValueError('Out-of-order OCR input')
            self.last_frame=event['frame']
            if event['mode']=='S':self.source={k:'' for k in legacy.FIELDS}
            if self.source is None:raise ValueError('Missing source snapshot')
            self.source.update(event['changes'])
            current={k:normalize_value(self.source[k]) for k in FIELDS}
            changes={k:v for k,v in current.items() if (bool(v) if self.output is None else v!=self.output[k])}
            if self.output is None or changes:
                encoded=render('S' if self.output is None else 'D',changes)
                result.append(dict(frame_index=event['frame'],block=encoded))
            self.output=current
        return result


def archive(records):
    if not records:raise ValueError('Missing initial OCR snapshot')
    return HEADER+OPEN+'\n'.join(r['block'][len(OPEN):-len(CLOSE)] for r in records)+CLOSE


@lru_cache(maxsize=16384)
def source_events(block):
    return legacy.parse_events(block)


@lru_cache(maxsize=2048)
def convert_history(history_json):
    encoder=Encoder();records=json.loads(history_json);history=[]
    for record in records:
        events=source_events(record['block'])
        if len(events)!=1 or events[0]['frame']!=record['frame_index']:
            raise ValueError('OCR history metadata disagrees with source text')
        history.extend(encoder.convert(record['block']))
    old_text=legacy.archive_message(records) if records else None
    return encoder.source,encoder.output,encoder.last_frame,history,old_text


def validate_row(row):
    state=None
    for message in row['messages']:
        if message['role']!='user' or '<screen_ocr>' not in message['content']:continue
        text=message['content']
        if text.count('<screen_ocr>')!=1 or text.count('</screen_ocr>')!=1:
            raise ValueError('Expected one OCR block per user message')
        block=(observation_ocr(text) if '<image>' in text else text[text.index('<screen_ocr>'):])
        state=replay(block,state)
        for event in parse(block):
            if any(normalize_value(v)!=v for v in event['changes'].values()):
                raise ValueError('Noncanonical OCR formatting')
    for r in json.loads(row['ocr_history_json']):
        if not isinstance(r['frame_index'],int):raise ValueError('Lost provenance frame')
        parse(r['block'])


def revise_ocr_row(row):
    if row['context_contract'].endswith('__'+VERSION):
        validate_row(row)
        return row
    messages=[dict(m) for m in row['messages']]
    encoder=Encoder()
    source,output,last_frame,history,old_archive=convert_history(row['ocr_history_json'])
    encoder.source=dict(source) if source is not None else None
    encoder.output=dict(output) if output is not None else None
    encoder.last_frame=last_frame
    index=row['ocr_history_message_index']
    if index is not None:
        if messages[index]['role']!='user' or messages[index]['content']!=old_archive:
            raise ValueError('OCR archive message differs from metadata')
        messages[index]['content']=archive(history)
    elif old_archive is not None:raise ValueError('Orphaned OCR history')
    for frame,index in zip(row['source_frame_indices'],row['user_message_indices'],strict=True):
        text=messages[index]['content']
        if messages[index]['role']!='user' or not text.startswith('<image>'):
            raise ValueError('Unexpected image message')
        block=text.removeprefix('<image>').strip()
        if not block:continue
        events=source_events(block)
        if len(events)!=1 or events[0]['frame']!=frame:raise ValueError('OCR/image frame mismatch')
        converted=encoder.convert(block)
        messages[index]['content']='<image>'+('\n'+converted[0]['block'] if converted else '')
    if encoder.output is None:raise ValueError('Missing initial OCR state')
    revised=dict(row,messages=messages,ocr_history_json=json.dumps(history,ensure_ascii=False),
                 context_contract=row['context_contract']+'__'+VERSION)
    return revised

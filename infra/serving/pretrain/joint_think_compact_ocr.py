"""Lossless OCR subfield deltas and one envelope per causal archive.

Only the already selected L region is split. No OCR candidates, navigation
values, recognition jitter or historical events are discarded or normalized.
The original regional strings, including escapes and line order, reconstruct
exactly after every event. Existing think/state/action contracts are unchanged.
"""
from __future__ import annotations

import copy
import re
from pretrain import joint_think_protocol as p

VERSION = 'joint_think_rolling5_compact_ocr_v2'
SYSTEM_VERSION = 'joint_think_rolling5_compact_ocr_legend_v1'
FIELDS = ('P', 'U', 'L.text', 'L.nav', 'L.order', 'B', 'R', 'C', 'I', 'K')
OPEN, CLOSE = '<screen_ocr>\n', '\n</screen_ocr>'
HEADER = '此前已移除图片和动作的 OCR 历史（按帧序）：\n'
EVENT = re.compile(r'([1-9][0-9]*) (S|D)\Z')
# OCR sometimes renders the navigation arrow as t, t=, -= or ↑=.
# Group those lines with navigation too, but retain every original character.
DISTANCE = re.compile(r'[\s^<>=+\-~tTiIlL!:/\u2190-\u21ff\u2300-\u23ff\u25a0-\u27ff]*[0-9][0-9.,\s]*(?:km|m)\s*\Z', re.I)
LEGEND = '''OCR 代码说明：<screen_ocr> 内每条记录先写“帧号 S/D”，下一行起为“字段=文字”。历史中的多条记录共用一对 OCR 标记；没有 OCR 块表示文字未变化。
- S（快照）：新段的完整文字状态，未列字段为空；D（增量）：只替换列出字段的全部文字，其他字段沿用；例如 B= 表示底部文字消失。
- 区域代码：P=左上教程/交互提示；U=右上金钱/弹药；B=底部任务目标/对白字幕；R=手机；C=中央任务成功/失败；I=电脑任务控件；K=右下操作说明。
- 左下 L 分为 L.text（通知等非距离文字）和 L.nav（小地图导航距离及箭头）。距离变化只更新 L.nav，不重复通知；例如 L.nav=↑ 860m。距离和文字出现、变化、消失的过程均保留。
- 同字段多条文字用竖线分隔，文字本身的竖线和反斜线用反斜线转义。默认 L.text 各行在前、L.nav 各行在后；非默认排列用 L.order 列出距离行在原 L 区域中的位置（从 0 开始，逗号分隔），L.order= 恢复默认排列。此字段只记录阅读顺序。
- 帧号是源视频的观测顺序，相邻帧间隔 0.2 秒。仅提供所选有用区域；没有新增全屏招牌或广告。每次清空历史后，首帧重新给出 S，无文字也提供空 S。'''


def split_lines(value):
    """Split only unescaped pipes; leave each encoded line byte-for-byte intact."""
    if not value:
        return []
    result, start, escaped = [], 0, False
    for i, char in enumerate(value):
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '|':
            result.append(value[start:i]); start = i + 1
    result.append(value[start:])
    return result


def split_state(state):
    p.require(set(state) == set(p.ROIS), 'Unexpected regional OCR state')
    texts, nav, positions = [], [], []
    for index, line in enumerate(split_lines(state['L'])):
        if DISTANCE.fullmatch(line):
            nav.append(line); positions.append(index)
        else:
            texts.append(line)
    default = list(range(len(texts), len(texts) + len(nav)))
    return {key: state[key] for key in p.ROIS if key != 'L'} | {
        'L.text': '|'.join(texts), 'L.nav': '|'.join(nav),
        'L.order': ','.join(map(str, positions)) if positions != default else '',
    }


def join_state(state):
    p.require(set(state) == set(FIELDS), 'Unexpected compact OCR fields')
    texts, nav = split_lines(state['L.text']), split_lines(state['L.nav'])
    positions = ([int(x) for x in state['L.order'].split(',')] if state['L.order']
                 else list(range(len(texts), len(texts) + len(nav))))
    n = len(texts) + len(nav)
    p.require(len(positions) == len(nav) and positions == sorted(set(positions))
              and all(0 <= x < n for x in positions), 'Invalid L reading order')
    nav_by_position = dict(zip(positions, nav, strict=True)); source = iter(texts)
    lines = [nav_by_position[i] if i in nav_by_position else next(source) for i in range(n)]
    return {key: state[key] for key in p.ROIS if key != 'L'} | {'L': '|'.join(lines)}


def incremental(current, previous, frame):
    state = split_state(current)
    old = split_state(previous) if previous is not None else None
    p.require(join_state(state) == current, 'OCR split changed regional text/order')
    changed = {k: state[k] for k in FIELDS if (bool(state[k]) if old is None else state[k] != old[k])}
    if old is not None and not changed:
        return ''
    mode = 'S' if old is None else 'D'
    return OPEN + '\n'.join([f'{frame} {mode}'] + [k + '=' + v for k, v in changed.items()]) + CLOSE


def body(block):
    p.require(block.startswith(OPEN) and block.endswith(CLOSE), 'Invalid compact OCR envelope')
    return block[len(OPEN):-len(CLOSE)]


def parse_events(block):
    events, current, last = [], None, 0
    for line in body(block).splitlines():
        match = EVENT.fullmatch(line)
        if match:
            frame = int(match[1]); p.require(frame > last, 'Non-increasing OCR frames')
            current = dict(frame=frame, mode=match[2], changes={})
            events.append(current); last = frame
        else:
            key, sep, value = line.partition('=')
            p.require(current is not None and sep and key in FIELDS and key not in current['changes'],
                      'Invalid/duplicate compact OCR field')
            current['changes'][key] = value
    p.require(bool(events), 'Missing OCR event')
    return events


def replay_ocr(block, previous):
    if not block:
        p.require(previous is not None, 'Missing initial OCR snapshot')
        return dict(previous)
    state = split_state(previous) if previous is not None else None
    for event in parse_events(block):
        if event['mode'] == 'S':
            p.require(state is None, 'Snapshot unexpectedly resets nonempty OCR history')
            state = {key: '' for key in FIELDS}
        else:
            p.require(state is not None, 'Delta without initial snapshot')
        state.update(event['changes'])
        # Verify every intermediate state, including clears and order changes.
        join_state(state)
    return join_state(state)


def archive_message(records):
    p.require(bool(records), 'Empty OCR archive')
    previous_frame = 0
    bodies = []
    for record in records:
        events = parse_events(record['block'])
        p.require(len(events) == 1 and events[0]['frame'] == record['frame_index'] > previous_frame,
                  'Archive timestamp mismatch')
        previous_frame = record['frame_index']; bodies.append(body(record['block']))
    return HEADER + OPEN + '\n'.join(bodies) + CLOSE


def system_prompt(original):
    before, marker, tail = original.partition('OCR 代码说明：')
    _, end, after = tail.partition('\n\n图片与动作历史采用')
    p.require(bool(marker and end), 'Unknown source OCR legend')
    return before + LEGEND + end + after


def compress_segment(original):
    result = copy.deepcopy(original)
    result['id'] = original['id'].rsplit('__', 1)[0] + '__' + VERSION
    result['context_contract'] = VERSION
    result['messages'][0]['content'] = system_prompt(original['messages'][0]['content'])
    old = None; replayed = None; states = {}; events = 0
    for i, frame in enumerate(original['source_frame_indices']):
        user = original['messages'][1 + 2 * i]['content']
        block = user.removeprefix('<image>').strip()
        current = p.replay_ocr(block, old)
        compact = incremental(current, old, frame)
        p.require(bool(block) == bool(compact), 'Compression changed event timing')
        replayed = replay_ocr(compact, replayed)
        p.require(replayed == current, 'Compressed OCR differs from original frame')
        result['messages'][1 + 2 * i]['content'] = '<image>' + ('\n' + compact if compact else '')
        states[frame] = current; old = current; events += bool(compact)
    return result, states, events

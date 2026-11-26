"""Versioned joint think/state/OCR contract; does not change legacy exporters."""
from __future__ import annotations

import json
import re

VERSION = 'joint_think_event_reset_regional_incremental_ocr_v2'
SYSTEM_PROMPT_VERSION = 'joint_think_ocr_legend_v2'
STATE_KEYS = ('instruction', 'big_think', 'small_think', 'memory')
ROIS = ('P', 'U', 'L', 'B', 'R', 'C', 'I', 'K')
RESERVED = ('<think>', '</think>', '<a>', '</a>', '<image>', '<screen_ocr>', '</screen_ocr>', '<|')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def empty_state():
    return dict(instruction=None, big_think='', small_think='', memory='')


def validate_text(value):
    require(isinstance(value, str) and bool(value.strip()), 'Empty/non-string update')
    require(not any(token in value for token in RESERVED), 'Reserved token in state text')


def validate_update(update):
    require(isinstance(update, dict), 'Think must be an object')
    require(set(update) <= set(STATE_KEYS), 'Unknown think field')
    require(('instruction' in update) == ('big_think' in update), 'Big think/instruction must be paired')
    require(('small_think' in update) == ('memory' in update), 'Small think/memory must be paired')
    for key, value in update.items():
        if key == 'instruction' and value is None:
            continue
        # Empty memory is a valid explicit clear; absence means retain it.
        if key == 'memory' and value == '':
            continue
        validate_text(value)
    return update


def format_think(update):
    validate_update(update)
    return canonical(update) if update else ''


def parse_think(payload):
    if not payload.strip():
        return {}
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, 'Duplicate think field')
            obj[key] = value
        return obj
    return validate_update(json.loads(payload, object_pairs_hook=unique))


def event_updates(event):
    """Return supervised output and reducer update, including input-only fallback."""
    output, applied = {}, {}
    if event and event.get('big'):
        big = event['big']
        output.update(big_think=big['reasoning'], instruction=big['instruction'])
        applied.update(output)
    if event and event.get('small'):
        small = event['small']
        pair = dict(small_think=small['small_think'], memory=small['memory'])
        applied.update(pair)
        if not small['fallback']:
            output.update(pair)
    validate_update(output)
    validate_update(applied)
    return output, applied


def advance(state, event):
    _, update = event_updates(event)
    result = dict(state)
    result.update(update)
    return result


def assistant_output(action, update):
    require(re.fullmatch(r'<a>[^\s<>]+</a>', action) is not None, 'Invalid source action envelope')
    payload = format_think(update)
    return '<think>' + ('\n' + payload + '\n' if payload else '') + '</think>\n\n' + action


def build_system(source_system, state):
    require(set(state) == set(STATE_KEYS), 'Invalid state keys')
    base, marker, _ = source_system.partition('\n\n### 当前 instruction\n')
    require(bool(marker), 'Source action prompt anchor missing')
    return base + '''

每轮先输出 <think>...</think>，再输出 <a>...</a>。无需更新时，think 内容为空，仍须输出结束标记 </think>。
需要更新时，think 内输出 JSON 对象：大目标更新使用 big_think（简短判断）和 instruction（新目标，清空为 null）；局部更新使用 small_think（当前局部判断）和 memory（事实、状态与未知摘要）。每组字段成对出现，可在同一对象中同时更新两组；未出现的状态字段保持不变。memory 不是未来操作清单。
一次更新完成后清空对话历史；从下一轮起将更新状态放入 system，并从新观测开始。已有状态是过去观测的摘要，不是本轮需要重复输出的标签。只依据已观察到的内容判断，不使用未来画面或动作。
OCR 代码说明：<screen_ocr> 第一行的 S/D 表示更新方式；后续“区域代码=文字”中的 P、B 等表示画面区域。
- S（Snapshot，快照）：本段首帧的完整区域文字；未列出的区域为空。
- D（Delta，增量）：只替换列出区域的全部文字，其他区域沿用此前内容；例如 B= 表示底部文字消失。没有 OCR 块表示各区域文字均无变化。
- 区域代码：P=左上教程/交互提示；U=右上金钱/弹药；L=左下导航距离/通知；B=底部任务目标/对白字幕；R=手机界面；C=中央任务成功/失败等终态；I=电脑任务控件；K=右下操作说明。例如 P=按进入潜行模式。表示左上提示。
同一区域的多条文字用竖线分隔，文字本身的竖线和反斜线使用反斜线转义。仅保留上述有用区域；手机、终态和电脑文字须由当帧界面标识启用，不提供全屏场景招牌或广告。段内按时间保留 S/D 消息；每次清空历史后，新段首帧重新提供 S，无文字时也提供空 S。

当前有效状态（来自此前已完成的更新）：
''' + canonical(state)


def parse_source_ocr(user_text):
    """Keep source ROI strings/escaping exactly; do not re-OCR or debounce."""
    require(user_text.count('<image>') == 1, 'Expected one source image')
    matches = list(re.finditer(r'<screen_ocr>\n(.*?)\n</screen_ocr>', user_text, re.S))
    require(len(matches) <= 1, 'Multiple source OCR snapshots')
    rest = re.sub(r'<screen_ocr>\n.*?\n</screen_ocr>', '', user_text, flags=re.S)
    require(rest.strip() == '<image>', 'Unexpected source observation text')
    state = {key: '' for key in ROIS}
    if matches:
        lines = matches[0][1].splitlines()
        require(lines and lines[0] == 'S', 'Expected self-contained source S')
        seen = set()
        for line in lines[1:]:
            key, separator, value = line.partition('=')
            require(separator and key in ROIS and key not in seen, 'Invalid source OCR region')
            seen.add(key)
            state[key] = value
    return state


def incremental_ocr(current, previous):
    kind = 'S' if previous is None else 'D'
    changed = {key: current[key] for key in ROIS
               if (bool(current[key]) if previous is None else current[key] != previous[key])}
    if previous is not None and not changed:
        return '', None
    lines = [kind] + [key + '=' + value for key, value in changed.items()]
    return '<screen_ocr>\n' + '\n'.join(lines) + '\n</screen_ocr>', kind


def replay_ocr(block, previous):
    if not block:
        require(previous is not None, 'Missing initial OCR snapshot')
        return dict(previous)
    require(block.startswith('<screen_ocr>\n') and block.endswith('\n</screen_ocr>'), 'Invalid OCR block')
    lines = block[len('<screen_ocr>\n'):-len('\n</screen_ocr>')].splitlines()
    require(lines[0] in ('S', 'D'), 'Invalid OCR event kind')
    require(lines[0] == 'S' or previous is not None, 'Delta without snapshot')
    result = {key: '' for key in ROIS} if lines[0] == 'S' else dict(previous)
    seen = set()
    for line in lines[1:]:
        key, sep, value = line.partition('=')
        require(sep and key in ROIS and key not in seen, 'Invalid delta region')
        result[key] = value
        seen.add(key)
    return result

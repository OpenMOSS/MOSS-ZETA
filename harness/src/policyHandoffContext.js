const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { imagePathToBuffer } = require('./captureStore');
const { stateFromSystem } = require('./policyHistoryRecovery');
const { extractPolicyResponse } = require('./policyResponse');
const { parseHighFrequencyPackage } = require('../../cadence/src/highFrequencyActionParser');

const VERSION = 'session-gpt-authoritative-history-v1';
const FORMAT = 'think_memory_minimap_compact_v1';
const RENDERING = 'think-memory-minimap-compact-v1';
const TRANSFER = 'generated-dialogue-transfer-v1';
const REPLACE = 'same-session-history-replace-v1';
const ARCHIVE = '此前已移除图片和动作的 OCR 历史（按出现顺序）：\n';
const TEMPLATE = fs.readFileSync(path.join(__dirname, '../prompts/policy-h16k8-handoff-system.txt'), 'utf8');
const enabled = config => Boolean(config?.policyManualSwitch?.enabled &&
  (config.policyTransport === 'lumine_session' || config.policyManualSwitch.targets?.small?.policyTransport === 'lumine_session'));

function systemMessage(host) {
  const state = { instruction: host.currentSession?.currentPolicyInstruction?.text || host.taskInstruction || '暂无明确的阶段目标，需要根据后续观测确认。',
    memory: host.currentSession?.brain?.memory || '', model_thought: host.currentSession?.brain?.think || '' };
  let text = TEMPLATE;
  for (const [key, value] of Object.entries(state)) text = text.replaceAll(`{${key}}`, value);
  return { role: 'system', content: text.trim() };
}

function validateAssistant(content) {
  const parsed = extractPolicyResponse(content, '', RENDERING);
  if (content !== `<think>${content.slice(7, content.indexOf('</think>'))}</think>${parsed.content}`)
    throw Error('Handoff requires the exact compact assistant envelope');
  const action = parseHighFrequencyPackage(parsed.content, { chunksPerPackage: 6, chunkDurationMs: 200 / 6 });
  const chunks = action.actionText.replace(/^[+-]\d[+-]\d[+-]\d[+-]\d/, '').split(';');
  if (chunks.length !== 6 || chunks.slice(0, 3).some(c => c !== chunks[0]) || chunks.slice(3).some(c => c !== chunks[3]))
    throw Error('Frozen Policy profile requires two ordered 100ms button states');
  const order = ['LT', 'LB', 'L3', 'view', 'up', 'down', 'left', 'right', 'RT', 'RB', 'R3', 'menu', 'X', 'Y', 'B', 'A'];
  for (const chunk of chunks) {
    const tokens = chunk.match(/LT|LB|L3|view|up|down|left|right|RT|RB|R3|menu|X|Y|B|A/g) || [];
    if (tokens.join('') !== chunk || order.filter(t => tokens.includes(t)).join('') !== chunk)
      throw Error('Frozen Policy profile requires canonical button order without duplicates');
  }
  return parsed;
}

async function minimap(observation, python = 'python3') {
  if (observation.handoffMinimap) return observation.handoffMinimap;
  const bytes = observation.buffer || await imagePathToBuffer(observation.imagePath);
  const buffer = await new Promise((resolve, reject) => {
    const proc = spawn(python, [path.join(__dirname, '../scripts/policy-handoff-image.py')], { stdio: ['pipe', 'pipe', 'pipe'] });
    const out = [], err = []; proc.stdout.on('data', b => out.push(b)); proc.stderr.on('data', b => err.push(b));
    proc.once('error', reject); proc.once('close', code => code === 0 ? resolve(Buffer.concat(out)) : reject(Error('Frozen minimap conversion failed: ' + Buffer.concat(err).toString().slice(-1200))));
    proc.stdin.on('error', reject); proc.stdin.end(bytes);
  });
  const imagePath = path.join(path.dirname(observation.imagePath), `${path.parse(observation.imagePath).name}-handoff-minimap.png`);
  await fsp.writeFile(imagePath, buffer);
  observation.handoffMinimap = { imagePath, mimeType: 'image/png', width: 512, height: 384, capturedAt: observation.capturedAt };
  return observation.handoffMinimap;
}

// Both targets first receive this same, checkpoint-rendered logical transcript.
// Empty aggregate remains in the harness, but the frozen h16/k8 model layout
// omits it until an S/D event has left the image window. Apply this to BOTH APIs.
async function materialize(messages) {
  const result = [];
  for (const m of messages) {
    if (m.ocrAggregate && !(m.ocrRecords?.length)) continue;
    if (!Array.isArray(m.content)) { result.push({ role: m.role, content: m.content }); continue; }
    const pictures = m.content.filter(p => p.type === 'local_image');
    if (pictures.length !== 2 || pictures[0].imageKind !== 'full_frame' || pictures[1].imageKind !== 'minimap')
      throw Error('Logical handoff observation requires original frame plus its paired minimap');
    const urls = await Promise.all(pictures.map(async p => ({ type: 'image_url', image_url: { url:
      `data:${p.mimeType};base64,${(await imagePathToBuffer(p.imagePath)).toString('base64')}` } })));
    const block = m.content.filter(p => p.type === 'text').map(p => p.text).join('').trim();
    result.push({ role: m.role, content: [{ type: 'text', text: block + '\n完整游戏画面：' }, urls[0],
      { type: 'text', text: '\n同帧小地图放大图：' }, urls[1]] });
  }
  return result;
}

function nativeMessages(messages) {
  return messages.map(m => {
    if (!Array.isArray(m.content)) return { role: m.role, content: m.content };
    const images = m.content.filter(p => p.type === 'image_url');
    if (images.length !== 2) throw Error('Paired model input missing a main image or minimap');
    const text = m.content[0]?.text;
    if (typeof text !== 'string' || !text.endsWith('\n完整游戏画面：')) throw Error('Missing frozen observation label');
    return { role: 'user', content: [{ type: 'text', text: text.slice(0, -'\n完整游戏画面：'.length) }, images[0]] };
  });
}
function historyPairs(messages) {
  const rest = messages.filter(m => m.role !== 'system' && !(m.role === 'user' && typeof m.content === 'string'));
  if (!rest.length || rest.length % 2 !== 1 || rest.at(-1).role !== 'user') throw Error('Expected complete history pairs followed by one new observation');
  for (let i = 0; i < rest.length - 1; i += 2) {
    if (rest[i].role !== 'user' || rest[i + 1].role !== 'assistant') throw Error('Incomplete generated Policy history');
    validateAssistant(rest[i + 1].content);
  }
  return (rest.length - 1) / 2;
}
module.exports = { VERSION, FORMAT, RENDERING, TRANSFER, REPLACE, ARCHIVE, TEMPLATE,
  enabled, systemMessage, stateFromSystem, validateAssistant, minimap, materialize, nativeMessages, historyPairs };

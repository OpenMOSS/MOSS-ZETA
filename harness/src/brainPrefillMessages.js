const { createHash } = require('node:crypto');
const { imagePathToDataUrl } = require('./captureStore');
const { MEMORY_PROMPT, FALLBACK_PROMPT, FULL_PROMPT, outputKeys } = require('./brainProtocol');

const VERSION = 'brain-causal-prefix-v1';
const hash = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');
const clone = value => structuredClone(value);

function stableSystem(config, documents) {
  const prompts = { small: documents.get('brainMemorySystemPrompt') || MEMORY_PROMPT,
    fallback: config.brainFallbackScope === 'full' ? (documents.get('brainSystemPrompt') || FULL_PROMPT) : FALLBACK_PROMPT, full: documents.get('brainSystemPrompt') || FULL_PROMPT };
  return [
    `你是 GTA Brain。前面的完整 user 消息只提供已冻结的因果证据，不是更新请求。最后一条 user 消息提供当前权威 previous、updateType 和触发信息；只执行该 updateType 对应的规则。small 只返回 memory；fallback ${config.brainFallbackScope === 'full' ? '返回 instruction/memory/think' : '只返回 memory/think 且固定 instruction'}；full 返回 instruction/memory/think。不得输出其他字段。未执行的 Policy 提议不是已执行事实。`,
    ...['small', 'fallback', 'full'].map(type => `以下规则仅适用于 updateType=${type}：\n${prompts[type]}`),
    documents.get('gtaKnowledge'),
    `Executed packages use signed Lx Ly Rx Ry levels -9..+9 and ${config.hfChunksPerPackage || 6} button chunks of ${config.hfChunkDurationMs || 33}ms. Images and completed inputs are evidence, not proof of successful outcomes. Explicit observation IDs/timestamps determine chronology, including an older Policy source repeated in the final message.`
  ].filter(Boolean).join('\n\n');
}

function observationView(record) {
  return { type: 'observation', observationId: record.observationId, actionSeq: Number(record.actionSeq ?? record.actionCount),
    imagePath: record.imagePath, mimeType: record.mimeType || 'image/jpeg', capturedAt: record.capturedAt || record.at,
    captureInvalidated: Boolean(record.captureInvalidated), executionAtCapture: clone(record.executionAtCapture || null),
    ocr: { results: (record.ocr?.results || []).filter(row => String(row.text || '').trim()).map(row => ({
      roi: row.roi, text: row.text, ...(Number.isFinite(row.confidence ?? row.score) ? { confidence: row.confidence ?? row.score } : {}) })) },
    navigation: record.navigation?.minimap?.imagePath ? { minimap: { imagePath: record.navigation.minimap.imagePath,
      mimeType: record.navigation.minimap.mimeType || 'image/jpeg' } } : null };
}

// Called synchronously only for a bounded set of already selected observations.
// No disk IO, image encoding, or full timeline copying occurs at offer time.
function freezeSnapshot(snapshot) {
  if (!snapshot?.sessionId || !Number.isInteger(snapshot.brainRevision) || snapshot.brainRevision < 0
    || !Number.isInteger(snapshot.cutoffActionSeq) || snapshot.cutoffActionSeq < 0) throw Error('Invalid Brain prefill identity/cutoff');
  const observations = snapshot.trajectory || snapshot.observations || [];
  if (!observations.length || observations.length > 8) throw Error('Brain prefill requires 1..8 historical screenshots');
  const previousActionSeq = Number(snapshot.previousActionSeq || 0);
  const offeredAt = snapshot.offeredAt || new Date().toISOString();
  const offeredMs = Date.parse(offeredAt);
  const trajectory = observations.map(observationView);
  const ids = new Set();
  for (const observation of trajectory) {
    if (!observation.observationId || ids.has(observation.observationId) || !observation.imagePath || observation.captureInvalidated
      || !Number.isInteger(observation.actionSeq) || observation.actionSeq > snapshot.cutoffActionSeq
      || !Number.isFinite(Date.parse(observation.capturedAt)) || Date.parse(observation.capturedAt) > offeredMs) {
      throw Error('Brain prefill contains invalid, duplicated, or future observation evidence');
    }
    ids.add(observation.observationId);
  }
  trajectory.sort((a, b) => Date.parse(a.capturedAt) - Date.parse(b.capturedAt));
  const evidenceRecords = [];
  for (const record of snapshot.evidenceRecords || []) {
    if (!['executed_action', 'interrupted_action'].includes(record.type)) continue;
    const seq = Number(record.actionSeq);
    if (!Number.isInteger(seq) || seq > snapshot.cutoffActionSeq) throw Error('Brain prefill action exceeds causal cutoff');
    if (record.type === 'executed_action' && record.completed === false) throw Error('Incomplete action is not executed evidence');
    const completedAt = record.execution?.completedAt || record.completedAt || record.at;
    if (completedAt && Date.parse(completedAt) > offeredMs) throw Error('Brain prefill contains future completion evidence');
    if (seq < previousActionSeq || (record.type === 'executed_action' && seq === previousActionSeq)) continue;
    evidenceRecords.push({ type: record.type, actionId: record.actionId, actionSeq: seq,
      basedOnObservationId: record.basedOnObservationId, raw: record.raw || record.actionText,
      requestedPackage: record.requestedPackage, completedAt,
      ...(record.type === 'interrupted_action' ? { completed: false, executedChunkCount: record.executedChunkCount ?? null, actualSegments: record.actualSegments || [] } : {}) });
  }
  return { sessionId: snapshot.sessionId, brainRevision: snapshot.brainRevision, cutoffActionSeq: snapshot.cutoffActionSeq,
    basedOnObservationId: snapshot.basedOnObservationId, previousActionSeq, offeredAt, trajectory, evidenceRecords,
    snapshotPreparationMs: Number.isFinite(snapshot.snapshotPreparationMs) ? snapshot.snapshotPreparationMs : null };
}

function evidenceContent(snapshot, { sourceObservationId = null, frameOffset = 0 } = {}) {
  const content = [], evidence = { observations: [], actions: [], interruptedActions: [], imageCount: 0 };
  const consumed = new Set();
  const addActions = (maximum, beforeFrame) => {
    const completed = snapshot.evidenceRecords.filter((record, index) => record.type === 'executed_action'
      && record.actionSeq <= maximum && !consumed.has(index));
    if (completed.length) {
      content.push({ type: 'text', text: JSON.stringify({ executedSincePreviousObservation: completed.map(record => ({
        actionId: record.actionId, actionSeq: record.actionSeq, package: record.raw, completedAt: record.completedAt })) }) });
      evidence.actions.push(...completed.map(record => ({ actionId: record.actionId, actionSeq: record.actionSeq,
        basedOnObservationId: record.basedOnObservationId, beforeFrame })));
      snapshot.evidenceRecords.forEach((record, index) => { if (completed.includes(record)) consumed.add(index); });
    }
  };
  for (const [index, record] of snapshot.trajectory.entries()) {
    const frame = frameOffset + index + 1;
    addActions(record.actionSeq, frame);
    const interrupted = snapshot.evidenceRecords.filter((item, position) => item.type === 'interrupted_action'
      && !consumed.has(position) && Date.parse(item.completedAt) <= Date.parse(record.capturedAt));
    if (interrupted.length) {
      content.push({ type: 'text', text: JSON.stringify({ interruptedActions: interrupted,
        note: 'Partial requested packages are not fully executed evidence.' }) });
      evidence.interruptedActions.push(...interrupted);
      snapshot.evidenceRecords.forEach((item, position) => { if (interrupted.includes(item)) consumed.add(position); });
    }
    const source = record.observationId === sourceObservationId;
    content.push({ type: 'text', text: JSON.stringify({ frame, observationId: record.observationId,
      capturedAt: record.capturedAt, actionSeq: record.actionSeq, ...(source ? { isPolicyThinkSource: true } : {}),
      ...(record.ocr.results.length ? { ocr: record.ocr.results } : {}),
      ...(record.executionAtCapture?.activePackageSeq ? { actionInProgress: true,
        note: 'An input package was still executing; its full result is not yet visible.' } : {}) }) });
    content.push({ type: 'image_ref', path: record.imagePath, mimeType: record.mimeType });
    evidence.observations.push({ observationId: record.observationId, actionSeq: record.actionSeq, frame,
      isPolicyThinkSource: source, imagePath: record.imagePath, capturedAt: record.capturedAt,
      minimapPath: record.navigation?.minimap?.imagePath || null, hasMinimap: Boolean(record.navigation?.minimap?.imagePath) });
    if (record.navigation?.minimap?.imagePath) content.push({ type: 'text', text: `Minimap crop from frame ${frame}, not a new observation.` },
      { type: 'image_ref', path: record.navigation.minimap.imagePath, mimeType: record.navigation.minimap.mimeType });
  }
  // Completions after the last image remain explicitly after that observation.
  addActions(snapshot.cutoffActionSeq, null);
  const remainingInterrupted = snapshot.evidenceRecords.filter((record, index) => record.type === 'interrupted_action' && !consumed.has(index));
  if (remainingInterrupted.length) {
    content.push({ type: 'text', text: JSON.stringify({ interruptedAfterLastObservation: remainingInterrupted }) });
    evidence.interruptedActions.push(...remainingInterrupted);
  }
  evidence.imageCount = content.filter(part => part.type === 'image_ref').length;
  return { content, evidence };
}

async function materialize(messages, signal) {
  const result = [], images = [];
  for (const message of messages) {
    if (!Array.isArray(message.content)) { result.push({ ...message }); continue; }
    const content = [];
    for (const part of message.content) {
      signal?.throwIfAborted();
      if (part.type !== 'image_ref') { content.push({ ...part }); continue; }
      const url = await imagePathToDataUrl(part.path, part.mimeType);
      signal?.throwIfAborted();
      images.push({ imagePath: part.path, dataUrlSha256: hash(url) });
      content.push({ type: 'image_url', image_url: { url } });
    }
    result.push({ role: message.role, content });
  }
  return { messages: result, images };
}

async function buildPrefix(snapshot, system, signal) {
  const { content, evidence } = evidenceContent(snapshot);
  content.unshift({ type: 'text', text: JSON.stringify({ evidencePrefix: VERSION, sessionId: snapshot.sessionId,
    brainRevision: snapshot.brainRevision, cutoffActionSeq: snapshot.cutoffActionSeq,
    note: 'Frozen historical evidence only. The final request supplies the current authoritative state and update type.' }) });
  const referenceMessages = [{ role: 'system', content: system }, { role: 'user', content }];
  const { messages, images } = await materialize(referenceMessages, signal);
  return { snapshot, messages, referenceMessages, inputEvidence: evidence, images,
    prefixHash: hash(messages), systemSha256: hash(system), version: VERSION };
}

async function buildFinal(prefix, request, state, signal) {
  const records = request.evidenceRecords || state.timeline.readRecords();
  const available = [...(request.trajectory || []), ...records.filter(record => record.type === 'observation')];
  const ids = [...new Set([request.basedOnObservationId, request.policyEvidence?.observationId].filter(Boolean))];
  const trajectory = ids.map(id => available.find(record => record.observationId === id && record.imagePath && !record.captureInvalidated));
  if (trajectory.some(record => !record)) throw Error('Brain final request is missing its latest trigger or Policy source screenshot');
  const suffixSnapshot = freezeSnapshot({ sessionId: state.sessionId, brainRevision: state.brain.revision,
    cutoffActionSeq: request.cutoffActionSeq, previousActionSeq: prefix.snapshot.cutoffActionSeq,
    basedOnObservationId: request.basedOnObservationId, trajectory, evidenceRecords: records });
  const { content, evidence } = evidenceContent(suffixSnapshot, { sourceObservationId: request.policyEvidence?.observationId,
    frameOffset: prefix.inputEvidence.observations.length });
  const view = { goal: state.userGoal, previous: clone(request.previous || require('./brainSchema').brainView(state)),
    updateType: request.updateType || 'full', requiredOutputKeys: outputKeys(request.updateType || 'full', request.brainFallbackScope),
    basedOnObservationId: request.basedOnObservationId, cutoffActionSeq: request.cutoffActionSeq,
    ...(request.policyTriggerThinking ? { trigger: request.policyTriggerThinking } : {}),
    ...(request.policyEvidence ? { policyEvidence: { ...request.policyEvidence,
      note: 'An observation-based judgment; the associated action may be unexecuted. Only completed action evidence proves an input occurred.' } } : {}),
    ...(request.policyProposal ? { policyProposal: { ...request.policyProposal, note: 'An uncommitted proposal, not independent observation evidence.' } } : {}),
    ...(state.recoveryMode ? { recovery: 'Inspect latest HUD/task outcome; old instruction may be invalid.' } : {}) };
  content.push({ type: 'text', text: JSON.stringify(view) });
  const suffix = [{ role: 'user', content }];
  const materialized = await materialize(suffix, signal);
  return { messages: [...prefix.messages, ...materialized.messages], referenceMessages: [...prefix.referenceMessages, ...suffix],
    inputEvidence: { observations: [...prefix.inputEvidence.observations, ...evidence.observations],
      actions: [...prefix.inputEvidence.actions, ...evidence.actions],
      interruptedActions: [...prefix.inputEvidence.interruptedActions, ...evidence.interruptedActions],
      imageCount: prefix.inputEvidence.imageCount + evidence.imageCount },
    images: [...prefix.images, ...materialized.images], view };
}

function prefixEligible(prefix, request, state, system) {
  if (!prefix || prefix.snapshot.sessionId !== state.sessionId || prefix.snapshot.brainRevision !== state.brain.revision
    || prefix.snapshot.cutoffActionSeq > request.cutoffActionSeq || prefix.systemSha256 !== hash(system)) return false;
  const trigger = [...(request.trajectory || []), ...(request.evidenceRecords || [])]
    .find(record => record.observationId === request.basedOnObservationId);
  if (!trigger) return false;
  const triggerAt = Date.parse(trigger.capturedAt || trigger.at);
  return Number.isFinite(triggerAt) && prefix.snapshot.trajectory.every(record => Date.parse(record.capturedAt) <= triggerAt);
}

module.exports = { VERSION, hash, stableSystem, freezeSnapshot, evidenceContent, buildPrefix, buildFinal, prefixEligible };

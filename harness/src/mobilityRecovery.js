const { neutralControllerState, postprocessPausedControllerState } = require('./actionRuntime');

const WALK = { forward: [0, 1], backward: [0, -1], left: [-1, 0], right: [1, 0],
  forward_left: [-1, 1], forward_right: [1, 1], backward_left: [-1, -1], backward_right: [1, -1] };
const DRIVE = ['forward', 'forward_left', 'forward_right', 'reverse', 'reverse_left', 'reverse_right'];
function recoverySettings(raw = {}) {
  const bounded = (key, def, min, max) => {
    const v = Number(raw[key] ?? def);
    if (!Number.isFinite(v) || v < min || v > max) throw Error(`Invalid recovery ${key}`);
    return v;
  };
  const integer = (key, def, min, max) => {
    const value = bounded(key, def, min, max);
    if (!Number.isInteger(value)) throw Error(`Invalid recovery ${key}: integer required`);
    return value;
  };
  return { enabled: raw.enabled === true, walkStepMs: bounded('walkStepMs', 500, 50, 3000),
    driveStepMs: bounded('driveStepMs', 1000, 50, 3000), brakeMs: bounded('brakeMs', 500, 50, 3000),
    walkMagnitude: bounded('walkMagnitude', 0.7, 0.3, 1), steeringMagnitude: bounded('steeringMagnitude', 0.6, 0.3, 1),
    triggerMagnitude: bounded('triggerMagnitude', 0.6, 0.1, 1), maxSegments: integer('maxSegments', 3, 1, 6),
    maxCallsPerEpisode: integer('maxCallsPerEpisode', 3, 1, 10), cooldownMs: bounded('cooldownMs', 5000, 0, 60000),
    maxDurationMs: bounded('maxDurationMs', 10000, 50, 30000) };
}
function validateRecovery(request, config) {
  if (!request || !['on_foot', 'driving'].includes(request.mode) || !['directed', 'scripted'].includes(request.strategy)
    || Object.keys(request).some(k => !['mode', 'strategy', 'moves', 'reason'].includes(k))) throw Error('Invalid recover_mobility arguments');
  if (request.strategy === 'scripted') {
    if (request.moves !== undefined) throw Error('Scripted recovery cannot accept model directions');
  } else if (!Array.isArray(request.moves) || request.moves.length < 1 || request.moves.length > config.maxSegments
    || request.moves.some(m => request.mode === 'on_foot' ? !Object.hasOwn(WALK, m) : !DRIVE.includes(m))) throw Error('Invalid recovery moves');
  if (request.reason !== undefined && (typeof request.reason !== 'string' || request.reason.length > 1000)) throw Error('Invalid recovery reason');
  return structuredClone(request);
}
function recoveryPlan(request, config, { attempt = 0, lastState } = {}) {
  validateRecovery(request, config);
  let moves = request.moves;
  if (request.strategy === 'scripted') {
    if (request.mode === 'driving') moves = attempt % 2 ? ['reverse_right', 'forward_left'] : ['reverse_left', 'forward_right'];
    else {
      let x = lastState?.leftStick?.x || 0, y = lastState?.leftStick?.y || 0;
      if (Math.hypot(x, y) < 0.1) { x = 0; y = 1; }
      const sign = attempt % 2 ? -1 : 1;
      moves = [[-x, -y], [-y * sign, x * sign], [x, y]];
    }
  }
  const segments = [];
  let lastDirection = null;
  for (const move of moves.slice(0, Math.floor(config.maxSegments))) {
    let state = neutralControllerState();
    if (request.mode === 'on_foot') {
      const [x, y] = Array.isArray(move) ? move : WALK[move];
      const r = Math.hypot(x, y);
      state.leftStick = { x: x / r * config.walkMagnitude, y: y / r * config.walkMagnitude };
      segments.push({ move, state: postprocessPausedControllerState(state), durationMs: config.walkStepMs });
    } else {
      const direction = move.startsWith('reverse') ? 'reverse' : 'forward';
      if (direction !== lastDirection) {
        // Brake opposite the previous travel direction. On entry braking from
        // forward motion is the only default; Brain must identify driving mode.
        const brake = neutralControllerState();
        brake[lastDirection === 'reverse' ? 'rightTrigger' : 'leftTrigger'] = config.triggerMagnitude;
        segments.push({ move: 'brake', state: brake, durationMs: config.brakeMs });
      }
      state[direction === 'reverse' ? 'leftTrigger' : 'rightTrigger'] = config.triggerMagnitude;
      state.leftStick.x = move.endsWith('_left') ? -config.steeringMagnitude : move.endsWith('_right') ? config.steeringMagnitude : 0;
      segments.push({ move, state, durationMs: config.driveStepMs });
      lastDirection = direction;
    }
  }
  if (segments.reduce((n, s) => n + s.durationMs, 0) > config.maxDurationMs) throw Error('Recovery duration budget exceeded');
  return segments;
}

// One executor, fixed configuration, no visual classifier or hidden correction.
async function executeRecovery({ request, config, runtime, shouldStop, record, prepare = async () => {}, interrupted = async () => {}, context = {} }) {
  const segments = recoveryPlan(request, config, context);
  const completed = [];
  let primary;
  try {
    await runtime.reset('brain_recovery_begin');
    for (const [index, segment] of segments.entries()) {
      if (shouldStop()) throw Object.assign(Error('Recovery cancelled'), { name: 'AbortError' });
      // A recovery segment applies one fixed input across six chunks.
      const chunks = 6, chunkDurationMs = segment.durationMs / chunks;
      const planned = { candidateId: `brain-recovery-${context.callId}-${index}`, origin: 'brain_recovery',
        segment: index, move: segment.move, requestedDurationMs: segment.durationMs, appliedState: segment.state,
        executorChunks: chunks, executorChunkDurationMs: chunkDurationMs };
      await prepare(planned);
      try {
        const ok = await runtime.executeRecoveryPackage(Array.from({ length: chunks }, () => structuredClone(segment.state)), { chunkDurationMs,
          carryEnabled: false, candidateId: planned.candidateId, shouldStop });
        if (!ok || shouldStop()) throw Object.assign(Error('Recovery interrupted'), { name: 'AbortError' });
      } catch (error) {
        try { await interrupted({ ...planned, completed: false, executionStatus: 'unknown_or_partial', message: error.message }); }
        catch (loggingError) { error.cleanupErrors = [...(error.cleanupErrors || []), { phase: 'recovery_interruption_evidence', message: loggingError.message }]; }
        throw error;
      }
      const evidence = { ...planned, completed: true };
      completed.push(evidence);
      await record(evidence);
    }
    return { status: 'inputs_completed', effectConfirmed: false, completed };
  } catch (error) { primary = error; throw error; }
  finally {
    try { await runtime.reset('brain_recovery_end'); }
    catch (error) { if (!primary) throw error; primary.cleanupErrors = [...(primary.cleanupErrors || []), { phase: 'recovery_release', message: error.message }]; }
  }
}
const RECOVERY_PROMPT = `可用工具 recover_mobility（structured-v1）。本节是上文普通状态JSON输出规则的显式例外：每次回复在独占工具请求JSON与正常状态JSON中二选一，不可混合。工具用于执行一次有界移动探测，再由新图核实效果。当多帧与已执行的重复移动明确显示无净位移、目标尚未达成，且当前可正常徒步/驾车时，优先调用工具做短探测，不要仅在think里写“后退或绕开”就当作受阻已处理。仍有实际推进、证据不足或仅需改变任务目标时，正常更新状态，不调用工具。工具请求必须独占JSON：{"tool_request":{"name":"recover_mobility","arguments":{"mode":"on_foot","strategy":"directed","moves":["backward","left"],"reason":"证据支持的受阻原因"}}}。徒步moves可选forward/backward/left/right/forward_left/forward_right/backward_left/backward_right；驾车可选forward/forward_left/forward_right/reverse/reverse_left/reverse_right。方向相对最新画面，不能把未知后方当成畅通。策略scripted不提供moves，执行固定规则候选。时长、力度、段数与调用上限由运行时配置，禁止自行增加字段。不要在菜单、剧情、死亡、瞄准射击或不明控制状态下调用。工具只证明输入是否执行，不能把执行完成写成脱困；工具执行完成且释放确认后，Framework保留原Policy session、instruction/memory/think和全部有效历史，仅追加正常新观察继续Policy；不重建上下文、不插入工具说明、不伪造assistant动作，也不追加Brain复核，不把输入完成当作脱困成功。后续正常状态更新仍用原JSON协议。`;
module.exports = { recoverySettings, validateRecovery, recoveryPlan, executeRecovery, RECOVERY_PROMPT };

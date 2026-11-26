const { randomUUID, createHash } = require('node:crypto');
const { executeRecovery, validateRecovery } = require('./mobilityRecovery');

async function changePolicyTarget(agent, command, commit) {
  if (agent.manualPolicy.sessionBacked) return changeSessionPolicyTarget(agent, command, commit);
  const signature = agent.policyStateSignature();
  await agent.policyBoundary('manual Policy target change');
  await agent.inFlightPrediction;
  const wasPaused = agent.gamePaused;
  if (agent.manualPolicy.settings.pauseDuringSwitch && !wasPaused) await agent.pauseGame('policy_target_change');
  if (agent.stopRequested || agent.paused || signature !== agent.policyStateSignature()) throw Error('Policy handover invalidated');
  // The aborted stateless request owns no remote history. Remove its unmatched
  // observation, but preserve every fully received assistant exactly as stored.
  while (agent.messages.at(-1)?.role === 'user' && !agent.messages.at(-1)?.ocrAggregate) agent.messages.pop();
  const snapshot = await agent.buildRequestMessages(agent.messages);
  const historySha256 = createHash('sha256').update(JSON.stringify(snapshot)).digest('hex');
  if (agent.stopRequested || agent.paused || signature !== agent.policyStateSignature()) throw Error('Policy handover invalidated during snapshot');
  commit();
  agent.currentModel = agent.manualPolicy.activeConfig().modelName;
  agent.frameworkRuntime.policyActor.setModel(agent.currentModel);
  await agent.recordHistoryEvent('policy_target_context_preserved', { revision: command.revision,
    historySha256, messageCount: snapshot.length, aggregatePreserved: true, contextOwner: 'harness' });
  if (!wasPaused && agent.gamePaused && !agent.stopRequested && !agent.paused) await agent.resumeGame('policy_target_change');
  if (!agent.stopRequested && !agent.paused) agent.executor.resumeAccepting();
}

async function changeSessionPolicyTarget(agent, command, commit) {
  // Stop dispatching, but settle the existing full reply and its last action.
  // Client abort alone cannot establish whether a remote turn committed.
  agent.manualPolicy.stage = 'waiting_reply';
  await agent.inFlightPrediction;
  if (agent.stopRequested || agent.paused) throw Error('Policy handoff stopped');
  const brainPending = () => agent.pendingHighFrequencyBarrier || agent.pendingAutomaticBrainFallback || agent.brainCycle
    || agent.frameworkRuntime.brainRuntime.scheduler.pending || agent.frameworkRuntime.brainRuntime.scheduler.inferencePromise;
  if (brainPending())
    throw Object.assign(Error('Brain state must commit before Policy handoff'), { handoffDeferred: true });
  agent.manualPolicy.stage = 'draining_actions';
  await agent.executor.waitForIdle();
  await agent.actionWrites;
  if (agent.stopRequested || agent.paused) throw Error('Policy handoff stopped during action drain');
  if (brainPending()) throw Object.assign(Error('Brain update triggered by the final action must commit first'), { handoffDeferred: true });
  const signature = agent.policyStateSignature();
  agent.manualPolicy.stage = 'releasing_input';
  await agent.policyBoundary('manual session/GPT handoff', { preservePolicySession: true });
  await agent.frameworkRuntime.actionRuntime.reset('manual_policy_handoff_release');
  const wasPaused = agent.gamePaused;
  if (agent.manualPolicy.settings.pauseDuringSwitch && !wasPaused) await agent.pauseGame('policy_target_change');
  if (agent.stopRequested || agent.paused || signature !== agent.policyStateSignature()) throw Error('Policy handoff invalidated');
  agent.manualPolicy.stage = 'handoff_context';
  const snapshot = await agent.buildRequestMessages(agent.messages);
  if (snapshot.at(-1)?.role !== 'assistant' && snapshot.some(m => m.role === 'assistant')) throw Error('Policy handoff has an uncommitted observation');
  if (agent.stopRequested || agent.paused || signature !== agent.policyStateSignature()) throw Error('Policy handoff invalidated during snapshot');
  const transport = agent.frameworkRuntime.policyActor.transport;
  transport.selectTarget(command.target);
  commit();
  agent.currentModel = agent.manualPolicy.activeConfig().modelName;
  agent.frameworkRuntime.policyActor.setModel(agent.currentModel);
  await agent.recordHistoryEvent('policy_target_context_preserved', { revision: command.revision,
    ...transport.snapshot(), historySha256: createHash('sha256').update(JSON.stringify(snapshot)).digest('hex'),
    messageCount: snapshot.length, aggregatePreserved: true, contextOwner: 'harness' });
  agent.manualPolicy.stage = 'resuming';
  if (!wasPaused && agent.gamePaused && !agent.stopRequested && !agent.paused) await agent.resumeGame('policy_target_change');
  if (!agent.stopRequested && !agent.paused) agent.executor.resumeAccepting();
}

// Quiesce generation/execution without aborting or resetting a healthy session.
async function settlePolicyForRecovery(agent, reason, cancelled = () => agent.stopRequested || agent.paused, beforeRelease) {
  const actor = agent.frameworkRuntime.policyActor;
  const sessionId = actor?.transport?.sessionId, epoch = agent.contextEpoch;
  await agent.inFlightPrediction;
  if (cancelled()) throw Object.assign(Error('Recovery cancelled while settling Policy'), { name: 'AbortError' });
  await agent.executor?.waitForIdle();
  await agent.actionWrites;
  // Protection can be preparing a release/menu boundary concurrently with the
  // Brain choosing a tool. Finish it before unpausing for physical tool input.
  await agent.brainProtectionBoundary;
  if (cancelled()) throw Object.assign(Error('Recovery cancelled while settling actions'), { name: 'AbortError' });
  if (epoch !== agent.contextEpoch || sessionId !== actor?.transport?.sessionId)
    throw Error('Policy context changed while settling recovery; original session cannot be preserved');
  if(agent.executor)agent.executor.accepting = false;
  let failure;
  try { await beforeRelease?.(); }
  catch (error) { failure = error; }
  try { await agent.frameworkRuntime.actionRuntime.reset(reason); }
  catch (error) {
    if (!failure) failure = error;
    else failure.cleanupErrors = [...(failure.cleanupErrors || []), { phase: 'recovery_boundary_release', message: error.message }];
  }
  if (failure) throw failure;
}

async function performRecovery(agent, request, signal) {
  const config = agent.defaultConfig.mobilityRecovery;
  if (!config?.enabled) throw Error('Mobility recovery is disabled');
  validateRecovery(request, config);
  if (!agent.executor || agent.defaultConfig.policyStreaming || agent.defaultConfig.policyHotSwapEnabled)
    throw Error('Mobility recovery requires the non-streamed high-frequency executor');
  if (agent.taskState !== 'NORMAL_GAMEPLAY' || agent.recoveryMode || agent.paused || agent.stopRequested
    || agent.recoveryToolRunning || agent.manualPolicy?.phase === 'switching') throw Error('Recovery not permitted at this control state');
  if ((agent.recoveryCallCount || 0) >= config.maxCallsPerEpisode) throw Error('Episode recovery call limit reached');
  if (Date.now() - (agent.lastRecoveryAt || 0) < config.cooldownMs) throw Error('Recovery cooldown active');
  agent.recoveryToolRunning = true;
  // Hold new dispatch until tool completion is accepted. Existing requests and
  // actions settle in the same session; no post-tool Brain review is needed.
  agent.recoveryAwaitingBrain = true;
  agent.recoveryCallCount = (agent.recoveryCallCount || 0) + 1;
  const callId = randomUUID(), epochBefore = agent.contextEpoch;
  let primary, observation, receipt, finalReleased = false;
  let ownsEpoch;
  const cancelled = () => agent.stopRequested || agent.paused || signal?.aborted
    || (ownsEpoch !== undefined && agent.contextEpoch !== ownsEpoch);
  try {
    await settlePolicyForRecovery(agent, 'brain_recovery_boundary_release', cancelled);
    ownsEpoch = agent.contextEpoch;
    if (cancelled()) throw Object.assign(Error('Recovery cancelled at boundary'), { name: 'AbortError' });
    if (agent.gamePaused) await agent.resumeGame('brain_mobility_recovery');
    if (cancelled()) throw Object.assign(Error('Recovery cancelled before input'), { name: 'AbortError' });
    if(agent.gamePaused)throw Error('Recovery game resume was not confirmed');
    agent.assertRecoveryControl?.(request);
    const records = agent.currentSession.timeline.records;
    const previous = records.findLast(r => r.type === 'executed_action' && r.execution?.completed !== false);
    const lastState = previous?.states?.at(-1) || previous?.actionPackage?.states?.at(-1);
    await agent.archiveEvent?.('brain_recovery_before', agent.lastObservation);
    await agent.recordHistoryEvent('brain_recovery_requested', { callId, request, config, epochBefore, controlEpoch: ownsEpoch });
    receipt = await executeRecovery({ request, config, runtime: agent.frameworkRuntime.actionRuntime,
      shouldStop: cancelled, context: { callId, attempt: agent.recoveryCallCount - 1, lastState },
      prepare: async candidate => {
        const record = agent.currentSession.addPolicyCandidate({ ...candidate, states: [candidate.appliedState] });
        await agent.recordHistoryEvent('brain_recovery_candidate', record);
      },
      interrupted: evidence => agent.recordHistoryEvent('brain_recovery_interrupted', { ...evidence, callId }),
      record: async evidence => {
        const record = { ...evidence, callId, basedOnObservationId: agent.frameworkRuntime.brainRuntime.scheduler.latestObservationId,
          states: [evidence.appliedState], execution: { durationMs: evidence.requestedDurationMs,
            durationSource: 'configured_engine_action', completed: evidence.completed } };
        agent.frameworkRuntime.executedAction(record);
        await agent.recordHistoryEvent('brain_recovery_input', record);
      } });
    if (cancelled()) throw Object.assign(Error('Recovery cancelled before continuation'), { name: 'AbortError' });
    await agent.frameworkRuntime.actionRuntime.reset('brain_recovery_final_release');
    finalReleased = true;
    // A pre-tool capture may finish late. Fence captures by acquisition start,
    // without changing the Policy context epoch or the session ID.
    agent.recoveryObservationGeneration = (agent.recoveryObservationGeneration || 0) + 1;
    observation = await agent.capture('Gameplay observation after explicit Brain recovery inputs; effect unconfirmed.');
    await agent.archiveEvent?.('brain_recovery_after', observation);
    const scheduler = agent.frameworkRuntime.brainRuntime.scheduler;
    receipt.recoveryContext = { cutoffActionSeq: agent.currentSession.timeline.actionSeq,
      basedOnObservationId: observation.frameworkObservationId || scheduler.latestObservationId,
      updateType: 'full', initial: false };
    receipt.callId = callId;
    await agent.recordHistoryEvent('brain_recovery_result', { ...receipt,
      imagePath: observation.imagePath, effectConfirmed: false });
    agent.frameworkRuntime.automaticBrainBarrierPending = false;
    return receipt;
  } catch (error) { primary = error; error.recoveryExecutionError = true; throw error; }
  finally {
    try {
      if (!finalReleased || primary || cancelled()) await agent.frameworkRuntime.actionRuntime.reset('brain_recovery_cleanup_release');
      // Successful execution continues directly to Policy; only user pause or
      // an error may re-open the menu. There is no post-tool Brain review.
      if (agent.paused && !agent.stopRequested && !agent.gamePaused) await agent.pauseGame('brain_recovery_user_pause');
      if (primary && !agent.stopRequested && !agent.gamePaused) await agent.pauseGame('brain_recovery_failed');
    } catch (error) {
      if (!primary) { error.recoveryExecutionError = true; throw error; }
      primary.cleanupErrors = [...(primary.cleanupErrors || []), { phase: 'recovery_final_release_pause', message: error.message }];
    } finally {
      agent.lastRecoveryAt = Date.now();
      agent.recoveryToolRunning = false;
      if (primary) agent.recoveryAwaitingBrain = false;
    }
  }
}
function startRecovery(agent, request, signal) {
  if (agent.recoveryPromise) return Promise.reject(Error('Recovery already in progress'));
  const promise = performRecovery(agent, request, signal);
  agent.recoveryPromise = promise;
  void promise.finally(() => { if (agent.recoveryPromise === promise) agent.recoveryPromise = null; }).catch(() => {});
  return promise;
}
module.exports = { changePolicyTarget, performRecovery, startRecovery, settlePolicyForRecovery };

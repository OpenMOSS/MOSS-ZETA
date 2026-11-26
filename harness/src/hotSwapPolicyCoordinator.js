const { PolicyActor } = require('./policyActor');
const { AsyncPolicyOcr } = require('./asyncPolicyOcr');
const { SubmissionGate } = require('../../cadence/src/fixedSubmissionClock');
const { buildPolicySystemMessage } = require('./policyContext');
const { SessionState } = require('./sessionState');

// Owns model contexts only. The active capture clock and native input thread are
// shared; a standby failure never invokes their pause/reset paths.
class HotSwapPolicyCoordinator {
  constructor(agent, options = {}) {
    this.agent = agent;
    this.prepareSession = options.prepareSession || ((config, request) => require('./llmClient').preparePolicySession(config, request));
    this.activeId = 'A'; this.ownershipVersion = 0; this.updateVersion = 0;
    this.slots = new Map(); this.jobs = new Set(); this.pending = null;
    this.retiredResponseCount = 0; this.reboundGates = new Set();
  }

  initialize() {
    const a = this.agent;
    if (!a.defaultConfig.policyStreaming || !a.executor?.streaming) throw Error('Policy hot swap requires streamed native fixed cadence');
    if (!a.defaultConfig.policyStandbyBaseUrl || a.defaultConfig.policyStandbyBaseUrl === a.defaultConfig.baseUrl)
      throw Error('Policy hot swap requires two distinct Policy services');
    this.slots.set('A', this.makeSlot('A', a.frameworkRuntime.policyActor, 0));
    this.record('policy_handoff_initialized', { activeOwner: 'A', ownershipVersion: 0 });
  }

  configFor(id) {
    const config = this.agent.defaultConfig;
    const slot = this.slots.get(id);
    return { ...config, ...(id === 'B' ? { baseUrl: config.policyStandbyBaseUrl,
      apiKey: config.policyStandbyApiKey || config.apiKey } : {}),
      policyAwaitNextPrepare: Boolean(this.pending?.slot === slot),
      policyUpdateVersion: slot?.updateVersion > 0 && slot.actor.transport.sessionId === slot.preparedContextId ? slot.updateVersion : undefined };
  }
  get activeSlot() { return this.slots.get(this.activeId); }
  activeConfig() { return this.configFor(this.activeId); }
  totalResponses() { return this.retiredResponseCount + [...this.slots.values()].reduce((n, s) => n + s.actor.responseCount, 0); }
  metadata() { return { policyOwnerId: this.activeId, policyOwnershipVersion: this.ownershipVersion,
    policyUpdateVersion: this.activeSlot?.updateVersion || 0 }; }
  getStatus() { return { ...this.metadata(), pending: this.pending ? {
    updateVersion: this.pending.version, ownerId: this.pending.slot.id, phase: this.pending.phase,
    updateType: this.pending.cycle.type, producedPerfMs: this.pending.producedPerfMs,
    warmReadyPerfMs: this.pending.warmReadyPerfMs || null } : null }; }
  record(type, event) {
    const payload = { at: new Date().toISOString(), perfMs: performance.now(), ...event };
    this.agent.deferPolicyEvidence(() => this.agent.recordHistoryEvent(type, payload));
  }
  track(promise) {
    this.jobs.add(promise); promise.finally(() => this.jobs.delete(promise)).catch(() => {}); return promise;
  }

  makeSlot(id, existing, version) {
    const a = this.agent;
    const slot = { id, updateVersion: version, messages: [], ocr: new AsyncPolicyOcr(), retired: false };
    if (existing) { slot.messages = a.messages; slot.ocr = a.policyOcr; }
    const active = () => this.activeSlot === slot;
    const tags = () => ({ policyOwnerId: id, policyUpdateVersion: slot.updateVersion,
      policyContextId: slot.actor?.transport.sessionId });
    const bindings = {
      sessionState: a.currentSession,
      getConfig: () => this.configFor(id),
      getMessages: () => active() ? a.messages : slot.messages,
      appendMessage: message => (active() ? a.messages : slot.messages).push(message),
      onDelivered: messages => slot.ocr.acknowledge(messages.map(m => m.ocrToken).filter(Boolean)),
      onSessionLost: async () => { if (active()) return a.rebuildPolicyContext({ reason: 'session_lost' }); throw Error('Standby Policy session lost'); },
      buildRequestMessages: messages => a.buildRequestMessages(messages),
      createCandidate: candidate => a.currentSession.addPolicyCandidate({ ...candidate, ...tags() }),
      recordCandidate: candidate => a.recordPolicyCandidate(candidate),
      recordModelResponse: record => a.recordModelResponse({ ...record, ...tags() }),
      recordEvent: (type, record) => a.recordHistoryEvent(type, { ...record, ...tags() }),
      buildRetryText: () => a.buildInvalidPackageRetryText(),
      requestBrainReplan: () => false,
      emit: (type, payload) => a.emit(type, { ...payload, ...tags() }),
      shouldStop: () => a.stopRequested || slot.retired,
      model: a.currentModel, chatCompletion: a.chatCompletion,
      maxRetries: a.modelRequestMaxRetries, retryBaseDelayMs: a.modelRequestRetryBaseDelayMs
    };
    slot.actor = existing || new PolicyActor(bindings);
    if (existing) Object.assign(existing, bindings);
    return slot;
  }

  stage(cycle, scheduler) {
    if (this.pending?.cycle === cycle) return;
    this.cancel('newer_update');
    const a = this.agent;
    const pendingBrain = scheduler.pending;
    if (!pendingBrain || !['small', 'fallback'].includes(cycle.type)) throw Error('Only buffered small/fallback updates may be staged');
    const id = this.activeId === 'A' ? 'B' : 'A';
    const previous = this.slots.get(id);
    if (previous) { this.retiredResponseCount += previous.actor.responseCount; previous.retired = true; previous.actor.stop('standby_replaced'); }
    const slot = this.makeSlot(id, null, ++this.updateVersion);
    const revision = { ...pendingBrain, revision: scheduler.revisionCount + 1,
      actualPublishActionSeq: a.currentSession.timeline.actionSeq,
      policyTriggerThinking: pendingBrain.policyTriggerThinking || cycle.thinking || '' };
    const shadow = Object.create(a.currentSession);
    shadow.brain = structuredClone(a.currentSession.brain);
    shadow.currentPolicyInstruction = structuredClone(a.currentSession.currentPolicyInstruction);
    SessionState.prototype.commitBrainRevision.call(shadow, revision);
    slot.shadow = shadow;
    slot.actor.sessionState = shadow;
    const host = Object.create(a); host.currentSession = shadow;
    slot.systemMessage = buildPolicySystemMessage(host);
    slot.messages = [slot.systemMessage];
    // Independent lag-one delivery state, with only already-recognized evidence.
    slot.ocr.previousRequest = a.policyOcr.previousRequest;
    this.copyOcr(slot);
    this.slots.set(id, slot);
    const pending = this.pending = { version: slot.updateVersion, slot, cycle, scheduler,
      pendingBrain, expectedOwner: this.activeId, expectedOwnershipVersion: this.ownershipVersion,
      epoch: a.contextEpoch, phase: 'warming', producedPerfMs: cycle.producedPerfMs || performance.now(),
      controller: new AbortController(), startedPerfMs: performance.now() };
    this.record('policy_handoff_update_staged', { ...this.pendingTags(pending),
      updateType: cycle.type, brainResponseReadyAt: pendingBrain.responseReadyAt || null, instructionUnchanged: shadow.currentPolicyInstruction.text === a.currentSession.currentPolicyInstruction.text,
      oldHistoryImages: 0, memory: shadow.brain.memory, think: shadow.brain.think });
    const timeoutMs = a.defaultConfig.policyHandoffPrepareTimeoutMs || 15000;
    pending.timer = setTimeout(() => this.failPending(pending, Error('Standby preparation timed out')), timeoutMs);
    this.track((async () => {
      if (previous) {
        const released = await previous.actor.transport.waitForRelease();
        if (released.pending.length) throw Error('Standby previous session release was not acknowledged');
      }
      if (!this.isCurrent(pending)) return;
      this.record('policy_handoff_warm_started', this.pendingTags(pending));
      const match = slot.systemMessage.content.match(/\n\n当前有效目标：([\s\S]*?)\n\nMemory：([\s\S]*?)\n\n模型思考：([\s\S]*)$/);
      if (!match) throw Error('Standby system lacks canonical state');
      const state = { instruction: match[1], memory: match[2], model_thought: match[3] };
      // /prepare creates a remote session even before the first generation.
      // Mark it owned before sending so timeout/cancel also retires that session.
      slot.preparedContextId = slot.actor.transport.sessionId;
      slot.actor.transport.config = { ...this.configFor(id), modelName: a.currentModel };
      slot.actor.transport.sent = true;
      const ready = await this.prepareSession(this.configFor(id), { sessionId: slot.actor.transport.sessionId,
        state, updateVersion: pending.version, signal: pending.controller.signal, timeoutMs });
      if (!this.isCurrent(pending)) return;
      if (!ready?.ready || ready.history_turns !== 0 || ready.update_version !== pending.version
        || ![0, 1].every(route => ready.routes?.some(r => r.route === route && r.prepared)))
        throw Error('Standby did not acknowledge two prepared routes and an empty versioned history');
      pending.warmReadyPerfMs = performance.now(); pending.phase = 'waiting_fresh_observation';
      this.record('policy_handoff_warm_ready', { ...this.pendingTags(pending), receipt: ready });
    })().catch(error => this.failPending(pending, error)));
  }

  pendingTags(p) { return { updateVersion: p.version, fromOwner: p.expectedOwner, toOwner: p.slot.id,
    ownershipVersion: p.expectedOwnershipVersion + 1, contextId: p.slot.actor.transport.sessionId }; }
  isCurrent(p) { const a = this.agent; return this.pending === p && !p.cancelRequested && !p.controller.signal.aborted && !a.stopRequested
    && !a.paused && !a.taskTransition && !a.finalizing && a.contextEpoch === p.epoch
    && this.activeId === p.expectedOwner && this.ownershipVersion === p.expectedOwnershipVersion
    && a.brainCycle === p.cycle && p.scheduler.pending === p.pendingBrain; }
  copyOcr(slot) {
    const active = this.agent.policyOcr;
    const merged = [...slot.ocr.results, ...active.results, ...(active.latest ? [active.latest] : [])];
    slot.ocr.results = [...new Map(merged.filter(r => !slot.ocr.closed.has(r.observationId))
      .map(r => [r.observationId, r])).values()].slice(-8);
    slot.ocr.latest = active.latest;
  }

  offerObservation(observation) {
    const p = this.pending;
    if (!p || p.phase !== 'waiting_fresh_observation' || p.inference || !this.isCurrent(p)) return;
    if (!Number.isFinite(observation.captureScheduledPerfMs)
      || observation.captureScheduledPerfMs <= p.warmReadyPerfMs) return;
    const sourceGate = observation.submissionGate;
    if (!sourceGate || performance.now() >= sourceGate.expiresAt) return;
    const a = this.agent, slot = p.slot;
    const release = a.captureStore.pin(observation.imagePath);
    const fresh = { ...observation };
    // Independent gate: neither service can consume the other's send receipt.
    fresh.submissionGate = new SubmissionGate(sourceGate.slot, sourceGate.submitAt, sourceGate.executeAt,
      true, sourceGate.maxLatenessMs);
    const timer = setTimeout(() => fresh.submissionGate.release(), Math.max(0, sourceGate.submitAt - performance.now()));
    this.copyOcr(slot);
    const ocr = slot.ocr.reserve(fresh);
    slot.messages = [slot.systemMessage, { role: 'user', ocrToken: ocr.token, ocrRecords: ocr.records, content: [
      { type: 'text', text: ocr.text + '\n' }, { type: 'local_image', imageKind: 'full_frame',
        observationId: fresh.frameworkObservationId, imagePath: fresh.imagePath, mimeType: fresh.mimeType,
        capturedAt: fresh.capturedAt, width: fresh.width, height: fresh.height }] }];
    let firstExecutablePerfMs = null;
    fresh.onPolicyStream = event => {
      if (event.type === 'state' && firstExecutablePerfMs === null) {
        firstExecutablePerfMs = event.receivedPerfMs || performance.now();
        this.record('policy_handoff_first_executable', { ...this.pendingTags(p), requestId: event.requestId,
          observationId: fresh.frameworkObservationId, firstExecutablePerfMs });
      }
    };
    p.phase = 'inferencing';
    this.record('policy_handoff_candidate_requested', { ...this.pendingTags(p), observationId: fresh.frameworkObservationId,
      captureScheduledPerfMs: fresh.captureScheduledPerfMs, scheduledStartPerfMs: fresh.scheduledStartPerfMs });
    p.inference = this.track((async () => {
      const result = await a.frameworkRuntime.inferPolicy(fresh, slot.actor);
      if (result?.recordEvidence) a.deferPolicyEvidence(result.recordEvidence);
      a.responseCount = this.totalResponses();
      if (!this.isCurrent(p)) return;
      if (!result || result.instructionUpdateRequested || !result.actionPackage?.states
        || result.actionPackage.states.length !== 6) {
        this.record('policy_handoff_candidate_rejected', { ...this.pendingTags(p), reason: 'no_complete_action_or_call_brain' });
        return;
      }
      // Backend must finish warming the next alternating lane before takeover.
      const wire = result.response?.raw || {};
      const readiness = wire.next_route_preparation;
      if (wire.lumine_update_version !== p.version || !readiness?.ready || readiness.awaited !== true
        || readiness.update_version !== p.version || ![0, 1].includes(wire.route)
        || readiness.route !== 1 - wire.route || !Number.isInteger(wire.turn)
        || readiness.history_turns !== wire.turn || !/^[a-f0-9]{64}$/.test(readiness.prefix_sha256 || '')
        || readiness.prefix_sha256 !== wire.dialogue_audit?.next_prefix_sha256)
        throw Error('Standby response lacks matching version and next-route prefix readiness');
      const now = performance.now();
      const maxAge = a.defaultConfig.policyHandoffMaxObservationAgeMs || 400;
      if (now - fresh.captureScheduledPerfMs > maxAge || now >= fresh.scheduledStartPerfMs + 200) {
        this.record('policy_handoff_candidate_rejected', { ...this.pendingTags(p), reason: 'stale_observation',
          ageMs: now - fresh.captureScheduledPerfMs });
        return;
      }
      const signature = JSON.stringify([slot.shadow.brain.revision, slot.shadow.brain.think || '',
        slot.shadow.currentPolicyInstruction.text || a.taskInstruction, a.taskState, a.recoveryMode]);
      const pkg = { ...result.actionPackage, strictDeadline: true, carryEnabled: a.packageCarryAllowed(),
        contextEpoch: a.contextEpoch + 1, contextSignature: signature,
        policyOwnerId: slot.id, policyOwnershipVersion: this.ownershipVersion + 1, policyUpdateVersion: p.version,
        candidateId: result.candidate.candidateId, basedOnObservationId: result.candidate.basedOnObservationId,
        observationCapturedAt: fresh.capturedAt, captureTick: fresh.captureTick,
        firstExecutablePerfMs, scheduledStartPerfMs: fresh.scheduledStartPerfMs,
        captureReadyPerfMs: fresh.captureReadyPerfMs, captureSlotReadyPerfMs: fresh.captureSlotReadyPerfMs,
        captureTiming: fresh.captureTiming, clientTiming: result.response.clientTiming,
        vllmTiming: result.response.vllmTiming, requestMetrics: result.response.requestMetrics,
        firstExecutablePackageMs: result.response.firstExecutablePackageMs, responseCount: a.responseCount };
      p.phase = 'handoff';
      this.record('policy_handoff_requested', { ...this.pendingTags(p), requestId: pkg.requestId,
        observationId: fresh.frameworkObservationId, observationAgeMs: now - fresh.captureScheduledPerfMs,
        firstExecutablePerfMs, completeActionPerfMs: now, readiness });
      let receipt;
      try { receipt = await a.executor.handoff({ expectedOwner: this.activeId, nextOwner: slot.id,
        updateVersion: p.version, ownershipVersion: this.ownershipVersion + 1,
        observationPerfMs: fresh.captureScheduledPerfMs, maxObservationAgeMs: maxAge,
        firstExecutablePerfMs, pkg, isCurrent: () => this.isCurrent(p),
        commit: nativeReceipt => this.commit(p, result, fresh, signature, nativeReceipt) });
      } catch (error) { error.executionHandoffFatal = true; throw error; }
      if (!receipt?.accepted) this.record('policy_handoff_rejected', { ...this.pendingTags(p), receipt });
    })().catch(error => {
      if (error.executionHandoffFatal) {
        this.record('policy_handoff_runtime_fault', { ...this.pendingTags(p), error: error.message, activeContinues: false });
        a.fail(error);
      } else if (error.nonFatalSlotSkip || error.policyDependencyDropped) {
        this.record('policy_handoff_candidate_rejected', { ...this.pendingTags(p), reason: error.message });
      } else this.failPending(p, error);
    }).finally(() => {
      clearTimeout(timer); fresh.submissionGate.cancel('standby_request_finished'); release();
      p.inference = null;
      if (this.pending === p && p.cancelRequested) {
        const failure = p.deferredFailure;
        if (failure) this.failPending(p, failure, true);
        else this.cancel(p.cancelRequested, true);
      }
      if (this.isCurrent(p)) p.phase = 'waiting_fresh_observation';
    }));
  }

  commit(p, result, observation, signature, receipt) {
    // Called synchronously after the helper accepted the same ownership version.
    // No awaits and no controller reset are allowed inside this transaction.
    const a = this.agent, old = this.activeSlot;
    if (a.stopRequested) {
      this.record('policy_handoff_stopped_after_accept', { ...this.pendingTags(p), receipt });
      return;
    }
    old.messages = a.messages; old.ocr = a.policyOcr;
    p.scheduler.pending.policyTriggerThinking = p.cycle.thinking || p.scheduler.pending.policyTriggerThinking || '';
    p.scheduler.deferCommit = false;
    if (!p.scheduler.tryRelease()) throw Error('Accepted handoff lost its buffered Brain revision');
    a.brainCycle = null;
    this.activeId = p.slot.id; this.ownershipVersion += 1;
    a.frameworkRuntime.policyActor = p.slot.actor;
    p.slot.actor.sessionState = a.currentSession;
    a.messages = p.slot.messages; a.policyOcr = p.slot.ocr;
    const oldEpoch = a.contextEpoch;
    a.contextEpoch += 1; a.policyContextSignature = signature;
    this.lastHandoff = { fromEpoch: oldEpoch, toEpoch: a.contextEpoch,
      candidateCapturePerfMs: observation.captureScheduledPerfMs };
    a.discardPendingCadenceObservation();
    // Retired requests are allowed to finish only as audit evidence. Their
    // continuations capture the old epoch and cannot append to the new context.
    old.retired = true; old.actor.stop('policy_handoff_retired');
    const oldPrediction = a.inFlightPrediction;
    if (oldPrediction) this.track(Promise.resolve(oldPrediction).catch(() => {}));
    a.inFlightPrediction = null;
    clearTimeout(p.timer); this.pending = null;
    a.responseCount = this.totalResponses();
    this.record('policy_handoff_committed', { ...this.pendingTags(p), receipt,
      producedPerfMs: p.producedPerfMs, warmReadyPerfMs: p.warmReadyPerfMs,
      requestId: result.actionPackage.requestId, observationId: observation.frameworkObservationId,
      contextEpoch: a.contextEpoch, activeInstruction: a.currentSession.currentPolicyInstruction.text,
      activeMemory: a.currentSession.brain.memory, activeThink: a.currentSession.brain.think });
    const trigger = a.policyBrainTrigger(result, observation, { afterCandidate: result.candidate.candidateId });
    if (trigger) a.queuePolicyBrainTrigger(trigger);
    // A next-grid screenshot can finish while B's preceding response is being
    // validated. Rebind that still-fresh image to B instead of losing a beat.
    const next = this.rebindObservation(a.latestCadenceObservation);
    if (next) a.offerCadenceObservation(next);
  }

  rebindObservation(observation) {
    const a = this.agent, transition = this.lastHandoff, sourceGate = observation?.submissionGate;
    if (!observation || !transition || a.contextEpoch !== transition.toEpoch
      || observation.captureContextEpoch !== transition.fromEpoch || !sourceGate
      || observation.captureScheduledPerfMs <= transition.candidateCapturePerfMs
      || a.stopRequested || a.paused || a.taskTransition || a.finalizing) return null;
    const now = performance.now();
    if (now >= sourceGate.expiresAt || now - observation.captureScheduledPerfMs > (a.defaultConfig.policyHandoffMaxObservationAgeMs || 400)) return null;
    const rebound = { ...observation, captureSourceContextEpoch: observation.captureContextEpoch,
      captureContextEpoch: a.contextEpoch, captureInvalidated: false };
    const gate = new SubmissionGate(sourceGate.slot, sourceGate.submitAt, sourceGate.executeAt,
      true, sourceGate.maxLatenessMs, finished => {
        this.reboundGates.delete(gate);
        this.record('policy_handoff_rebound_submission', { ...this.metadata(), observationId: observation.frameworkObservationId,
          slot: finished.slot, sent: finished.dispatched, reason: finished.reason || null });
      });
    Object.defineProperty(rebound, 'submissionGate', { enumerable: false, value: gate });
    this.reboundGates.add(gate);
    setTimeout(() => gate.release(), Math.max(0, gate.submitAt - now));
    this.record('policy_handoff_capture_rebound', { ...this.metadata(), observationId: observation.frameworkObservationId,
      sourceContextEpoch: transition.fromEpoch, contextEpoch: transition.toEpoch,
      captureScheduledPerfMs: observation.captureScheduledPerfMs, scheduledSubmitPerfMs: gate.submitAt,
      scheduledActionPerfMs: gate.executeAt });
    return rebound;
  }

  waitForHandoff() {
    return this.pending?.phase === 'handoff' ? this.pending.inference : Promise.resolve();
  }

  failPending(p, error, force = false) {
    if (this.pending !== p) return;
    if (p.phase === 'handoff' && !force) {
      p.deferredFailure = error; this.cancel('standby_failed'); return;
    }
    this.record('policy_handoff_failed', { ...this.pendingTags(p), phase: p.phase,
      error: error.message || String(error), activeContinues: true });
    this.cancel('standby_failed', true);
    if (this.agent.brainCycle === p.cycle) {
      this.agent.frameworkRuntime.invalidateBrainContext('standby_update_failed');
      p.scheduler.deferCommit = false;
      this.agent.brainCycle = null;
    }
  }
  cancel(reason, force = false) {
    const p = this.pending; if (!p) return;
    if (p.phase === 'handoff' && !force) {
      p.cancelRequested = reason;
      this.record('policy_handoff_cancellation_pending', { ...this.pendingTags(p), reason });
      return;
    }
    this.pending = null; clearTimeout(p.timer); p.controller.abort(Error(reason));
    p.slot.retired = true; p.slot.actor.stop(reason);
    this.record('policy_handoff_cancelled', { ...this.pendingTags(p), reason, activeOwner: this.activeId });
  }
  async stop(reason) {
    this.cancel(reason);
    for (const gate of this.reboundGates) gate.cancel(reason);
    this.reboundGates.clear();
    for (const slot of this.slots.values()) { slot.retired = true; slot.actor.stop(reason); }
    await Promise.allSettled([...this.jobs]);
    await Promise.allSettled([...this.slots.values()].map(s => s.actor.transport.waitForRelease()));
  }
}
module.exports = { HotSwapPolicyCoordinator };

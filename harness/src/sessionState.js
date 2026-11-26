const fs = require("node:fs/promises");
const path = require("node:path");
const { emptyMemory, validateBrainOutput, validateMemory } = require("./brainSchema");

function clone(value) {
  return value === undefined ? null : JSON.parse(JSON.stringify(value));
}

function nowIso() {
  return new Date().toISOString();
}

class SessionTimeline {
  enableArchive(file,limit=128) {this.archiveFile=file;this.recordLimit=limit;require('node:fs').mkdirSync(path.dirname(file),{recursive:true});}
  trim() {if(!this.archiveFile||this.records.length<=this.recordLimit)return;const rows=this.records.slice(0,-this.recordLimit);require('node:fs').appendFileSync(this.archiveFile,rows.map(r=>JSON.stringify(r)).join('\n')+'\n');this.records.splice(0,rows.length);}
  readRecords({fromActionSeq=-Infinity,toActionSeq=Infinity}={}) {
    const keep=r=>Number(r.actionSeq??r.actionCount??0)>=fromActionSeq&&Number(r.actionSeq??r.actionCount??0)<=toActionSeq;
    if(!this.archiveFile)return this.records.filter(keep);
    const disk=require('node:fs'),rows=[];let fd;
    try{fd=disk.openSync(this.archiveFile,'r');}catch(e){if(e.code!=='ENOENT')throw e;}
    if(fd!==undefined){const decoder=new (require('node:string_decoder').StringDecoder)('utf8'),buffer=Buffer.alloc(65536);let pending='';try{let n;while((n=disk.readSync(fd,buffer,0,buffer.length,null))>0){pending+=decoder.write(buffer.subarray(0,n));let end;while((end=pending.indexOf('\n'))>=0){const line=pending.slice(0,end);pending=pending.slice(end+1);if(line){const row=JSON.parse(line);if(keep(row))rows.push(row);}}}pending+=decoder.end();if(pending.trim()){const row=JSON.parse(pending);if(keep(row))rows.push(row);}}finally{disk.closeSync(fd);}}
    return rows.concat(this.records.filter(keep));
  }

  constructor() {
    this.records = [];
    this.historyRecordCursor = 0;
    this.historyDirtyRecords = new Set();
    this.historyRecordIndexes = new WeakMap();
    this.observationSeq = 0;
    this.policyCandidateSeq = 0;
    this.actionSeq = 0;
  }

  addObservation(observation = {}) {
    this.observationSeq += 1;
    const record = {
      type: "observation",
      observationId: `O${this.observationSeq}`,
      observationSeq: this.observationSeq,
      at: observation.capturedAt || nowIso(),
      ...clone(observation),
      actionSeq: Number(observation.actionSeq ?? observation.actionCount ?? this.actionSeq)
    };
    this.historyRecordIndexes.set(record, this.records.length);
    this.records.push(record);
    this.trim();
    return record;
  }

  addPolicyCandidate(candidate = {}) {
    this.policyCandidateSeq += 1;
    const record = {
      type: "policy_action_candidate",
      candidateId: `P${this.policyCandidateSeq}`,
      candidateSeq: this.policyCandidateSeq,
      at: nowIso(),
      ...clone(candidate)
    };
    this.historyRecordIndexes.set(record, this.records.length);
    this.records.push(record);
    this.trim();
    return record;
  }

  addExecutedAction(action = {}) {
    this.actionSeq += 1;
    const record = {
      type: "executed_action",
      actionId: `A${this.actionSeq}`,
      actionSeq: this.actionSeq,
      at: nowIso(),
      ...clone(action)
    };
    this.historyRecordIndexes.set(record, this.records.length);
    this.records.push(record);
    this.trim();
    return record;
  }

  touch(record) {
    if (record) this.historyDirtyRecords.add(record);
  }

  historyChanges() {
    if (this.archiveFile) throw new Error("Incremental history requires the complete in-memory Brain timeline.");
    const changes = [];
    const keys = ["framework", "timeline"];
    for (const key of ["observationSeq", "policyCandidateSeq", "actionSeq"]) {
      changes.push({ path: [...keys, key], value: this[key] });
    }
    const added = this.records.slice(this.historyRecordCursor);
    if (added.length) changes.push({ path: [...keys, "records"], append: added });
    for (const record of this.historyDirtyRecords) {
      const index = this.historyRecordIndexes.get(record);
      if (index !== undefined && index < this.historyRecordCursor) {
        changes.push({ path: [...keys, "records", String(index)], value: record });
      }
    }
    this.historyRecordCursor = this.records.length;
    this.historyDirtyRecords.clear();
    return changes;
  }

  snapshot() {
    return {
      observationSeq: this.observationSeq,
      policyCandidateSeq: this.policyCandidateSeq,
      actionSeq: this.actionSeq,
      records: clone(this.records)
    };
  }
}

class SessionState {
  constructor(options = {}) {
    if (!options.sessionId) {
      throw new Error("SessionState requires sessionId.");
    }
    this.sessionId = options.sessionId;
    this.agentMode = options.agentMode;
    this.startedAt = options.startedAt || nowIso();
    this.userGoal = String(options.userGoal || "").trim();
    this.timeline = new SessionTimeline();
    this.brain = {
      plan: "",
      phaseBoundary: "",
      memory: emptyMemory(),
      think: "",
      revision: 0,
      scheduler: null,
      lastBrainCommitActionSeq: 0,
      lastMemoryUpdateActionSeq: 0,
      lastInstructionUpdateActionSeq: 0
    };
    this.currentPolicyInstruction = this.userGoal
      ? {
          instructionId: "instruction-0",
          version: 0,
          basedOnObservationId: null,
          text: this.userGoal
        }
      : null;
    this.pendingBrainRevision = null;
    this.pauseHarness = { phase: "IDLE", gamePaused: false };
    this.taskState = String(options.taskState || "NORMAL_GAMEPLAY");
    this.recoveryMode = false;
    this.sceneContext = {};
    this.continuation = options.continuation ? clone(options.continuation) : null;
    if (this.continuation) {
      this.loadContinuation(this.continuation);
    }
  }

  loadContinuation(handoff) {
    if (handoff.schemaVersion !== 2) throw new Error("Continuation requires schemaVersion=2");
    const memory = require("./brainProtocol").migrateMemory(handoff.memory);
    if (handoff.currentGoal) this.userGoal = String(handoff.currentGoal);
    if (handoff.currentPolicyInstruction) {
      this.currentPolicyInstruction = clone(handoff.currentPolicyInstruction);
    }
    if (handoff.sceneContext) this.sceneContext = clone(handoff.sceneContext);
    this.brain.memory = memory;
    this.brain.think = String(handoff.think || handoff.model_thought || "");
    this.brain.plan = String(handoff.plan || "");
    this.brain.phaseBoundary = String(handoff.phaseBoundary || "");
  }

  setPauseState(phase, gamePaused) {
    this.pauseHarness = {
      phase: String(phase || "IDLE"),
      gamePaused: Boolean(gamePaused)
    };
  }

  setTaskState(state, recoveryMode = this.recoveryMode) {
    this.taskState = String(state || "UNKNOWN");
    this.recoveryMode = Boolean(recoveryMode);
  }

  invalidatePolicyInstruction(text = "Recovery required: inspect the current screen before choosing the next action.") {
    this.currentPolicyInstruction = {
      instructionId: `instruction-recovery-${this.timeline.observationSeq}`,
      version: this.brain.revision,
      basedOnObservationId: this.timeline.observationSeq
        ? `O${this.timeline.observationSeq}`
        : null,
      text: String(text)
    };
  }

  addObservation(observation) {
    return this.timeline.addObservation(observation);
  }

  addPolicyCandidate(candidate) {
    return this.timeline.addPolicyCandidate(candidate);
  }

  addExecutedAction(action) {
    return this.timeline.addExecutedAction(action);
  }

  setPendingBrainRevision(revision) {
    this.pendingBrainRevision = clone(revision);
  }

  commitBrainRevision(revision) {
    const raw = revision?.result || {};
    const type = revision.updateType || 'full';
    const scope = raw.brainFallbackScope || 'memory_think';
    const keys = require('./brainProtocol').outputKeys(type, scope);
    const result = validateBrainOutput(Object.fromEntries(keys.map(k => [k, raw[k]])), this.agentMode, type, scope);
    const fullUpdate = type === 'full' || (type === 'fallback' && scope === 'full');
    const previous = this.brain;
    const instruction = fullUpdate ? result.instruction : this.currentPolicyInstruction?.text || this.userGoal;
    const actionSeq = Number(revision.actualPublishActionSeq ?? this.timeline.actionSeq);
    this.brain = { ...previous, memory: result.memory,
      think: type === 'small' ? (revision.policyTriggerThinking || previous.think) : result.think,
      revision: Number(revision.revision || previous.revision + 1),
      scheduler: { cutoffActionSeq: revision.cutoffActionSeq, basedOnObservationId: revision.basedOnObservationId },
      lastBrainCommitActionSeq: Number(revision.cutoffActionSeq ?? actionSeq),
      // A successfully refreshed but unchanged value still fulfils the fallback obligation.
      lastMemoryUpdateActionSeq: actionSeq,
      lastInstructionUpdateActionSeq: fullUpdate ? actionSeq : previous.lastInstructionUpdateActionSeq,
      changeFlags: { memoryChanged: previous.memory !== result.memory, instructionChanged: instruction !== this.currentPolicyInstruction?.text }
    };
    if (fullUpdate) this.currentPolicyInstruction = { instructionId: `instruction-${this.brain.revision}`, version: this.brain.revision,
      basedOnObservationId: revision.basedOnObservationId, text: instruction };
    this.pendingBrainRevision = null;
    return this.brain;
  }

  brainTrajectory(cutoffActionSeq, limit = 15) {
    return this.brainObservationWindow(cutoffActionSeq, limit);
  }

  brainObservationWindow(cutoffActionSeq, limit = 8) {
    const max = Math.max(1, Math.trunc(Number(limit) || 8));
    const startActionSeq = Number(this.brain.lastBrainCommitActionSeq || 0);
    const observations = this.timeline.readRecords({fromActionSeq:startActionSeq,toActionSeq:Number(cutoffActionSeq)})
      .filter((record) => record.type === "observation" && !record.captureInvalidated && record.brainEligible !== false)
      .filter((record) => {
        const actionSeq = Number(record.actionSeq ?? record.actionCount ?? 0);
        return actionSeq >= startActionSeq && actionSeq <= Number(cutoffActionSeq);
      });
    if (observations.length <= max) return clone(observations);
    if (max === 1) return clone(observations.slice(-1));

    const selected = new Set();
    const add = (record) => { if (record) selected.add(record); };
    add(observations[0]);
    if (max > 1) add(observations.at(-1));
    // File names, OCR timings and confidence jitter are not evidence changes.
    const ocrText = (record) => (record.ocr?.results || [])
      .map((result) => ({ roi: result.roi || "", text: String(result.text || "").trim() }))
      .filter((result) => result.text)
      .sort((a, b) => a.roi.localeCompare(b.roi) || a.text.localeCompare(b.text));
    const signature = (record) => JSON.stringify({
      ocr: ocrText(record), taskState: record.taskState || null
    });
    const objectiveSignature = (record) => JSON.stringify(ocrText(record)
      .filter((result) => result.roi === "mission_objective"));
    const objectives = observations.filter((record, index) =>
      objectiveSignature(record) !== "[]" &&
      (!index || objectiveSignature(record) !== objectiveSignature(observations[index - 1])));
    for (const record of objectives.reverse()) {
      if (selected.size >= max) break;
      add(record);
    }
    const changed = [];
    for (let index = 1; index < observations.length; index += 1) {
      if (signature(observations[index]) !== signature(observations[index - 1])) {
        changed.push(observations[index]);
      }
    }
    for (const record of changed.reverse()) {
      if (selected.size >= max) break;
      add(record);
    }
    const remaining = observations.filter((record) => !selected.has(record));
    const slots = max - selected.size;
    for (let index = 0; index < slots && remaining.length; index += 1) {
      const position = Math.floor(((index + 0.5) * remaining.length) / slots);
      add(remaining[Math.min(position, remaining.length - 1)]);
    }
    return clone(observations.filter((record) => selected.has(record)));
  }

  createContinuationHandoff() {
    const observations = this.timeline.readRecords()
      .filter((record) => record.type === "observation" && !record.captureInvalidated)
      .slice(-3)
      .map((record) => ({
        observationId: record.observationId,
        imagePath: record.imagePath || null,
        capturedAt: record.capturedAt || record.at,
        ocr: record.ocr || null
      }));
    return {
      schemaVersion: 2,
      createdAt: nowIso(),
      previousSessionId: this.sessionId,
      currentGoal: this.userGoal,
      currentPolicyInstruction: clone(this.currentPolicyInstruction),
      sceneContext: clone(this.sceneContext),
      importantObservations: observations,
      plan: this.brain.plan,
      phaseBoundary: this.brain.phaseBoundary,
      memory: clone(this.brain.memory), think: this.brain.think || ""
    };
  }

  historySnapshot() {
    // Copy only bounded state plus newly appended/explicitly changed records.
    // A normal snapshot remains available to callers that explicitly need all
    // evidence; this incremental interface is only for ordered persistence.
    const state = this.snapshot({ includeTimeline: false, timelineRecordLimit: 0 });
    delete state.timeline;
    const changes = Object.entries(state).map(([key, value]) => ({ path: ["framework", key], value }));
    changes.push(...this.timeline.historyChanges());
    return { __historyPatchChanges: changes };
  }

  snapshot({ includeTimeline = true, timelineRecordLimit = 32 } = {}) {
    return {
      schemaVersion: 2,
      sessionId: this.sessionId,
      startedAt: this.startedAt,
      userGoal: this.userGoal,
      brain: clone(this.brain),
      currentPolicyInstruction: clone(this.currentPolicyInstruction),
      pendingBrainRevision: clone(this.pendingBrainRevision),
      pauseHarness: clone(this.pauseHarness),
      taskState: this.taskState,
      recoveryMode: this.recoveryMode,
      sceneContext: clone(this.sceneContext),
      timeline: includeTimeline ? this.timeline.snapshot() : {
        observationSeq: this.timeline.observationSeq, policyCandidateSeq: this.timeline.policyCandidateSeq,
        actionSeq: this.timeline.actionSeq, recordCount: this.timeline.records.length,
        records: timelineRecordLimit > 0 ? clone(this.timeline.records.slice(-timelineRecordLimit)) : [],
        recordsTruncated: this.timeline.records.length > timelineRecordLimit
      },
      continuation: clone(this.continuation)
    };
  }
}

async function writeContinuationHandoff(sessionHistoryPath, handoff) {
  if (!sessionHistoryPath) return null;
  const filePath = path.join(path.dirname(sessionHistoryPath), "session-handoff.json");
  await fs.writeFile(filePath, `${JSON.stringify(handoff, null, 2)}\n`, "utf8");
  return filePath;
}

async function readContinuationHandoff(filePath) {
  if (!filePath) return null;
  return JSON.parse(await fs.readFile(path.resolve(filePath), "utf8"));
}

module.exports = {
  SessionState,
  SessionTimeline,
  readContinuationHandoff,
  writeContinuationHandoff
};

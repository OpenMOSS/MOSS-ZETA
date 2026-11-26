// Shared terminal contract. Never derive actual execution from planned duration.
const clone=x=>structuredClone(x);
function outcome(event,statusHint) {
 const pkg=event.package,helper=event.timing?.realtimeHelperCompletion;
 const host=(helper?{events:helper.events,durationMs:helper.actualDurationMs,completed:helper.event==='gamepad_package_completed',chunksExecuted:helper.events?.length,runtime:{carryActive:event.releaseConfirmed!==true}}:null);
 let status=statusHint==='cancelled'?'cancelled':'unknown',segments=[],release={confirmed:false,source:'unconfirmed'};
 if(host){
  release={confirmed:host.runtime?.carryActive===false&&!host.runtime?.packageEndHold&&!host.runtime?.packageEndHoldReleasing&&!host.runtime?.packageEndHoldError,source:'local_runtime_reset'};
  const rows=host.events;
  if(Array.isArray(rows)&&Number.isFinite(host.durationMs)) {
   segments=rows.map((e,i)=>({startOffsetMs:e.actualMs,endOffsetMs:rows[i+1]?.actualMs??host.durationMs,state:e.state,clockDomain:'windows_helper_monotonic_relative'}));
   const valid=segments.every(s=>validSegment(s));
   if(!valid)segments=[];
   status=valid&&segments.length?(host.completed===true&&host.chunksExecuted===pkg.states?.length&&rows.length===pkg.states?.length?'completed':'partial'):'unknown';
  }
 }else if(event.timing?.executionMode==='node_chunk_loop') {
  segments=(event.timing.chunks||[]).map(c=>({startOffsetMs:c.applyStartedOffsetMs+c.applyMs,endOffsetMs:c.startedAtOffsetMs+c.actualDurationMs,state:c.appliedState||pkg.states[c.index],clockDomain:'workstation_monotonic_relative'})).filter(validSegment);
  status=segments.length?(statusHint==='completed'&&segments.length===pkg.states.length?'completed':'partial'):'unknown';
  release={confirmed:event.releaseConfirmed===true,source:'local_runtime_reset'};
 }
 // Timed release can change one control inside an otherwise unchanged chunk.
 // Keep chunk receipts/raw requests, but split actual evidence at device write
 // acknowledgements rather than claiming the initial state lasted all chunk.
 const transitions=host?.controllerTransitions??event.timing?.controllerTransitions;
 const transitionEnd=host?.durationMs??event.timing?.actualDurationMs;
 if(Array.isArray(transitions)&&Number.isFinite(transitionEnd)) {
  const rows=transitions.filter(t=>Number.isFinite(t.completedOffsetMs)&&t.completedOffsetMs>=0&&t.completedOffsetMs<=transitionEnd);
  segments=rows.map((t,i)=>({startOffsetMs:t.completedOffsetMs,endOffsetMs:rows[i+1]?.completedOffsetMs??transitionEnd,state:t.state,
   clockDomain:host?'windows_helper_monotonic_relative':'workstation_monotonic_relative',timeBasis:'device_write_acknowledgement'}));
  if(!segments.every(validSegment)){segments=[];status='unknown';}
 }
 const endMs=Date.parse(event.completedAt||new Date().toISOString());
 return {package:pkg,status,reason:event.reason||null,confirmationId:'executor:'+pkg.candidateId,
  startMs:event.startedAt?Date.parse(event.startedAt):endMs,endMs,
  actualSegments:clone(segments),release,timeBasis:'workstation request/receipt bounds; segments use their stated monotonic domain'};
}
function validSegment(s){const st=s.state;return Number.isFinite(s.startOffsetMs)&&s.startOffsetMs>=0&&Number.isFinite(s.endOffsetMs)&&s.endOffsetMs>=s.startOffsetMs&&st&&['leftStick','rightStick'].every(k=>['x','y'].every(a=>Number.isFinite(st[k]?.[a])&&Math.abs(st[k][a])<=1))&&[0,1].includes(st.leftTrigger)&&[0,1].includes(st.rightTrigger)&&Array.isArray(st.buttons);}
module.exports={outcome,validSegment};

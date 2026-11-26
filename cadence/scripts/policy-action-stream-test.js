const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const { PolicyActionStream, fetchPolicyStream } = require('../src/policyActionStream');

test('fragmented tokens: only a terminated complete state can execute', () => {
  const p = new PolicyActionStream('<think>');
  for (const s of ['</thi', 'nk>', '<a>', '+0+9', '+0+0', 'R', 'T']) assert.deepEqual(p.push(s), []);
  const first = p.push(';'); assert.equal(first.length,1); assert.equal(first[0].state.rightTrigger,1);
  assert.equal(p.push('RT;RT;;;').length,4);
  assert.equal(p.push('</a>')[0].index,5);
});
test('unknown tokens, centered -0, inconsistent 100ms repetitions fail', () => {
  assert.throws(()=>new PolicyActionStream().push('</think><a>-0+0+0+0;'));
  assert.throws(()=>new PolicyActionStream().push('</think><a>+0+0+0+0QQ;'));
  assert.throws(()=>new PolicyActionStream().push('</think><a>+0+0+0+0RT;LT;'));
});
test('Brain marker is reported and never mistaken for an ordinary action', () => {
  const p=new PolicyActionStream();assert.equal(p.push('阶段<|call_brain|></think><a>+0+0+0+0;')[0].brainCall,true);
});
test('actual HTTP stream exposes first state before response completion', async () => {
  const r={version:'think-memory-minimap-compact-v1',response_format:'full',generation_prefix:'<think>',template_sha256:'fixture'};
  let firstAt, endAt;
  const server=http.createServer((req,res)=>{
    req.resume();req.on('end',()=>{
      res.writeHead(200,{'content-type':'text/event-stream'});
      const send=e=>res.write('data: '+JSON.stringify({...e,lumine_session_id:'s',lumine_request_id:'r'})+'\n\n');
      send({type:'start',lumine_rendering:r});
      send({type:'delta',text:'</think><a>+0+9+0+0RT;',token_ids:[1,2]});
      setTimeout(()=>{send({type:'delta',text:'RT;RT;RT;RT;RT</a>',token_ids:[3]});
        send({type:'result',status:200,result:{lumine_rendering:r}});endAt=performance.now();res.end();},40);
    });
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  try {
    const result=await fetchPolicyStream(`http://127.0.0.1:${server.address().port}`,{method:'POST',body:'{}'},1000,null,
      {sessionId:'s',requestId:'r',rendering:r,onEvent:e=>{if(e.type==='state' && firstAt==null)firstAt=performance.now();}});
    assert.ok(firstAt<endAt);assert.ok(result.__client_timing.first_executable_state_ms<result.__client_timing.request_total_ms);
  } finally {await new Promise(resolve=>server.close(resolve));}
});

const { SubmissionGate } = require('../src/fixedSubmissionClock');
test('streamed prepared submission tolerates a late timer until its original action window expires', async () => {
  const t=performance.now();
  const gate=new SubmissionGate(0,t+1000,t+1200,true);
  const pending=gate.wait();
  gate.submitAt=performance.now()-30;gate.executeAt=performance.now()+170;
  gate.release();const receipt=await pending;
  assert.ok(receipt.submission_lateness_ms>=30);assert.equal(gate.dispatched,true);
  const expired=new SubmissionGate(1,performance.now()+1000,performance.now()+1200,true);
  const dead=expired.wait();expired.executeAt=performance.now()-201;expired.release();
  await assert.rejects(dead,/Original action window expired/);
});

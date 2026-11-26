import unittest
from gamepad_stream_scheduler import StreamScheduler

class StreamTests(unittest.TestCase):
    def setUp(self):
        self.t = 0.; self.applied=[]; self.releases=[]; self.events=[]
        self.s=StreamScheduler(lambda x:self.applied.append((self.t,x)), lambda:self.releases.append(self.t), self.events.append, now=lambda:self.t, threaded=False)
        self.s.arm('test', timeout_ms=500)
    def tick(self,t): self.t=t;self.s.tick()
    def state(self,seq,index,start,value):
        return self.s.update(dict(epoch='test',sequence=seq,index=index,startAtHelperMs=start,packageSeq=str(seq),state=value))
    def test_late_tail_clip_and_carry(self):
        for i in range(6):self.state(1,i,0,'old')
        for t in [0,34,67,100,134,167,200]:self.tick(t)
        self.assertEqual(self.releases,[0]) # no neutral reset at boundary
        self.t=230
        for i in range(6):self.state(2,i,200,'first' if i<3 else 'second')
        for t in [264,297,330,364,397,400]:self.tick(t)
        r=[e for e in self.events if e['event']=='gamepad_stream_finished' and e['sequence']==2][0]
        self.assertEqual(r['actualDurationMs'],170)
        self.assertEqual(r['events'][0]['appliedHelperMs'],230)
        self.assertEqual(r['events'][3]['appliedHelperMs'],330)
        self.assertEqual(r['events'][-1]['endedHelperMs'],400)
        self.assertFalse(self.state(2,5,200,'second')['accepted'])
        self.assertEqual(self.releases,[0])
    def test_watchdog_ignores_later_chunks_duplicates_and_stale_epochs(self):
        self.state(1,0,0,'valid')
        self.tick(150);self.state(1,1,0,'valid')
        self.tick(200)
        for t in [300,499]:
            self.tick(t);self.assertFalse(self.state(1,0,0,'valid')['accepted'])
        self.assertTrue(self.s.armed)
        self.tick(500);self.assertFalse(self.s.armed);self.assertEqual(self.releases[-1],500)
        self.assertFalse(self.state(2,0,500,'new')['accepted'])
        self.assertEqual(len([e for e in self.events if e['event']=='gamepad_stream_watchdog']),1)
    def test_missing_capture_or_http_also_times_out(self):
        self.s.arm('test',timeout_ms=500,reference_ms=180)
        self.tick(679);self.assertTrue(self.s.armed)
        self.tick(680);self.assertFalse(self.s.armed)
        self.assertEqual(self.events[-1]['elapsedMs'],500)

    def test_watchdog_release_precedes_native_pause(self):
        calls=[]
        self.s.pause=lambda: calls.append(('pause',self.t,self.releases[-1])) or {'pauseSent':True}
        self.s.arm('test',timeout_ms=500,pause_enabled=True)
        self.tick(500)
        self.assertEqual(calls,[('pause',500,500)])
        self.assertTrue(self.events[-1]['pauseSent'])
    def test_one_second_watchdog_default_and_late_packet_rejection(self):
        self.s.arm('test')
        self.state(1,0,0,'valid')
        self.tick(500);self.assertTrue(self.s.armed)
        self.tick(999);self.assertTrue(self.s.armed)
        self.tick(1000);self.assertFalse(self.s.armed)
        self.assertEqual(self.events[-1]['timeoutMs'],1000)
        self.assertFalse(self.state(2,0,1000,'late')['accepted'])

    def test_next_packet_on_time_cuts_predecessor(self):
        self.t=230
        for i in range(6):self.state(1,i,200,'a')
        self.t=350
        for i in range(6):self.state(2,i,400,'b')
        self.tick(400)
        self.assertEqual(self.applied[-1],(400,'b'))
    def test_cancel_is_final_and_invalid_releases(self):
        self.state(1,0,0,'x');self.s.finalize(dict(epoch='test',sequence=1,valid=False))
        self.tick(100)
        self.assertEqual(len(self.applied),1);self.assertFalse(self.s.armed)
        self.assertFalse(self.state(2,0,100,'y')['accepted'])
    def test_missing_decode_holds_without_bursting(self):
        self.state(1,0,0,'a');self.tick(120)
        self.assertEqual(len(self.applied),1)
        self.state(1,3,0,'b')
        self.assertEqual(self.applied[-1],(120,'b'))
        self.tick(200)
        self.assertEqual(len(self.applied),2)

if __name__=='__main__':unittest.main()

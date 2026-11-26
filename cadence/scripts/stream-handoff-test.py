"""Deterministic tests of the production native scheduler (no game/input device)."""
import copy
import unittest
from gamepad_stream_scheduler import StreamScheduler


def state(rt=1, buttons=None):
    return dict(leftStick=dict(x=0, y=0), rightStick=dict(x=0, y=0), leftTrigger=0,
                rightTrigger=rt, buttons=buttons or [])


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.t = 0.
        self.applied, self.releases, self.events = [], [], []
        self.s = StreamScheduler(lambda x: self.applied.append((self.t, copy.deepcopy(x))),
                                 lambda: self.releases.append(self.t), self.events.append,
                                 now=lambda: self.t, threaded=False)
        self.s.arm('A0', timeout_ms=1000)
        for index in range(6): self.update(1, index, 0)

    def tick(self, t):
        self.t = t
        self.s.tick()

    def update(self, seq, index, start, epoch='A0', owner='A', version=0, update=0, rt=1):
        return self.s.update(dict(epoch=epoch, ownerId=owner, ownershipVersion=version,
                                 updateVersion=update, sequence=seq, packageSeq=f'{epoch}:{seq}',
                                 index=index, startAtHelperMs=start, state=state(rt)))

    def handoff(self, **changes):
        msg = dict(epoch='A0', nextEpoch='B1', expectedOwner='A', nextOwner='B',
                   ownershipVersion=1, updateVersion=1, sequence=3, packageSeq='B1:3',
                   startAtHelperMs=200, observedAtHelperMs=20, maxObservationAgeMs=400,
                   states=[state() for _ in range(6)])
        msg.update(changes)
        return self.s.handoff(msg)

    def test_rt_handoff_keeps_state_and_revokes_pending_old_packets(self):
        self.t = 150
        self.update(2, 0, 200, rt=0)  # Must never execute after handoff.
        self.assertTrue(self.handoff()['accepted'])
        self.assertEqual(self.releases, [0])
        self.assertEqual(self.applied[-1][1]['rightTrigger'], 1)
        self.tick(200)
        self.assertEqual(self.applied[-1], (200, state()))
        self.assertTrue(all(s['rightTrigger'] == 1 for _, s in self.applied))
        self.assertEqual(self.releases, [0])
        applied = [e for e in self.events if e['event'] == 'gamepad_stream_handoff_applied']
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]['previousState']['rightTrigger'], 1)
        self.assertEqual(applied[0]['state']['rightTrigger'], 1)
        self.assertFalse(applied[0]['systemReleased'])
        discarded = [e for e in self.events if e['event'] == 'gamepad_stream_finished' and e['sequence'] == 2]
        self.assertFalse(discarded[0]['started'])
        self.assertEqual(discarded[0]['reason'], 'service_handoff')

    def test_empty_action_is_valid_and_releases_by_model_at_exact_window(self):
        self.t = 150
        self.assertTrue(self.handoff(states=[state(0) for _ in range(6)])['accepted'])
        self.tick(199)
        self.assertEqual(self.applied[-1][1]['rightTrigger'], 1)
        self.tick(200)
        self.assertEqual(self.applied[-1][1]['rightTrigger'], 0)
        self.assertEqual(self.releases, [0])

    def test_b_then_a_handoff_has_distinct_epochs_and_monotonic_owners(self):
        self.t = 180
        self.assertTrue(self.handoff()['accepted'])
        self.tick(200)
        self.t = 380
        self.assertTrue(self.handoff(epoch='B1', nextEpoch='A2', expectedOwner='B', nextOwner='A',
                                     ownershipVersion=2, updateVersion=3, sequence=5, packageSeq='A2:5',
                                     startAtHelperMs=400, observedAtHelperMs=210)['accepted'])
        self.tick(400)
        self.assertEqual(self.s.owner, 'A')
        self.assertEqual(self.s.ownership_version, 2)
        self.assertEqual(self.releases, [0])
        self.assertEqual([e['ownerId'] for e in self.events if e['event'] == 'gamepad_stream_handoff_applied'], ['B', 'A'])

    def test_old_epoch_and_old_owner_cannot_change_input_or_refresh_watchdog(self):
        self.t = 180
        self.handoff()
        self.tick(250)
        before = self.s.last_valid
        self.assertFalse(self.update(4, 0, 250, rt=0)['accepted'])
        self.assertFalse(self.update(4, 0, 250, epoch='B1', owner='A', version=0, rt=0)['accepted'])
        self.assertEqual(self.s.last_valid, before)
        self.assertTrue(all(s['rightTrigger'] == 1 for _, s in self.applied))

    def test_failed_handoff_preserves_active_queue_and_does_not_refresh_watchdog(self):
        self.t = 150
        self.update(2, 0, 200)
        before_valid, before_packets = self.s.last_valid, list(self.s.packets)
        bad = [dict(expectedOwner='B'), dict(ownershipVersion=0), dict(updateVersion=0),
               dict(nextOwner='A'), dict(nextEpoch='A0'), dict(observedAtHelperMs=-251),
               dict(states=[state()]), dict(states=[{'rightTrigger': 1}] * 6),
               dict(states=[state(buttons=['invalid'])] * 6), dict(startAtHelperMs=-50)]
        for changes in bad:
            self.assertFalse(self.handoff(**changes)['accepted'], changes)
            self.assertEqual(self.s.last_valid, before_valid)
            self.assertEqual(list(self.s.packets), before_packets)
        self.assertEqual(self.releases, [0])
        self.tick(200)
        self.assertEqual(self.s.active['sequence'], 2)

    def test_late_successor_is_clipped_at_original_absolute_end(self):
        self.tick(200)
        self.t = 230
        self.assertTrue(self.handoff(observedAtHelperMs=40)['accepted'])
        self.assertEqual(self.applied[-1][0], 230)
        self.tick(400)
        finished = [e for e in self.events if e['event'] == 'gamepad_stream_finished' and e['packageSeq'] == 'B1:3'][0]
        self.assertEqual(finished['actualDurationMs'], 170)
        self.assertEqual(finished['scheduledEndHelperMs'], 400)
        self.assertEqual(self.releases, [0])

    def test_handoff_does_not_revive_already_expired_active_watchdog(self):
        self.t = 1000  # Scheduler thread has not ticked yet; handoff races its deadline.
        ack = self.handoff(startAtHelperMs=1000, observedAtHelperMs=800)
        self.assertEqual(ack['reason'], 'watchdog_latched')
        self.assertEqual(self.releases, [0, 1000])
        self.assertFalse(self.s.armed)
        self.assertEqual(self.s.owner, 'A')

    def test_rejected_standby_responses_do_not_prolong_active_watchdog(self):
        for t in (400, 600, 800, 999):
            self.tick(t)
            self.handoff(startAtHelperMs=t, observedAtHelperMs=t, ownershipVersion=99)
        self.tick(1000)
        self.assertEqual(self.releases, [0, 1000])
        self.assertFalse(self.s.armed)

    def test_after_handoff_watchdog_requires_new_active_service_action(self):
        self.t = 180
        self.handoff()
        self.tick(1179)
        self.assertTrue(self.s.armed)
        self.tick(1180)
        self.assertFalse(self.s.armed)
        self.assertEqual(self.releases, [0, 1180])

    def test_finite_state_validation_precedes_old_queue_mutation(self):
        for bad_value in (float('nan'), float('inf'), 2, None, True):
            malformed = [state() for _ in range(6)]
            malformed[5]['rightTrigger'] = bad_value
            self.t = 100
            self.assertEqual(self.handoff(states=malformed)['reason'], 'incomplete_or_invalid_action')
            self.assertEqual(list(self.s.packets), [1])
            self.assertEqual(self.releases, [0])

if __name__ == '__main__': unittest.main()

"""Synthetic scheduler checks; no virtual or physical gamepad is used."""
import threading
import unittest
from gamepad_stream_scheduler import StreamScheduler


class WatchdogNotificationTests(unittest.TestCase):
    def fixture(self):
        self.now = 0
        self.events, self.releases = [], []
        self.scheduler = StreamScheduler(lambda state: None,
            lambda: self.releases.append(self.now), self.events.append,
            now=lambda: self.now, threaded=False)
        self.scheduler.arm('fixture-epoch', pause_enabled=True)

    def test_trigger_is_flushed_after_release_before_pause_hold(self):
        self.fixture()
        def pause():
            self.assertEqual(self.releases[-1], 1000)
            self.assertEqual(self.events[-1]['event'], 'gamepad_stream_watchdog_triggered')
            self.assertFalse(self.scheduler.armed)
            self.assertTrue(self.events[-1]['pausePending'])
            self.now += 150
            return {'pauseSent': True, 'pauseEndHelperMs': self.now}
        self.scheduler.pause = pause
        self.now = 1000
        self.scheduler.tick()
        trigger, complete = self.events[-2:]
        self.assertEqual(complete['event'], 'gamepad_stream_watchdog')
        self.assertEqual(trigger['watchdogId'], complete['watchdogId'])
        self.assertEqual(trigger['releasedHelperMs'], 1000)
        self.assertEqual(complete['pauseEndHelperMs'], 1150)
        self.assertEqual(complete['timeoutMs'], 1000)
        self.assertTrue(complete['pauseCompleted'])
        self.scheduler.tick()
        self.assertEqual(len(self.events), 2, 'one trigger and one completion only')

    def test_trigger_arrives_while_lock_is_held_and_commands_finish_after_pause(self):
        self.fixture()
        entered, finish = threading.Event(), threading.Event()
        command_started, command_done = threading.Event(), threading.Event()
        result = {}
        def pause():
            entered.set()
            if not finish.wait(2):
                raise TimeoutError('fixture pause was not released')
            return {'pauseSent': True}
        self.scheduler.pause = pause
        self.now = 1000
        tick = threading.Thread(target=self.scheduler.tick)
        tick.start()
        self.assertTrue(entered.wait(2))
        self.assertEqual([e['event'] for e in self.events], ['gamepad_stream_watchdog_triggered'])
        def late_command():
            command_started.set()
            result['ack'] = self.scheduler.update(dict(epoch='fixture-epoch', sequence=1,
                index=0, startAtHelperMs=1000, packageSeq='late', state='never-applied'))
            command_done.set()
        command = threading.Thread(target=late_command)
        try:
            command.start()
            self.assertTrue(command_started.wait(2))
            self.assertFalse(command_done.wait(.02), 'native command waits for scheduler pause lock')
        finally:
            finish.set()
            tick.join(2)
            command.join(2)
        self.assertFalse(tick.is_alive())
        self.assertFalse(command.is_alive())
        self.assertFalse(result['ack']['accepted'])
        self.assertEqual(self.events[-1]['event'], 'gamepad_stream_watchdog')

    def test_pause_failure_completes_barrier_with_explicit_failure(self):
        self.fixture()
        def fail():
            raise RuntimeError('fixture P key failure')
        self.scheduler.pause = fail
        self.now = 1000
        with self.assertRaisesRegex(RuntimeError, 'P key failure'):
            self.scheduler.tick()
        self.assertEqual([e['event'] for e in self.events],
            ['gamepad_stream_watchdog_triggered', 'gamepad_stream_watchdog'])
        self.assertFalse(self.events[-1]['pauseCompleted'])
        self.assertIn('P key failure', self.events[-1]['pauseError'])
        self.assertFalse(self.scheduler.armed)


if __name__ == '__main__':
    unittest.main()

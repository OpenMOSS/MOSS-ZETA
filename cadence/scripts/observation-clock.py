"""Independent capture/HTTP submission deadlines; no game input or image processing."""
import json
import sys
import time

print(json.dumps({'event': 'ready'}), flush=True)
config = json.loads(sys.stdin.readline())
epoch, period, offset = (float(config[k]) for k in ('epochMs', 'periodMs', 'offsetMs'))
assert period == 200 and 0 < offset < period
slot = 0
while True:
    now = time.perf_counter() * 1000
    slot = max(slot, int((now - epoch) // period))
    for event, delta in [('capture', 0), ('submit', offset)]:
        deadline = epoch + slot * period + delta
        remaining = deadline - time.perf_counter() * 1000
        if remaining > 0:
            time.sleep(remaining / 1000)
        print(json.dumps({'event': event, 'slot': slot, 'scheduledMs': deadline,
                          'emittedMs': time.perf_counter() * 1000}, separators=(',', ':')), flush=True)
    slot += 1

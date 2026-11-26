#!/usr/bin/env python3
"""Wait for an authenticated internal worker; never print credentials."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--timeout', type=float, default=1200)
args = p.parse_args()
c = json.loads(args.config.read_text())
key = Path(c['api_key_file']).read_text().strip()
deadline = time.monotonic() + args.timeout
while time.monotonic() < deadline:
    try:
        req = urllib.request.Request(f'http://127.0.0.1:{c["port"]}/health', headers={'Authorization': 'Bearer ' + key})
        with urllib.request.urlopen(req, timeout=5) as response:
            health = json.load(response)
        if 'think-memory-minimap-compact-v1' not in health.get('capabilities', []):
            raise SystemExit('Worker is running an incompatible recipe')
        if health['identity']['format_version'] != c['format_version']:
            raise SystemExit('Worker format differs from configured checkpoint')
        if health['identity']['model_revision'] != c['model_revision']:
            raise SystemExit('Worker loaded an unexpected model')
        print('Authenticated worker ready')
        break
    except (OSError, ValueError):
        time.sleep(2)
else:
    raise SystemExit('Worker readiness timed out')

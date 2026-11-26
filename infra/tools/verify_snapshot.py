"""Verify the exported source snapshot without access to original absolute paths."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'provenance/snapshot-files.json').read_text())
errors = []
for name, expected in manifest['sha256'].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(name)
if errors:
    raise SystemExit('Snapshot mismatch: ' + ', '.join(errors))
print('Verified %d snapshot files. This is source integrity, not deployment acceptance.' % len(manifest['sha256']))

"""Record source integrity without creating deployment acceptance evidence."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'provenance/snapshot-files.json'
IGNORED = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'results', 'state', 'cache', 'private', 'secrets'}


def main():
    files = {}
    for path in sorted(ROOT.rglob('*')):
        if path == DEST or any(part in IGNORED for part in path.relative_to(ROOT).parts) or not path.is_file():
            continue
        files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((ROOT.parent / 'cadence').rglob('*')):
        if path.is_file() and not any(part in IGNORED for part in path.relative_to(ROOT.parent).parts):
            files['../' + str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    DEST.write_text(json.dumps({'project': 'ZETA', 'component': 'infra', 'layout': 'ZETA monorepo v0.1.0',
        'meaning': 'Source integrity only; no new GPU, model-generation or real-game acceptance', 'sha256': files}, indent=2) + '\n')
    print(f'Bound {len(files)} source release files')


if __name__ == '__main__':
    main()

"""Bind the newly integrated export; never rewrite the upstream frozen inventory."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'provenance/release-files.json'
DIRECTORIES = ('harness', 'infra', 'cadence', 'website', 'tools', 'provenance', '.github')
IGNORE = {'.git', 'node_modules', '__pycache__', '.pytest_cache', 'captures', 'logs', 'artifacts', 'private', 'secrets', 'state', 'results', 'cache'}


def main():
    files = {}
    candidates = [ROOT / name for name in ('README.md', 'README.zh-CN.md', 'AGENTS.md', 'package.json', 'package-lock.json', '.gitignore')]
    for directory in DIRECTORIES:
        candidates.extend((ROOT / directory).rglob('*'))
    # Include retained files from the historical upstream inventory, including
    # vendor paths with ignored directory names. Omitted research files stay out.
    upstream = json.loads((ROOT / 'infra/provenance/upstream-snapshot-files.json').read_text())['sha256']
    curated = set(upstream) | set(json.loads((ROOT / 'infra/provenance/source-files.json').read_text())['files'])
    candidates.extend(ROOT / name if name.startswith('cadence/') else ROOT / 'infra' / name for name in curated)
    for path in sorted(set(candidates)):
        if path == DEST or not path.is_file():
            continue
        relative = str(path.relative_to(ROOT))
        upstream_file = relative.startswith('infra/') and str(path.relative_to(ROOT / 'infra')) in curated
        if any(part in IGNORE for part in path.relative_to(ROOT).parts) and not upstream_file:
            continue
        files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    DEST.write_text(json.dumps({'project': 'ZETA', 'version': '0.1.0',
        'meaning': 'Integrated source identity; no new model, GPU or live-game acceptance', 'sha256': files}, indent=2) + '\n')
    print(f'Bound {len(files)} integrated source files')


if __name__ == '__main__':
    main()

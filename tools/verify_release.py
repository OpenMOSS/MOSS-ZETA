"""Check integrated source identity, frozen inference code, media hashes and documentation links."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ('runtime/frozen_vllm/', 'runtime/helpers/', 'serving/')
RESEARCH_ONLY = ('validation-runtime', 'acceptance')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'provenance/release-files.json').read_text())
    errors = [name for name, value in manifest['sha256'].items()
              if not (ROOT / name).is_file() or digest(ROOT / name) != value]
    if errors:
        raise SystemExit('Integrated source mismatch: ' + ', '.join(errors))
    for name in RESEARCH_ONLY:
        assert not (ROOT / 'infra' / name).exists(), 'Research-only directory is outside the source release: ' + name
    upstream = json.loads((ROOT / 'infra/provenance/upstream-snapshot-files.json').read_text())['sha256']
    for name, record in json.loads((ROOT / 'infra/provenance/source-files.json').read_text())['files'].items():
        upstream.setdefault(name, record['sha256'])
    protected = 0
    for name, value in upstream.items():
        if name.startswith(PROTECTED):
            assert digest(ROOT / 'infra' / name) == value, 'Frozen inference source changed: ' + name
            protected += 1
    demos = json.loads((ROOT / 'website/demos.json').read_text())
    for row in [demos['featured'], *demos['clips']]:
        for field in ('video', 'poster'):
            assert digest(ROOT / 'website' / row[field]) == row[field + '_sha256'], row[field]
    paper = json.loads((ROOT / 'website/paper-figures.json').read_text())
    for row in paper['figures']:
        assert digest(ROOT / 'website' / row['image']) == row['image_sha256'], row['image']
        assert digest(ROOT / 'website' / row['pdf']) == row['source_sha256'], row['pdf']
    reported = json.loads((ROOT / 'website/paper-content.json').read_text())
    scale = reported['data_scale']
    assert scale['action_pretraining_hours_approx'] == scale['free_play_hours_approx'] + scale['task_directed_hours_approx']
    assert scale['instruction_tuning_hours_approx'] <= scale['task_directed_hours_approx']
    docs = [ROOT / 'README.md', ROOT / 'README.zh-CN.md', ROOT / 'harness/README.md', ROOT / 'cadence/README.md']
    docs += [ROOT / 'infra/README.md', *list((ROOT / 'infra/docs').rglob('*.md'))]
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if re.match(r'https?:|mailto:|#', target):
                continue
            assert (path.parent / target.split('#')[0]).exists(), f'Broken documentation link: {path}: {target}'
    print(f'Verified {len(manifest["sha256"])} integrated files, {protected} unchanged inference files, 1 promotional demo, {len(demos["clips"])} gameplay clips and {len(paper["figures"])} paper figures.')
    print('Source and CPU checks do not establish GPU, autonomous-generation or live-game acceptance.')


if __name__ == '__main__':
    main()

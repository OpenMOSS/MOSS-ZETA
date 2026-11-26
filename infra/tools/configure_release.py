"""Generate fresh instance configs; never rewrite a model or frozen source."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def materialize(value, replacements):
    if isinstance(value, dict):
        return {key: materialize(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    dest = args.run_dir.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    replacements = {'/CHECKOUT': str(ROOT), '/ASSETS/ZETA/model': str(args.model_dir.resolve()),
                    '/PRIVATE': str(args.private_dir.resolve()), '/NEW_RUN': str(dest)}
    for name in ('a', 'b', 'gateway', 'cadence'):
        config = json.loads((ROOT / f'configs/{name}.example.json').read_text())
        (dest / f'{name}.json').write_text(json.dumps(materialize(config, replacements), indent=2) + '\n')
    print(dest)


if __name__ == '__main__':
    main()

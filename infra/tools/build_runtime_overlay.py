"""Combine release Python sources with verified, separately supplied native libraries.

The destination must be new. Neither the source environment nor model is edited.
Use the same Python/CUDA/Torch ABI as runtime/observed-packages.txt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-root', type=Path, required=True, help='Directory containing the matching vllm/ package')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output must be a new directory; existing environments are never patched')
    manifest = json.loads((ROOT / 'runtime/binary-manifest.json').read_text())
    binaries = []
    for item in manifest['files']:
        path = args.binary_root / item['path']
        if not path.is_file() or sha(path) != item['sha256']:
            raise SystemExit('Native library mismatch: ' + item['path'])
        binaries.append((item['path'], path.resolve()))
    shutil.copytree(ROOT / 'runtime/frozen_vllm', args.output)
    for name, source in binaries:
        dest = args.output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source)
    print(args.output.resolve())


if __name__ == '__main__':
    main()

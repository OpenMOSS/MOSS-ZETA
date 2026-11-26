"""Run a JSONL observation stream through two deployed models.

The first line is {format_version, state} or {format_version, prefix_messages,
images: {sha256: path}}. Subsequent lines contain image_path, ocr, and optional
state/reset_state/request_id. Each line is consumed only after the previous
inference returns. This replays recorded pixels, not game action execution.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from .client import DualPipeline
from .protocol import Context, system_prompt


async def run(args):
    config = json.loads(args.config.read_text())
    keys = [Path(p).read_text().strip() for p in config['api_key_files']]
    template = Path(config['system_template']).read_text().strip()
    with args.observations.open() as source:
        first = json.loads(next(source))
        version = first['format_version']
        if 'prefix_messages' in first:
            context = Context.from_prefix(first['prefix_messages'], version)
        else:
            context = Context(system_prompt(template, first['state'], version), version)
        async with DualPipeline(config['endpoints'], keys, context,
                                system_template=template, policy=args.policy,
                                max_tokens=args.max_tokens) as pipeline:
            for sha, filename in first.get('images', {}).items():
                if pipeline.add_image(Path(filename).read_bytes()) != sha:
                    raise ValueError('Seed image hash mismatch')
            with args.output.open('x') as output:
                for line in source:
                    observation = json.loads(line)
                    result = await pipeline.step(Path(observation['image_path']).read_bytes(),
                        observation['ocr'], state=observation.get('state'),
                        reset_state=observation.get('reset_state', False),
                        request_id=observation.get('request_id'))
                    output.write(json.dumps(result, ensure_ascii=False)+'\n')
                    output.flush()
                    print(json.dumps(dict(turn=result['turn'], route=result['route'],
                        latency_ms=result['client_total_ms'], tokens=result['usage']['completion_tokens'],
                        parsed=result['parsed']), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', choices=['prefetch', 'serial'], default='prefetch')
    parser.add_argument('--max-tokens', type=int, default=256)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()

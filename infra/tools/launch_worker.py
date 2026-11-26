"""Launch a report-profile H200 worker with the matching compiled runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_runtime(config, runtime):
    engine = config['engine']
    if (engine['tensor_parallel_size'], engine['dtype'], config['encoder_cache_tokens']) != (4, 'bfloat16', 32768):
        raise ValueError('Expected final TP4 / BF16 / encoder-cache32768 profile')
    graph = engine['compilation_config']
    if graph['max_cudagraph_capture_size'] != 8192 or not graph['cudagraph_mm_encoder']:
        raise ValueError('Final profile requires decoder and encoder CUDA Graph coverage')
    for relative in ('config/scheduler.py', 'v1/core/encoder_cache_manager.py'):
        if sha(runtime / 'vllm' / relative) != sha(ROOT / 'runtime/frozen_vllm/vllm' / relative):
            raise ValueError('Compiled runtime does not have the final source overlay: ' + relative)
    binaries = json.loads((ROOT / 'runtime/binary-manifest.json').read_text())
    for item in binaries['files']:
        path = runtime / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError('Missing matching native library; run build_runtime_overlay.py: ' + item['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--vllm-runtime', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = args.vllm_runtime.resolve()
    check_runtime(config, runtime)
    os.environ['ZETA_ENCODER_CACHE_TOKENS'] = str(config['encoder_cache_tokens'])
    sys.path[:0] = [str(runtime), str(ROOT / 'serving/dual_pipeline'), str(ROOT / 'serving'), str(ROOT / 'runtime/helpers')]
    import vllm
    if Path(vllm.__file__).resolve().parent != runtime / 'vllm':
        raise ValueError('Imported an unexpected vLLM package')
    import asyncio
    from lumine_dual.worker import serve
    asyncio.run(serve(config))


if __name__ == '__main__':
    main()

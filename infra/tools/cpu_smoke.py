"""Component tests only: no model loading, GPU or generation acceptance."""
from pathlib import Path
import copy
from contextlib import contextmanager
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'serving/dual_pipeline'), str(ROOT / 'serving')]
from lumine_dual.protocol import Context, VERSION, parse_action
from lumine_dual.wire import generation_options
from lumine_dual.lossless_png import encode_rgb
from lumine_protocol import can_continue, load_profile, reconstruct
from configure_release import materialize
import launch_worker


class ServingTests(unittest.TestCase):
    def test_empty_small_and_brain_protocol_envelopes(self):
        for thought in ('', '保持路线。', '阶段需要复核。<|call_brain|>{"instruction":"继续任务","small_think":"检查目标。"}'):
            result = parse_action('<think>' + thought + '</think><a>+0+0+0+0;;;;;</a>', VERSION)
            self.assertEqual(result['call_brain'], '<|call_brain|>' in thought)
        profile = load_profile()
        self.assertEqual(profile['generation_prefix'], '<think>')
        row = reconstruct('</think><a>+0+0+0+0;;;;;</a>', [248069], 'stop', '</a>', profile, True)
        self.assertTrue(row['protocol_valid'])

    def test_thinking_cannot_be_disabled_and_sampling_is_forwarded(self):
        with self.assertRaises(ValueError):
            generation_options({'chat_template_kwargs': {'enable_thinking': False}})
        options = generation_options({'temperature': 0.7, 'top_p': 0.9, 'max_tokens': None})
        self.assertEqual((options['temperature'], options['top_p'], options['max_tokens']), (0.7, 0.9, None))

    def test_context_16_8_trim(self):
        context = Context('state', VERSION, window_max=16, window_keep=8)
        observed = []
        for frame in range(25):
            ids = [hashlib.sha256(f'{frame}:{which}'.encode()).hexdigest() for which in ('full', 'map')]
            block = '<screen_ocr>\nS\n</screen_ocr>' if frame == 0 else '<screen_ocr>\n</screen_ocr>'
            candidate, user, messages, prefix = context.stage(ids, block)
            observed.append(len(candidate.turns))
            self.assertEqual(messages[-1], user)
            self.assertEqual(prefix[-1]['role'], 'user')
            candidate.turns.append((user, '<think></think><a>+0+0+0+0;;;;;</a>'))
            context = candidate
        self.assertEqual((observed[15], observed[16], observed[24]), (15, 8, 8))

    def test_prefix_reuse_requires_exact_tokens_and_ordered_media(self):
        media = [('image', 0, 1, 'same')]
        self.assertTrue(can_continue([1, 2, 3], media, [1, 2], media))
        self.assertFalse(can_continue([1, 9, 3], media, [1, 2], media))
        self.assertFalse(can_continue([1, 2, 3], [('image', 0, 1, 'different')], [1, 2], media))

    def test_png_preserves_rgb_pixels(self):
        from PIL import Image
        image = Image.frombytes('RGB', (4, 3), bytes(range(36)))
        recovered = Image.open(io.BytesIO(encode_rgb(image)))
        self.assertEqual((recovered.mode, recovered.tobytes()), ('RGB', image.tobytes()))

    def test_portable_final_config(self):
        source = json.loads((ROOT / 'configs/a.example.json').read_text())
        config = materialize(source, {'/ASSETS/ZETA/model': '/tmp/model', '/CHECKOUT': '/tmp/code', '/NEW_RUN': '/tmp/run', '/PRIVATE': '/tmp/private'})
        self.assertEqual(config['engine']['model'], '/tmp/model')
        self.assertEqual(config['engine']['tensor_parallel_size'], 4)
        self.assertEqual(config['engine']['compilation_config']['max_cudagraph_capture_size'], 8192)
        self.assertEqual((config['encoder_cache_tokens'], config['prefix_cache_scope']), (32768, 'session'))


class LauncherTests(unittest.TestCase):
    @contextmanager
    def runtime_fixture(self):
        # Small synthetic package to test launch wiring without importing CUDA.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / 'infra'
            runtime = root / 'compiled-runtime'
            for relative in ('config/scheduler.py', 'v1/core/encoder_cache_manager.py'):
                for package in (release / 'runtime/frozen_vllm/vllm', runtime / 'vllm'):
                    dest = package / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / 'runtime/frozen_vllm/vllm' / relative, dest)
            (runtime / 'vllm/test-native.so').write_bytes(b'synthetic')
            (release / 'runtime/binary-manifest.json').write_text(json.dumps({
                'files': [{'path': 'vllm/test-native.so', 'bytes': 9}]}))
            config = json.loads((ROOT / 'configs/a.example.json').read_text())
            with patch.object(launch_worker, 'ROOT', release):
                yield root, runtime, config

    def test_launcher_needs_only_config_and_runtime(self):
        with self.runtime_fixture() as (root, runtime, config):
            config_path = root / 'a.json'
            config_path.write_text(json.dumps(config))
            vllm = types.ModuleType('vllm')
            vllm.__file__ = str(runtime / 'vllm/__init__.py')
            worker = types.ModuleType('lumine_dual.worker')
            worker.serve = AsyncMock()
            with patch.dict(sys.modules, {'vllm': vllm, 'lumine_dual.worker': worker}), \
                 patch.object(sys, 'argv', ['launch_worker.py', '--config', str(config_path),
                                           '--vllm-runtime', str(runtime)]), \
                 patch.object(sys, 'path', sys.path.copy()), patch.dict(os.environ):
                launch_worker.main()
                worker.serve.assert_awaited_once_with(config)
                self.assertEqual(os.environ['ZETA_ENCODER_CACHE_TOKENS'], '32768')
                self.assertEqual(sys.path[0], str(runtime))

    def test_launcher_rejects_incompatible_profile(self):
        with self.runtime_fixture() as (_, runtime, config):
            for field, value in (('tensor_parallel_size', 1), ('dtype', 'float16')):
                invalid = copy.deepcopy(config)
                invalid['engine'][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Expected final'):
                    launch_worker.check_runtime(invalid, runtime)
            invalid = copy.deepcopy(config)
            invalid['encoder_cache_tokens'] = 16384
            with self.assertRaisesRegex(ValueError, 'Expected final'):
                launch_worker.check_runtime(invalid, runtime)
            for field, value in (('max_cudagraph_capture_size', 4096), ('cudagraph_mm_encoder', False)):
                invalid = copy.deepcopy(config)
                invalid['engine']['compilation_config'][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'CUDA Graph'):
                    launch_worker.check_runtime(invalid, runtime)

    def test_launcher_rejects_wrong_overlay_and_missing_native_library(self):
        with self.runtime_fixture() as (_, runtime, config):
            source = runtime / 'vllm/config/scheduler.py'
            original = source.read_bytes()
            source.write_bytes(b'wrong overlay')
            with self.assertRaisesRegex(ValueError, 'final source overlay'):
                launch_worker.check_runtime(config, runtime)
            source.write_bytes(original)
            (runtime / 'vllm/test-native.so').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing matching native library'):
                launch_worker.check_runtime(config, runtime)


if __name__ == '__main__':
    unittest.main()

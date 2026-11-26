"""Measure renderer stages in its worker thread without changing model inputs."""
from collections import defaultdict
import cProfile
import pstats
import time
import threading
from pathlib import Path

def install(processor, output):
    import torch
    local = threading.local()
    counts = [0]
    for name in ['_cached_apply_hf_processor', '_get_hf_mm_data', '_get_cache_missing_items', '_apply_hf_processor_main', '_apply_hf_processor_text_only', '_apply_hf_processor_mm_only', '_merge_mm_kwargs', '_maybe_apply_prompt_updates', '_apply_prompt_updates', '_find_mm_placeholders']:
        if not hasattr(processor, name): continue
        original = getattr(processor, name)
        def timed(*args, _name=name, _fn=original, **kwargs):
            start = time.perf_counter()
            try: return _fn(*args, **kwargs)
            finally:
                if getattr(local, 'times', None) is not None:
                    if _name == '_apply_hf_processor_mm_only': local.times['hf_torch_threads'] = torch.get_num_threads()
                    local.times[_name] += (time.perf_counter()-start)*1000
        setattr(processor, name, timed)
    original = processor.apply
    last = {}
    def apply(*args, **kwargs):
        local.times = defaultdict(float)
        counts[0] += 1
        profile = None  # Detailed profiling was isolated to v1; scalar stage timers remain.
        start = time.perf_counter()
        if profile: profile.enable()
        try: return original(*args, **kwargs)
        finally:
            if profile:
                profile.disable()
                p = Path(output)/('renderer-profile-%d.txt'%counts[0])
                with p.open('w') as f:pstats.Stats(profile,stream=f).sort_stats('cumulative').print_stats(70)
            last.clear();last.update(local.times);last['apply_total_ms']=(time.perf_counter()-start)*1000
            last['profiled']=bool(profile);local.times=None
    processor.apply = apply
    return last

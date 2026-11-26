"""Publish an exact Mamba checkpoint before the growing OCR archive ends.

Only explicit preparation requests opt in. Token contents, positions, attention,
and the stock final checkpoint are unchanged. Prefix hashes still determine
whether a later request may reuse the saved state.
"""
from math import lcm
from foreground_scheduler import ForegroundScheduler
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE


def checkpoint_chunk(computed, proposed, boundary, block_size):
    target = max(0, boundary // block_size * block_size)
    return target - computed if computed < target < computed + proposed else proposed


class LatencyScheduler(ForegroundScheduler):
    def _mamba_block_aligned_split(self, request, num_new_tokens,
                                 num_new_local_computed_tokens=0,
                                 num_external_computed_tokens=0):
        stock = super()._mamba_block_aligned_split(
            request, num_new_tokens, num_new_local_computed_tokens,
            num_external_computed_tokens)
        params = request.sampling_params
        boundary = (params.extra_args or {}).get('ocr_prefix_checkpoint', 0) if params else 0
        if (type(boundary) is int and boundary > 0 and not self.use_eagle
                and num_external_computed_tokens == 0):
            computed = request.num_computed_tokens + num_new_local_computed_tokens
            # Preserve the recurrent kernel's chunk phase as well as the cache
            # block boundary. This runtime uses cache blocks of 528 tokens and
            # FLA chunks of 64: their shared boundary is 2112 tokens.
            alignment = lcm(self.cache_config.block_size, FLA_CHUNK_SIZE)
            return checkpoint_chunk(computed, stock, boundary, alignment)
        return stock

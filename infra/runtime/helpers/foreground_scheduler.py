"""Experimental, explicit request opt-in: omit the final extra APC checkpoint.

This keeps exact existing prefix-cache reads and normal model computation. It
does not shift, truncate, quantize or reuse invalid KV/Mamba states. A foreground
request whose resulting prefix will not be reused need not split its final
prefill solely to publish another block-aligned Mamba cache checkpoint.
Preparation requests retain stock scheduling. Partial budget-limited prefills
also retain stock alignment. Only ordinary decode / max_num_seqs=1 is tested.
"""
import json,os
from pathlib import Path
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

class ForegroundScheduler(AsyncScheduler):
    def _mamba_block_aligned_split(self,request,num_new_tokens,num_new_local_computed_tokens=0,num_external_computed_tokens=0):
        stock=super()._mamba_block_aligned_split(request,num_new_tokens,num_new_local_computed_tokens,num_external_computed_tokens)
        params=request.sampling_params
        opted=params is not None and (params.extra_args or {}).get('omit_final_prefix_checkpoint',False)
        computed=request.num_computed_tokens+num_new_local_computed_tokens
        if (opted and not self.use_eagle and num_external_computed_tokens==0
                and computed<request.num_prompt_tokens
                and computed+num_new_tokens>=request.num_prompt_tokens
                and stock!=num_new_tokens):
            path=os.environ.get('FOREGROUND_SCHEDULER_LOG')
            if path:
                with Path(path).open('a') as out:out.write(json.dumps({'request_id':request.request_id,'computed':computed,'prompt_tokens':request.num_prompt_tokens,'stock_chunk':stock,'foreground_chunk':num_new_tokens})+'\n')
            return num_new_tokens
        return stock

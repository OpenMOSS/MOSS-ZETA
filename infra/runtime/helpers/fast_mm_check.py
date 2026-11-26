"""Keep Transformers' multimodal token validation; avoid scalar Tensor loops.

Only the read-only view supplied to the existing validator is converted.
The token IDs, pixels, tokenizer, truncation errors and returned tensors stay intact.
Opt-in, process-local patch; shared Transformers installation is never edited.
"""
import torch
from transformers.processing_utils import ProcessorMixin

ORIGINAL=ProcessorMixin._check_special_mm_tokens

def fast_check(self,text,text_inputs,modalities):
    ids=text_inputs['input_ids']
    if isinstance(ids,torch.Tensor) and ids.device.type=='cpu' and ids.ndim==2:
        text_inputs={**text_inputs,'input_ids':ids.tolist()}
    return ORIGINAL(self,text,text_inputs,modalities)

def install():
    ProcessorMixin._check_special_mm_tokens=fast_check

def validate():
    """Exercise success, malformed image/video counts, lists, and empty batches."""
    from types import SimpleNamespace
    obj=SimpleNamespace(image_token='<image>',image_token_id=11,video_token='<video>',video_token_id=22)
    cases=[(['<image>x<video>'],[[11,0,22]]),(['<image><image>'],[[11,11]]),(['<image><image>'],[[11,0]]),(['<video>'],[[11,0]]),(['plain'],[[0,0]]),([],[])]
    checks=[]
    for text,ids in cases:
        for tensor in [False,True]:
            values=torch.tensor(ids,dtype=torch.int64).reshape(len(ids),-1) if ids else torch.empty((0,0),dtype=torch.int64)
            inp={'input_ids':values if tensor else ids,'sentinel':'unchanged'}
            outcomes=[]
            for fn in [ORIGINAL,fast_check]:
                try:fn(obj,text,inp,['image','video','audio']);outcomes.append(None)
                except ValueError as e:outcomes.append(str(e))
            assert outcomes[0]==outcomes[1],outcomes
            assert inp['sentinel']=='unchanged'
            if tensor:assert inp['input_ids'] is values
            checks.append({'text':text,'tensor':tensor,'same_error_or_success':True,'raised':outcomes[0] is not None})
    return checks

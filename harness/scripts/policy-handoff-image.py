"""Exact frozen-profile minimap pixels; stdin is the unchanged original frame."""
import io
import sys
from PIL import Image

with Image.open(io.BytesIO(sys.stdin.buffer.read())) as frame:
    if frame.size != (1280, 720):
        raise ValueError('Policy handoff requires a 1280x720 original frame')
    crop = frame.convert('RGB').crop((0, 528, 256, 720)).resize((512, 384), Image.Resampling.BICUBIC)
    crop.save(sys.stdout.buffer, format='PNG')

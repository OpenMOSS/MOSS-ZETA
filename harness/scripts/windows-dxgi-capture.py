"""Opt-in DXGI capture worker. Pin dxcam 0.3.0; JSON-lines IPC, one request at a time."""
import base64
import ctypes
import io
import json
import os
import sys
import time
from ctypes import wintypes

started = time.perf_counter()
if os.environ.get('AGI_GTA_CAPTURE_PYTHONPATH'):
    sys.path.insert(0, os.environ['AGI_GTA_CAPTURE_PYTHONPATH'])
import cv2
import dxcam
import numpy as np
from PIL import Image
from importlib.metadata import version

if version('dxcam') != '0.3.0':
    raise RuntimeError('This capture adapter requires dxcam==0.3.0')
cv2.setNumThreads(8)
u = ctypes.WinDLL('user32', use_last_error=True)
u.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]

def now(): return time.perf_counter() * 1000
def emit(record):
    sys.stdout.write(json.dumps(record, separators=(',', ':')) + '\n')
    sys.stdout.flush()

camera = dxcam.create(device_idx=int(os.environ.get('AGI_GTA_CAPTURE_DEVICE', '0')),
                     output_idx=int(os.environ.get('AGI_GTA_CAPTURE_OUTPUT', '0')),
                     output_color='BGR', max_buffer_len=2)
try:
    emit({'event': 'ready', 'backend': 'windows-dxgi', 'qpcMs': now(),
          'startupMs': (time.perf_counter() - started) * 1000,
          'displayId': camera._output.devicename, 'dxcamVersion': version('dxcam')})
    for line in sys.stdin:
        request = {}; begin = now()
        try:
            request = json.loads(line)
            if request.get('type') == 'clock':
                emit({'id': request['id'], 'ok': True, 'qpcMs': now()}); continue
            if request.get('type') != 'capture': raise ValueError('Unsupported capture request')
            display = str(request.get('displayId') or '')
            if display and display.lower() != camera._output.devicename.lower():
                raise ValueError('Requested display does not match configured DXGI output')
            width, height = int(request['width']), int(request['height'])
            if not (1 <= width <= 8192 and 1 <= height <= 8192): raise ValueError('Invalid output size')
            # No previous image fallback: absent new frames remain explicit misses.
            frame = camera.grab(new_frame_only=True)
            while frame is None and now() - begin < 25:
                time.sleep(.002); frame = camera.grab(new_frame_only=True)
            if frame is None: raise RuntimeError('no_new_frame')
            acquired = now()
            frame_ms = camera._duplicator.latest_frame_time * 1000
            sh, sw = frame.shape[:2]
            scale = min(width / sw, height / sh)
            cw, ch = max(1, round(sw * scale)), max(1, round(sh * scale))
            left, top = (width - cw) // 2, (height - ch) // 2
            resized = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)
            if (cw, ch) != (width, height):
                padded = np.zeros((height, width, 3), np.uint8)
                padded[top:top+ch, left:left+cw] = resized; resized = padded
            resized_at = now()
            stream = io.BytesIO()
            Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).save(
                stream, format='JPEG', quality=max(1, min(100, int(request.get('quality', 85)))), subsampling=2)
            encoded_at = now()
            data = base64.b64encode(stream.getvalue()).decode('ascii')
            ready = now(); cursor = wintypes.POINT(); cursor_ok = u.GetCursorPos(ctypes.byref(cursor))
            emit({'id': request['id'], 'ok': True, 'mimeType': 'image/jpeg', 'width': width, 'height': height,
                  'contentLeft': left, 'contentTop': top, 'contentWidth': cw, 'contentHeight': ch,
                  'data': data, 'backend': 'windows-dxgi', 'cursorIncluded': False,
                  'cursorX': cursor.x if cursor_ok else -1, 'cursorY': cursor.y if cursor_ok else -1,
                  'requestQpcMs': begin, 'frameQpcMs': frame_ms, 'readyQpcMs': ready,
                  'frameTimestampSource': 'DXGI desktop update (LastPresentTime; mouse-update fallback)',
                  'helperTotalMs': ready-begin, 'acquireMs': acquired-begin,
                  'resizeMs': resized_at-acquired, 'jpegEncodeMs': encoded_at-resized_at,
                  'base64Ms': ready-encoded_at, 'frameAgeAtHelperReadyMs': ready-frame_ms})
        except Exception as error:
            emit({'id': request.get('id'), 'ok': False, 'error': str(error), 'readyQpcMs': now()})
finally:
    camera.release()

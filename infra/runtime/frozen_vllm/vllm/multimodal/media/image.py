# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import time
from threading import Lock
from typing import Any

import numpy as np
import pybase64
import torch
from PIL import Image

from vllm.utils.serial_utils import tensor2base64

from ..image import convert_image_mode, rgba_to_rgb
from .base import MediaIO, MediaWithBytes

# Process-local decoded-image cache; never keyed by URL/filename.
_IMAGE_CACHE = OrderedDict()
_IMAGE_CACHE_LOCK = Lock()
_IMAGE_CACHE_BYTES = 0
_IMAGE_STATS = dict(hits=0, misses=0, evictions=0)


def image_cache_stats():
    with _IMAGE_CACHE_LOCK:
        return {**_IMAGE_STATS, "bytes": _IMAGE_CACHE_BYTES, "entries": len(_IMAGE_CACHE)}


MAGIC_NUMPY_PREFIX = b"\x93NUMPY"  # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0


class ImageMediaIO(MediaIO[Image.Image]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self, image_mode: str = "RGB", **kwargs) -> None:
        super().__init__()

        self.image_mode = image_mode
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

        # Extract RGBA background color from kwargs if provided
        # Default to white background for backward compatibility
        rgba_bg = kwargs.get("rgba_background_color", (255, 255, 255))
        # Convert list to tuple for consistency
        if isinstance(rgba_bg, list):
            rgba_bg = tuple(rgba_bg)

        # Validate rgba_background_color format
        if not (
            isinstance(rgba_bg, tuple)
            and len(rgba_bg) == 3
            and all(isinstance(c, int) and 0 <= c <= 255 for c in rgba_bg)
        ):
            raise ValueError(
                "rgba_background_color must be a list or tuple of 3 integers "
                "in the range [0, 255]."
            )
        self.rgba_background_color = rgba_bg

    def _convert_image_mode(
        self, image: Image.Image | MediaWithBytes[Image.Image]
    ) -> Image.Image:
        """Convert image mode with custom background color."""
        if isinstance(image, MediaWithBytes):
            image = image.media
        if image.mode == self.image_mode:
            return image
        elif image.mode == "RGBA" and self.image_mode == "RGB":
            return rgba_to_rgb(image, self.rgba_background_color)
        else:
            return convert_image_mode(image, self.image_mode)

    def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image]:
        return self._load_bytes_timed(data, None)

    @staticmethod
    def _add_timing(
        timing: dict[str, float] | None,
        name: str,
        elapsed: float,
        timing_lock: Any = None,
    ) -> None:
        if timing is not None:
            if timing_lock is None:
                timing[name] = timing.get(name, 0.0) + elapsed * 1000.0
            else:
                with timing_lock:
                    timing[name] = timing.get(name, 0.0) + elapsed * 1000.0

    def _load_bytes_timed(
        self,
        data: bytes,
        timing: dict[str, float] | None,
        timing_lock: Any = None,
    ) -> MediaWithBytes[Image.Image]:
        global _IMAGE_CACHE_BYTES
        enabled = os.getenv("VLLM_LUMINE_CONTEXT_FORMAT") == "v0930_v8_9"
        limit = int(os.getenv("VLLM_LUMINE_IMAGE_CACHE_BYTES", "268435456")) if enabled else 0
        ttl = float(os.getenv("VLLM_LUMINE_IMAGE_CACHE_TTL", "300"))
        key = None
        if limit > 0:
            key = hashlib.sha256(data + json.dumps(
                [os.getenv("VLLM_LUMINE_CACHE_NAMESPACE", ""), self.image_mode, self.kwargs, os.getenv("VLLM_LUMINE_IMAGE_PREPROCESSING", "raw")],
                sort_keys=True, default=str).encode()).hexdigest()
            now = time.monotonic()
            with _IMAGE_CACHE_LOCK:
                for old in list(_IMAGE_CACHE):
                    if _IMAGE_CACHE[old][0] <= now:
                        _IMAGE_CACHE_BYTES -= _IMAGE_CACHE.pop(old)[2]
                        _IMAGE_STATS["evictions"] += 1
                hit = _IMAGE_CACHE.get(key)
                if hit is not None:
                    _IMAGE_CACHE.move_to_end(key)
                    _IMAGE_STATS["hits"] += 1
                    return MediaWithBytes(hit[1].copy(), data)
                _IMAGE_STATS["misses"] += 1
        try:
            open_started = time.perf_counter()
            image = Image.open(BytesIO(data))
            self._add_timing(
                timing,
                "media_image_container_open_ms",
                time.perf_counter() - open_started,
                timing_lock,
            )
            decode_started = time.perf_counter()
            image.load()
            self._add_timing(
                timing,
                "media_image_decode_ms",
                time.perf_counter() - decode_started,
                timing_lock,
            )
            mode_started = time.perf_counter()
            image = self._convert_image_mode(image)
            if os.getenv("VLLM_LUMINE_IMAGE_PREPROCESSING", "raw") == "training_qwen16_png":
                from qwen_vl_utils import fetch_image
                image = fetch_image({"image": image}, image_patch_size=16)
            self._add_timing(
                timing,
                "media_image_mode_convert_ms",
                time.perf_counter() - mode_started,
                timing_lock,
            )
        except (OSError, Image.UnidentifiedImageError) as e:
            raise ValueError(f"Failed to load image: {e}") from e
        if key is not None:
            size = image.width * image.height * len(image.getbands())
            with _IMAGE_CACHE_LOCK:
                if key in _IMAGE_CACHE:
                    _IMAGE_CACHE_BYTES -= _IMAGE_CACHE.pop(key)[2]
                if size <= limit:
                    while _IMAGE_CACHE and _IMAGE_CACHE_BYTES + size > limit:
                        _, old = _IMAGE_CACHE.popitem(last=False)
                        _IMAGE_CACHE_BYTES -= old[2]
                        _IMAGE_STATS["evictions"] += 1
                    _IMAGE_CACHE[key] = (time.monotonic() + ttl, image.copy(), size)
                    _IMAGE_CACHE_BYTES += size
        return MediaWithBytes(image, data)

    def load_base64(self, media_type: str, data: str) -> MediaWithBytes[Image.Image]:
        return self.load_base64_timed(media_type, data, None)

    def load_base64_timed(
        self,
        media_type: str,
        data: str,
        timing: dict[str, float] | None,
        timing_lock: Any = None,
    ) -> MediaWithBytes[Image.Image]:
        decode_started = time.perf_counter()
        raw = pybase64.b64decode(data, validate=True)
        self._add_timing(
            timing,
            "media_base64_decode_ms",
            time.perf_counter() - decode_started,
            timing_lock,
        )
        return self._load_bytes_timed(raw, timing, timing_lock)

    def load_file(self, filepath: Path) -> MediaWithBytes[Image.Image]:
        return self.load_file_timed(filepath, None)

    def load_file_timed(
        self,
        filepath: Path,
        timing: dict[str, float] | None,
        timing_lock: Any = None,
    ) -> MediaWithBytes[Image.Image]:
        read_started = time.perf_counter()
        data = filepath.read_bytes()
        self._add_timing(
            timing,
            "media_file_read_ms",
            time.perf_counter() - read_started,
            timing_lock,
        )
        return self._load_bytes_timed(data, timing, timing_lock)

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "PNG",
    ) -> str:
        image = media

        with BytesIO() as buffer:
            image = self._convert_image_mode(image)
            if os.getenv("VLLM_LUMINE_IMAGE_PREPROCESSING", "raw") == "training_qwen16_png":
                from qwen_vl_utils import fetch_image
                image = fetch_image({"image": image}, image_patch_size=16)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode("utf-8")


class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):
    """Image embedding MediaIO implementation.

    Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self) -> None:
        super().__init__()

    def _load_pickled_torch(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        # Enable sparse tensor integrity checks to prevent out-of-bounds
        # writes from maliciously crafted tensors
        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.load(buffer, weights_only=True)
            return tensor.to_dense()

    def _load_numpy(self, data: bytes) -> torch.Tensor:
        with BytesIO(data) as buffer:
            return torch.from_numpy(np.load(buffer))

    def load_bytes(self, data: bytes) -> torch.Tensor:
        if data[:6] == MAGIC_NUMPY_PREFIX:
            return self._load_numpy(data)

        return self._load_pickled_torch(data)

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        if filepath.suffix == ".npy":
            return torch.from_numpy(np.load(filepath))

        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.load(filepath, weights_only=True)
            return tensor.to_dense()

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)

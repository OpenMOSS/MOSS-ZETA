r"""
Local OCR sidecar for bvalanche.

The service loads PaddleOCR once, then accepts saved screenshot paths over HTTP:

  POST /ocr
  {"imagePath": "captures/observe.png", "rois": ["subtitle_tight"], "minScore": 0.5}
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import cv2
from PIL import Image, ImageDraw
import paddle
from paddleocr import PaddleOCR, TextRecognition


DET_MODEL = "PP-OCRv6_small_det"
REC_MODEL = "PP-OCRv6_small_rec"


def image_bgr(image: Image.Image) -> np.ndarray:
    """PaddleX ndarray input is BGR; preserve every color channel."""
    return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


ROI_FRACTIONS = {
    "full_screen": (0.0, 0.0, 1.0, 1.0),
    "subtitle": (0.15, 0.76, 0.85, 0.98),
    "subtitle_tight": (0.25, 0.84, 0.75, 0.98),
    # Bottom-center current mission objective/subtitle, kept separate from
    # the legacy narrow subtitle ROI so the semantic name is stable.
    "mission_objective": (0.15, 0.76, 0.85, 0.98),
    "hud_left_bottom": (0.00, 0.70, 0.35, 1.00),
    "top_prompt": (0.00, 0.00, 0.60, 0.28),
    # Leave room for multi-line tutorial cards while staying in the upper-left
    # overlay area. OCR proposals filter most background signage.
    "tutorial_prompt": (0.03, 0.03, 0.50, 0.35),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bvalanche local OCR sidecar.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--roi", action="append", choices=sorted(ROI_FRACTIONS), default=[])
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument("--pipeline", choices=["recognizer", "full"], default="full")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--diff-gate", action="store_true")
    parser.add_argument("--diff-threshold", type=float, default=1.0)
    parser.add_argument("--fallback-full", action="store_true")
    parser.add_argument("--det-limit-side-len", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--enable-mkldnn", action="store_true")
    parser.add_argument("--single-threaded-http", action="store_true",
                        help="Keep CPU oneDNN initialization and inference on one thread.")
    parser.add_argument(
        "--model-root",
        default=os.environ.get("AGI_GTA_OCR_MODEL_ROOT", ""),
        help="Explicit PaddleX official model root (contains PP-OCRv6_small_rec).",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def as_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "items"):
        return dict(result.items())
    return {}


def roi_box(width: int, height: int, roi_name: str) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = ROI_FRACTIONS[roi_name]
    return (
        max(0, min(width, round(x1 * width))),
        max(0, min(height, round(y1 * height))),
        max(0, min(width, round(x2 * width))),
        max(0, min(height, round(y2 * height))),
    )


def numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_lines(raw_results: list[Any], min_score: float) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for result in raw_results:
        data = as_mapping(result)
        texts = jsonable(data.get("rec_texts")) or []
        scores = jsonable(data.get("rec_scores")) or []
        polys = jsonable(data.get("rec_polys")) or jsonable(data.get("dt_polys")) or []

        for index, text in enumerate(texts):
            score = numeric_or_none(scores[index] if index < len(scores) else None)
            if score is not None and score < min_score:
                continue
            lines.append(
                {
                    "text": str(text),
                    "score": score,
                    "box": jsonable(polys[index] if index < len(polys) else None),
                }
            )
    return lines


def average_score(lines: list[dict[str, Any]]) -> float | None:
    scores = [line["score"] for line in lines if line.get("score") is not None]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)


def choose_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        return "gpu"
    try:
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"


def model_directory(args: argparse.Namespace, model_name: str) -> str | None:
    """Resolve a downloaded model explicitly when the managed cache has it.

    PaddleX's model-name lookup is intentionally convenient, but it can select
    a stale per-user cache when a sidecar is started outside Node's managed
    environment.  Passing the exact artifact directory makes the model and
    character dictionary part of the deployment contract.
    """
    root = str(getattr(args, "model_root", "") or "").strip()
    if not root:
        return None
    directory = Path(root).expanduser().resolve() / model_name
    if (directory / "inference.yml").is_file() and (directory / "inference.pdiparams").is_file():
        return str(directory)
    return None


def model_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = str(getattr(args, "model_root", "") or "").strip()
    manifest: dict[str, Any] = {
        "root": str(Path(root).expanduser().resolve()) if root else "",
        "models": {},
    }
    for model_name in ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"):
        directory = model_directory(args, model_name)
        item: dict[str, Any] = {"path": directory or "", "files": {}}
        if directory:
            for filename in ("config.json", "inference.json", "inference.yml", "inference.pdiparams"):
                path = Path(directory) / filename
                if not path.is_file():
                    continue
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                item["files"][filename] = {
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            config_path = Path(directory) / "inference.yml"
            try:
                import yaml
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                characters = config.get("PostProcess", {}).get("character_dict", [])
                item["characterCount"] = len(characters) if isinstance(characters, list) else None
                item["characterSha256"] = hashlib.sha256(
                    "\n".join(str(character) for character in characters).encode("utf-8")
                ).hexdigest() if isinstance(characters, list) else ""
            except Exception as error:
                item["configError"] = str(error)
        manifest["models"][model_name] = item
    try:
        manifest["paddleVersion"] = str(getattr(paddle, "__version__", "unknown"))
        import paddleocr
        import paddlex
        manifest["paddleocrVersion"] = str(getattr(paddleocr, "__version__", "unknown"))
        manifest["paddlexVersion"] = str(getattr(paddlex, "__version__", "unknown"))
    except Exception:
        pass
    return manifest


def build_full_ocr(args: argparse.Namespace) -> PaddleOCR:
    device = choose_device(args.device)
    params = dict(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv6_small_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=args.det_limit_side_len,
        text_det_limit_type="min",
        text_recognition_batch_size=6,
        text_rec_score_thresh=0.0,
        engine="paddle_static",
        precision="fp32",
        use_tensorrt=False,
        device=device,
        cpu_threads=args.cpu_threads,
        enable_mkldnn=args.enable_mkldnn,
    )
    detection_dir = model_directory(args, "PP-OCRv6_small_det")
    recognition_dir = model_directory(args, "PP-OCRv6_small_rec")
    if detection_dir:
        params["text_detection_model_dir"] = detection_dir
    if recognition_dir:
        params["text_recognition_model_dir"] = recognition_dir
    return PaddleOCR(**params)


def build_text_recognizer(args: argparse.Namespace) -> TextRecognition:
    params = {"model_name": "PP-OCRv6_small_rec", "device": choose_device(args.device)}
    recognition_dir = model_directory(args, "PP-OCRv6_small_rec")
    if recognition_dir:
        params["model_dir"] = recognition_dir
    return TextRecognition(**params)


def image_fingerprint(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    target_width = min(128, max(1, width))
    target_height = min(72, max(1, height))
    return cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA).astype(np.float32)


def fingerprint_diff(current: np.ndarray, previous: np.ndarray | None) -> float | None:
    if previous is None or current.shape != previous.shape:
        return None
    return float(np.mean(np.abs(current - previous)))


def propose_text_lines(image: Image.Image) -> list[dict[str, Any]]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, saturation, value = cv2.split(hsv)
    mask = ((gray > 150) & ((saturation < 110) | (value > 220))).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    height, width = gray.shape
    kernel_width = max(12, round(width * 0.035))
    kernel_height = max(2, round(height * 0.018))
    dilated = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height)),
        iterations=1,
    )
    dilated = cv2.morphologyEx(
        dilated,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals: list[dict[str, Any]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if w < 20 or h < 8 or area < 200:
            continue
        density = cv2.countNonZero(mask[y : y + h, x : x + w]) / area
        if density < 0.015:
            continue

        bright_points = cv2.findNonZero(mask[y : y + h, x : x + w])
        if bright_points is not None:
            rx, ry, rw, rh = cv2.boundingRect(bright_points)
            x += rx
            y += ry
            w = rw
            h = rh

        pad_x = max(5, round(width * 0.01))
        pad_y = max(5, round(height * 0.05))
        box = [
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(width, x + w + pad_x),
            min(height, y + h + pad_y),
        ]
        proposals.append({"box": box, "density": round(float(density), 4)})

    return sorted(proposals, key=lambda item: (item["box"][1], item["box"][0]))


class OcrEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.default_rois = args.roi or ["subtitle_tight"]
        self.default_min_score = args.min_score
        self.requested_device = args.device
        self.device = choose_device(args.device)
        self.pipeline = args.pipeline
        self.diff_gate = args.diff_gate and args.pipeline == "recognizer"
        self.diff_threshold = max(0.0, float(args.diff_threshold))
        self.fallback_full = args.fallback_full
        self.model_root = str(getattr(args, "model_root", "") or "")
        self.manifest = model_manifest(args)
        self.warmup_ms = 0.0
        self.warmup_error = ""
        self.lock = threading.Lock()
        self.cache: dict[str, dict[str, Any]] = {}
        start = time.perf_counter()
        self.ocr = build_full_ocr(args) if args.pipeline == "full" or args.fallback_full else None
        self.recognizer = build_text_recognizer(args) if args.pipeline == "recognizer" else None
        self.init_ms = round((time.perf_counter() - start) * 1000.0, 2)
        if args.warmup:
            self.warmup()

    def maybe_synchronize(self) -> None:
        if self.device != "gpu":
            return
        try:
            paddle.device.synchronize()
        except Exception:
            pass

    def cached_result(self, roi_name: str, crop: Image.Image) -> tuple[dict[str, Any] | None, float | None, float]:
        started = time.perf_counter()
        fingerprint = image_fingerprint(crop)
        previous = self.cache.get(roi_name)
        diff_score = fingerprint_diff(fingerprint, previous.get("fingerprint") if previous else None)
        diff_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if previous and diff_score is not None and diff_score <= self.diff_threshold:
            result = copy.deepcopy(previous["result"])
            result["cached"] = True
            result["diffScore"] = round(diff_score, 4)
            result["diffMs"] = diff_ms
            result["latencyMs"] = diff_ms
            return result, diff_score, diff_ms
        return None, diff_score, diff_ms

    def store_cache(self, roi_name: str, crop: Image.Image, result: dict[str, Any]) -> None:
        cached = copy.deepcopy(result)
        cached["cached"] = False
        self.cache[roi_name] = {
            "fingerprint": image_fingerprint(crop),
            "result": cached,
        }

    def recognize_full_roi(self, roi_name: str, box: tuple[int, int, int, int], crop: Image.Image, min_score: float, *, fallback: bool = False) -> dict[str, Any]:
        if self.ocr is None:
            raise RuntimeError("full OCR pipeline is not initialized")
        start = time.perf_counter()
        with self.lock:
            raw_results = self.ocr.predict(image_bgr(crop))
            self.maybe_synchronize()
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        lines = extract_lines(raw_results, min_score=min_score)
        return {
            "roi": roi_name,
            "box": list(box),
            "text": " ".join(line["text"] for line in lines).strip(),
            "score": average_score(lines),
            "latencyMs": latency_ms,
            "pipeline": "full",
            "cached": False,
            "fallback": fallback,
            "proposalCount": None,
            "lines": lines,
        }

    def recognize_with_recognizer(self, roi_name: str, box: tuple[int, int, int, int], crop: Image.Image, min_score: float) -> dict[str, Any]:
        if self.recognizer is None:
            raise RuntimeError("recognizer OCR pipeline is not initialized")

        started = time.perf_counter()
        proposal_started = time.perf_counter()
        proposals = propose_text_lines(crop)
        proposal_ms = round((time.perf_counter() - proposal_started) * 1000.0, 2)
        lines: list[dict[str, Any]] = []
        recognition_ms = 0.0

        if proposals:
            arrays = [
                image_bgr(crop.crop(tuple(proposal["box"])))
                for proposal in proposals
            ]
            recognition_started = time.perf_counter()
            with self.lock:
                raw_results = self.recognizer.predict(arrays)
                self.maybe_synchronize()
            recognition_ms = round((time.perf_counter() - recognition_started) * 1000.0, 2)

            for proposal, result in zip(proposals, raw_results):
                data = as_mapping(result)
                text = str(data.get("rec_text") or "").strip()
                score = numeric_or_none(data.get("rec_score"))
                if not text:
                    continue
                if score is not None and score < min_score:
                    continue
                lines.append(
                    {
                        "text": text,
                        "score": score,
                        "box": proposal["box"],
                        "proposalDensity": proposal["density"],
                    }
                )

        if not lines and self.fallback_full:
            result = self.recognize_full_roi(roi_name, box, crop, min_score, fallback=True)
            result["proposalMs"] = proposal_ms
            result["recognitionMs"] = recognition_ms
            result["proposalCount"] = len(proposals)
            result["fallbackReason"] = "recognizer_empty"
            return result

        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return {
            "roi": roi_name,
            "box": list(box),
            "text": " ".join(line["text"] for line in lines).strip(),
            "score": average_score(lines),
            "latencyMs": latency_ms,
            "pipeline": "recognizer",
            "cached": False,
            "fallback": False,
            "proposalMs": proposal_ms,
            "recognitionMs": recognition_ms,
            "proposalCount": len(proposals),
            "lines": lines,
        }

    def recognize_roi(self, roi_name: str, box: tuple[int, int, int, int], crop: Image.Image, min_score: float, *, use_cache: bool = True) -> dict[str, Any]:
        diff_score = None
        diff_ms = 0.0
        if use_cache and self.diff_gate:
            cached, diff_score, diff_ms = self.cached_result(roi_name, crop)
            if cached is not None:
                return cached

        if self.pipeline == "full":
            result = self.recognize_full_roi(roi_name, box, crop, min_score)
        else:
            result = self.recognize_with_recognizer(roi_name, box, crop, min_score)

        result["diffScore"] = round(diff_score, 4) if diff_score is not None else None
        result["diffMs"] = diff_ms
        if use_cache and self.diff_gate:
            self.store_cache(roi_name, crop, result)
        return result

    def recognize_image(self, image: Image.Image, rois: list[str], min_score: float, *, use_cache: bool = True) -> list[dict[str, Any]]:
        width, height = image.size
        if self.pipeline == "full":
            # Match the Linux production runner: detect and recognize one full
            # frame, then route complete lines to the existing ROI HTTP schema.
            # Never crop letters before detection or reuse approximate grayscale
            # fingerprints; a changed colored objective must get fresh OCR.
            full = self.recognize_full_roi("full_screen", (0, 0, width, height), image, min_score)
            results = []
            for roi_name in rois:
                box = roi_box(width, height, roi_name)
                lines = []
                for line in full["lines"]:
                    polygon = line.get("box")
                    if not polygon:
                        continue
                    xs, ys = zip(*polygon)
                    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                    if box[0] <= cx < box[2] and box[1] <= cy < box[3]:
                        lines.append({**line, "box": [[x - box[0], y - box[1]] for x, y in polygon]})
                results.append({
                    **full, "roi": roi_name, "box": list(box), "lines": lines,
                    "text": " ".join(line["text"] for line in lines).strip(),
                    "score": average_score(lines), "inputScope": "full_screen",
                    "diffScore": None, "diffMs": 0.0,
                })
            return results

        return [self.recognize_roi(roi_name, roi_box(width, height, roi_name),
                                  image.crop(roi_box(width, height, roi_name)),
                                  min_score, use_cache=use_cache) for roi_name in rois]

    def recognize(
        self,
        image_path: Path | None,
        image_data: str | None,
        rois: list[str],
        min_score: float,
    ) -> list[dict[str, Any]]:
        if image_data:
            encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
            image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
        elif image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("imagePath or imageData is required")
        return self.recognize_image(image, rois, min_score)

    def warmup(self) -> None:
        image = Image.new("RGB", (1280, 720), (20, 24, 28))
        draw = ImageDraw.Draw(image)
        draw.rectangle((320, 610, 960, 680), fill=(0, 0, 0))
        draw.text((390, 632), "Warm up OCR for GTA subtitle.", fill=(255, 255, 255))
        start = time.perf_counter()
        try:
            self.recognize_image(image, self.default_rois, self.default_min_score, use_cache=False)
            self.warmup_ms = round((time.perf_counter() - start) * 1000.0, 2)
        except Exception as error:
            self.warmup_error = str(error)
            self.warmup_ms = round((time.perf_counter() - start) * 1000.0, 2)


def valid_rois(value: Any, fallback: list[str]) -> list[str]:
    raw = value if isinstance(value, list) else fallback
    rois = [str(item).strip() for item in raw if str(item).strip() in ROI_FRACTIONS]
    return rois or fallback


def min_score_from(value: Any, fallback: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, score))


def make_handler(engine: OcrEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "BvalancheOCR/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            sys.stdout.write("%s - %s\n" % (self.address_string(), format % args))
            sys.stdout.flush()

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] != "/health":
                self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            self.send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "initialized": True,
                    "ocrInitMs": engine.init_ms,
                    "warmupMs": engine.warmup_ms,
                    "warmupError": engine.warmup_error,
                    "requestedDevice": engine.requested_device,
                    "device": engine.device,
                    "pipeline": engine.pipeline,
                    "fullFrameLines": True,
                    "diffGate": engine.diff_gate,
                    "diffThreshold": engine.diff_threshold,
                    "fallbackFull": engine.fallback_full,
                    "defaultRois": engine.default_rois,
                    "modelRoot": engine.model_root,
                    "modelManifest": engine.manifest,
                },
            )

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/ocr":
                self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found", "results": []})
                return

            started = time.perf_counter()
            try:
                length = int(self.headers.get("content-length") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "invalid JSON", "latencyMs": 0, "results": []},
                )
                return

            image_path_value = str(payload.get("imagePath") or "").strip()
            image_path = Path(image_path_value) if image_path_value else None
            image_data = str(payload.get("imageData") or "").strip() or None
            rois = valid_rois(payload.get("rois"), engine.default_rois)
            min_score = min_score_from(payload.get("minScore"), engine.default_min_score)

            if not image_data and (not image_path or not image_path.exists()):
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "ok": False,
                        "error": f"image not found: {image_path_value}",
                        "latencyMs": round((time.perf_counter() - started) * 1000.0, 2),
                        "rois": rois,
                        "results": [],
                    },
                )
                return

            try:
                results = engine.recognize(
                    image_path=image_path,
                    image_data=image_data,
                    rois=rois,
                    min_score=min_score,
                )
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "error": "",
                        "latencyMs": round((time.perf_counter() - started) * 1000.0, 2),
                        "rois": rois,
                        "results": results,
                    },
                )
            except Exception as error:
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "ok": False,
                        "error": str(error),
                        "latencyMs": round((time.perf_counter() - started) * 1000.0, 2),
                        "rois": rois,
                        "results": [],
                    },
                )

    return Handler


def main() -> int:
    args = parse_args()
    engine = OcrEngine(args)
    server_type = HTTPServer if args.single_threaded_http else ThreadingHTTPServer
    server = server_type((args.host, args.port), make_handler(engine))
    print(
        json.dumps(
            {
                "event": "ocr_service_ready",
                "host": args.host,
                "port": args.port,
                "ocrInitMs": engine.init_ms,
                "warmupMs": engine.warmup_ms,
                "warmupError": engine.warmup_error,
                "requestedDevice": engine.requested_device,
                "device": engine.device,
                "pipeline": engine.pipeline,
                "diffGate": engine.diff_gate,
                "diffThreshold": engine.diff_threshold,
                "fallbackFull": engine.fallback_full,
                "defaultRois": engine.default_rois,
                "modelRoot": engine.model_root,
                "modelManifest": engine.manifest,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

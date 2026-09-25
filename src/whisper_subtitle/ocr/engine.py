# """Paddle PP-OCR v5 engine: detection + recognition via onnxruntime.

# Given a BGR frame and a timestamp, returns a list of OcrDetection
# objects, one per recognized text box.

# The engine is unaware of video structure — it sees one frame at a time.
# The frame_time parameter is passed through to each OcrDetection so the
# caller doesn't have to rebuild them.
# """

# import logging
# import math
# from collections.abc import Sequence
# from pathlib import Path

# import cv2
# import numpy as np
# import onnxruntime as ort

# from whisper_subtitle.exceptions import OcrError
# from whisper_subtitle.models import OcrDetection

# log = logging.getLogger(__name__)


# class PaddleOcrEngine:
#     """Paddle PP-OCR v5 detection + recognition."""

#     def __init__(
#         self,
#         det_model: Path,
#         rec_model: Path,
#         dict_file: Path,
#         *,
#         providers: Sequence[str] | None = None,
#         max_side: int = 1280,
#         stride: int = 32,
#         det_threshold: float = 0.20,
#         unclip_kernel_size: int = 5,
#         rec_margin_ratio: float = 0.22,
#         min_box_w: int = 5,
#         min_box_h: int = 5,
#         min_box_area: int = 100,
#         rec_threshold: float = 0.5,
#     ) -> None:
#         if providers is None:
#             providers = ["CPUExecutionProvider"]
#         providers = list(providers)

#         self.max_side = max_side
#         self.stride = stride
#         self.det_threshold = det_threshold
#         self.unclip_kernel_size = unclip_kernel_size
#         self.rec_margin_ratio = rec_margin_ratio
#         self.min_box_w = min_box_w
#         self.min_box_h = min_box_h
#         self.min_box_area = min_box_area
#         self.rec_threshold = rec_threshold

#         try:
#             self.det_session = ort.InferenceSession(str(det_model), providers=providers)
#             self.rec_session = ort.InferenceSession(str(rec_model), providers=providers)
#         except Exception as exc:
#             raise OcrError(f"Failed to load ONNX models: {exc}") from exc

#         self.det_input_name = self.det_session.get_inputs()[0].name
#         self.rec_input_name = self.rec_session.get_inputs()[0].name

#         log.debug("Detection providers: %s", self.det_session.get_providers())
#         log.debug("Recognition providers: %s", self.rec_session.get_providers())

#         try:
#             with open(dict_file, "r", encoding="utf-8") as f:
#                 chars = [line.rstrip("\n\r") for line in f]
#         except OSError as exc:
#             raise OcrError(f"Could not read dict file {dict_file}: {exc}") from exc

#         # PaddleOCR CTC reserves index 0 for the blank token.
#         self.characters = [""] + chars
#         log.debug("Dictionary characters: %d", len(chars))

#         # PP-OCR det normalization.
#         self.det_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#         self.det_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#         self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------

#     def process_frame(self, frame: np.ndarray, frame_time: float = 0.0) -> list[OcrDetection]:
#         """Detect and recognize all text boxes in one BGR frame."""
#         if frame is None:
#             raise OcrError("frame is None")

#         boxes = self._detect_boxes(frame)
#         return self._recognize_boxes(frame, boxes, frame_time)

#     # ------------------------------------------------------------------
#     # Detection
#     # ------------------------------------------------------------------

#     def _prepare_det_input(self, img: np.ndarray) -> tuple[np.ndarray, float, float]:
#         h, w = img.shape[:2]
#         scale = min(1.0, self.max_side / max(h, w))
#         new_w = int(round(w * scale))
#         new_h = int(round(h * scale))
#         new_w = max(self.stride, (new_w + self.stride - 1) // self.stride * self.stride)
#         new_h = max(self.stride, (new_h + self.stride - 1) // self.stride * self.stride)
#         resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#         return resized, new_w / w, new_h / h

#     def _enhance_local_contrast(self, img: np.ndarray) -> np.ndarray:
#         lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
#         l = self.clahe.apply(l)
#         return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

#     def _box_angle(self, contour) -> float:
#         rect = cv2.minAreaRect(contour)
#         box_pts = cv2.boxPoints(rect)
#         edges = [box_pts[1] - box_pts[0], box_pts[2] - box_pts[1]]
#         lengths = [np.linalg.norm(e) for e in edges]
#         long_edge = edges[0] if lengths[0] >= lengths[1] else edges[1]
#         angle = math.degrees(math.atan2(long_edge[1], long_edge[0])) % 180
#         if angle > 90:
#             angle -= 180
#         if angle > 45:
#             angle -= 90
#         elif angle < -45:
#             angle += 90
#         return angle

#     def _detect_boxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
#         det_img, scale_x, scale_y = self._prepare_det_input(frame)
#         det_img = self._enhance_local_contrast(det_img)
#         det_input = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
#         det_input = det_input.astype(np.float32) / 255.0
#         det_input = (det_input - self.det_mean) / self.det_std
#         det_input = np.transpose(det_input, (2, 0, 1))
#         det_input = np.expand_dims(det_input, axis=0).astype(np.float32)

#         det_output = self.det_session.run(None, {self.det_input_name: det_input})[0]
#         prob = det_output[0, 0]

#         bitmap = (prob > self.det_threshold).astype(np.uint8) * 255
#         kernel = cv2.getStructuringElement(
#             cv2.MORPH_RECT, (self.unclip_kernel_size, self.unclip_kernel_size)
#         )
#         bitmap = cv2.dilate(bitmap, kernel, iterations=1)
#         contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#         boxes = []
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             if w < self.min_box_w or h < self.min_box_h:
#                 continue
#             if w * h < self.min_box_area:
#                 continue
#             boxes.append((x, y, w, h, self._box_angle(contour)))

#         boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

#         scaled = []
#         for x, y, w, h, angle in boxes:
#             scaled.append((
#                 int(x / scale_x),
#                 int(y / scale_y),
#                 int((x + w) / scale_x),
#                 int((y + h) / scale_y),
#                 angle,
#             ))
#         return scaled

#     # ------------------------------------------------------------------
#     # Recognition
#     # ------------------------------------------------------------------

#     def _prepare_rec_crop(self, crop: np.ndarray) -> np.ndarray | None:
#         h, w = crop.shape[:2]
#         if h == 0 or w == 0:
#             return None
#         target_h = 48
#         target_w = max(1, int(w * target_h / h))
#         crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#         crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#         crop = crop.astype(np.float32) / 255.0
#         crop = (crop - 0.5) / 0.5
#         crop = np.transpose(crop, (2, 0, 1))
#         return np.expand_dims(crop, axis=0).astype(np.float32)

#     def _run_recognition(self, crop: np.ndarray) -> np.ndarray | None:
#         prep = self._prepare_rec_crop(crop)
#         if prep is None:
#             return None
#         output = self.rec_session.run(None, {self.rec_input_name: prep})[0]
#         if output.ndim != 3:
#             raise OcrError(f"Unexpected recognition output shape: {output.shape}")
#         return output[0]

#     def _decode_ctc(self, probs: np.ndarray) -> tuple[str, float]:
#         indices = np.argmax(probs, axis=1)
#         text: list[str] = []
#         confidences: list[float] = []
#         previous = -1
#         for t, idx in enumerate(indices):
#             if idx == 0:
#                 previous = idx
#                 continue
#             if idx == previous:
#                 continue
#             if idx < len(self.characters):
#                 text.append(self.characters[idx])
#                 confidences.append(float(probs[t, idx]))
#             previous = idx
#         confidence = float(np.mean(confidences)) if confidences else 0.0
#         return "".join(text), confidence

#     def _recognize_boxes(
#         self,
#         frame: np.ndarray,
#         boxes: list[tuple[int, int, int, int, float]],
#         frame_time: float,
#     ) -> list[OcrDetection]:
#         h_img, w_img = frame.shape[:2]
#         results: list[OcrDetection] = []

#         for x1, y1, x2, y2, angle in boxes:
#             box_h = y2 - y1
#             margin = max(4, int(box_h * self.rec_margin_ratio))
#             mx1 = max(0, x1 - margin)
#             my1 = max(0, y1 - margin)
#             mx2 = min(w_img, x2 + margin)
#             my2 = min(h_img, y2 + margin)

#             crop = frame[my1:my2, mx1:mx2]
#             if crop.size == 0:
#                 continue

#             try:
#                 probs = self._run_recognition(crop)
#             except Exception as exc:
#                 log.debug("Recognition failed on box %s: %s", (x1, y1, x2, y2), exc)
#                 continue

#             if probs is None:
#                 continue

#             text, confidence = self._decode_ctc(probs)
#             if confidence < self.rec_threshold or not text:
#                 continue

#             results.append(OcrDetection(
#                 frame_time=frame_time,
#                 box=(x1, y1, x2, y2),
#                 text=text,
#                 angle=angle,
#                 confidence=confidence,
#             ))

#         return results
"""Paddle PP-OCR v5 engine: detection + recognition via onnxruntime.

Given a BGR frame and a timestamp, returns a list of OcrDetection
objects, one per recognized text box.

The engine is unaware of video structure — it sees one frame at a time.
The frame_time parameter is passed through to each OcrDetection so the
caller doesn't have to rebuild them.

Timing
------
Pass ``enable_timing=True`` to collect per-stage timings. While enabled,
the engine accumulates ``det_*``, ``rec_*`` and ``frame_total`` counters
that can be inspected with :meth:`timing_report` / :meth:`log_timing_report`
and cleared with :meth:`reset_timing`. Per-frame totals are also logged
at DEBUG level. When disabled (the default) the timing hooks short-circuit
and cost nothing.
"""

import logging
import math
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from whisper_subtitle.exceptions import OcrError
from whisper_subtitle.models import OcrDetection

log = logging.getLogger(__name__)


class _TimingStats:
    """Accumulates named timings across many calls."""

    def __init__(self) -> None:
        self._calls: dict[str, int] = defaultdict(int)
        self._totals: dict[str, float] = defaultdict(float)

    def add(self, name: str, seconds: float) -> None:
        self._calls[name] += 1
        self._totals[name] += seconds

    def report(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "calls": self._calls[name],
                "total_ms": self._totals[name] * 1000.0,
                "avg_ms": self._totals[name] * 1000.0 / self._calls[name],
            }
            for name in self._totals
        }

    def reset(self) -> None:
        self._calls.clear()
        self._totals.clear()

    def format(self) -> str:
        rows = []
        for name, s in sorted(self.report().items()):
            rows.append(
                f"  {name:<20} "
                f"calls={s['calls']:<7d} "
                f"total={s['total_ms']:>9.1f} ms  "
                f"avg={s['avg_ms']:>8.2f} ms"
            )
        return "\n".join(rows) if rows else "  (no samples)"


class PaddleOcrEngine:
    """Paddle PP-OCR v5 detection + recognition."""

    def __init__(
        self,
        det_model: Path,
        rec_model: Path,
        dict_file: Path,
        *,
        providers: Sequence[str] | None = None,
        max_side: int = 1280,
        stride: int = 32,
        det_threshold: float = 0.20,
        unclip_kernel_size: int = 5,
        rec_margin_ratio: float = 0.22,
        min_box_w: int = 5,
        min_box_h: int = 5,
        min_box_area: int = 100,
        rec_threshold: float = 0.5,
        enable_timing: bool = False,
    ) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]
        providers = list(providers)

        self.max_side = max_side
        self.stride = stride
        self.det_threshold = det_threshold
        self.unclip_kernel_size = unclip_kernel_size
        self.rec_margin_ratio = rec_margin_ratio
        self.min_box_w = min_box_w
        self.min_box_h = min_box_h
        self.min_box_area = min_box_area
        self.rec_threshold = rec_threshold

        # ---- timing ----
        self.enable_timing = enable_timing
        self.timing: _TimingStats | None = _TimingStats() if enable_timing else None

        try:
            self.det_session = ort.InferenceSession(str(det_model), providers=providers)
            self.rec_session = ort.InferenceSession(str(rec_model), providers=providers)
        except Exception as exc:
            raise OcrError(f"Failed to load ONNX models: {exc}") from exc

        self.det_input_name = self.det_session.get_inputs()[0].name
        self.rec_input_name = self.rec_session.get_inputs()[0].name

        log.debug("Detection providers: %s", self.det_session.get_providers())
        log.debug("Recognition providers: %s", self.rec_session.get_providers())

        try:
            with open(dict_file, "r", encoding="utf-8") as f:
                chars = [line.rstrip("\n\r") for line in f]
        except OSError as exc:
            raise OcrError(f"Could not read dict file {dict_file}: {exc}") from exc

        # PaddleOCR CTC reserves index 0 for the blank token.
        self.characters = [""] + chars
        log.debug("Dictionary characters: %d", len(chars))

        # PP-OCR det normalization.
        self.det_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.det_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        """Measure a block. No-op (aside from a fast branch) when disabled."""
        if self.timing is None:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timing.add(name, time.perf_counter() - t0)

    def reset_timing(self) -> None:
        """Clear accumulated timing counters."""
        if self.timing is not None:
            self.timing.reset()

    def timing_report(self) -> dict[str, dict[str, float]]:
        """Return ``{stage: {calls, total_ms, avg_ms}}``. Empty if disabled."""
        return self.timing.report() if self.timing is not None else {}

    def log_timing_report(self, level: int = logging.INFO) -> None:
        """Log a formatted timing summary."""
        if self.timing is None:
            log.warning("Timing is disabled; nothing to report.")
            return
        log.log(level, "OCR timing report:\n%s", self.timing.format())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_time: float = 0.0) -> list[OcrDetection]:
        """Detect and recognize all text boxes in one BGR frame."""
        if frame is None:
            raise OcrError("frame is None")

        if self.timing is None:
            boxes = self._detect_boxes(frame)
            return self._recognize_boxes(frame, boxes, frame_time)

        t0 = time.perf_counter()
        boxes = self._detect_boxes(frame)
        results = self._recognize_boxes(frame, boxes, frame_time)
        elapsed = time.perf_counter() - t0
        self.timing.add("frame_total", elapsed)
        log.debug(
            "frame_time=%.3fs boxes=%d kept=%d total=%.1fms",
            frame_time, len(boxes), len(results), elapsed * 1000.0,
        )
        return results

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _prepare_det_input(self, img: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = img.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        new_w = max(self.stride, (new_w + self.stride - 1) // self.stride * self.stride)
        new_h = max(self.stride, (new_h + self.stride - 1) // self.stride * self.stride)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, new_w / w, new_h / h

    def _enhance_local_contrast(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _box_angle(self, contour) -> float:
        rect = cv2.minAreaRect(contour)
        box_pts = cv2.boxPoints(rect)
        edges = [box_pts[1] - box_pts[0], box_pts[2] - box_pts[1]]
        lengths = [np.linalg.norm(e) for e in edges]
        long_edge = edges[0] if lengths[0] >= lengths[1] else edges[1]
        angle = math.degrees(math.atan2(long_edge[1], long_edge[0])) % 180
        if angle > 90:
            angle -= 180
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        return angle

    def _detect_boxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        with self._timed("det_total"):
            with self._timed("det_preprocess"):
                det_img, scale_x, scale_y = self._prepare_det_input(frame)
                det_img = self._enhance_local_contrast(det_img)
                det_input = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
                det_input = det_input.astype(np.float32) / 255.0
                det_input = (det_input - self.det_mean) / self.det_std
                det_input = np.transpose(det_input, (2, 0, 1))
                det_input = np.expand_dims(det_input, axis=0).astype(np.float32)

            with self._timed("det_session"):
                det_output = self.det_session.run(None, {self.det_input_name: det_input})[0]

            with self._timed("det_postprocess"):
                prob = det_output[0, 0]

                bitmap = (prob > self.det_threshold).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (self.unclip_kernel_size, self.unclip_kernel_size)
                )
                bitmap = cv2.dilate(bitmap, kernel, iterations=1)
                contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                boxes = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w < self.min_box_w or h < self.min_box_h:
                        continue
                    if w * h < self.min_box_area:
                        continue
                    boxes.append((x, y, w, h, self._box_angle(contour)))

                boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

                scaled = []
                for x, y, w, h, angle in boxes:
                    scaled.append((
                        int(x / scale_x),
                        int(y / scale_y),
                        int((x + w) / scale_x),
                        int((y + h) / scale_y),
                        angle,
                    ))
        return scaled

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def _prepare_rec_crop(self, crop: np.ndarray) -> np.ndarray | None:
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return None
        target_h = 48
        target_w = max(1, int(w * target_h / h))
        crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - 0.5) / 0.5
        crop = np.transpose(crop, (2, 0, 1))
        return np.expand_dims(crop, axis=0).astype(np.float32)

    def _run_recognition(self, crop: np.ndarray) -> np.ndarray | None:
        with self._timed("rec_prepare"):
            prep = self._prepare_rec_crop(crop)
            if prep is None:
                return None
        with self._timed("rec_session"):
            output = self.rec_session.run(None, {self.rec_input_name: prep})[0]
        if output.ndim != 3:
            raise OcrError(f"Unexpected recognition output shape: {output.shape}")
        return output[0]

    def _decode_ctc(self, probs: np.ndarray) -> tuple[str, float]:
        indices = np.argmax(probs, axis=1)
        text: list[str] = []
        confidences: list[float] = []
        previous = -1
        for t, idx in enumerate(indices):
            if idx == 0:
                previous = idx
                continue
            if idx == previous:
                continue
            if idx < len(self.characters):
                text.append(self.characters[idx])
                confidences.append(float(probs[t, idx]))
            previous = idx
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return "".join(text), confidence

    def _recognize_boxes(
        self,
        frame: np.ndarray,
        boxes: list[tuple[int, int, int, int, float]],
        frame_time: float,
    ) -> list[OcrDetection]:
        h_img, w_img = frame.shape[:2]
        results: list[OcrDetection] = []

        with self._timed("rec_total"):
            for x1, y1, x2, y2, angle in boxes:
                box_h = y2 - y1
                margin = max(4, int(box_h * self.rec_margin_ratio))
                mx1 = max(0, x1 - margin)
                my1 = max(0, y1 - margin)
                mx2 = min(w_img, x2 + margin)
                my2 = min(h_img, y2 + margin)

                with self._timed("rec_crop"):
                    crop = frame[my1:my2, mx1:mx2]
                if crop.size == 0:
                    continue

                try:
                    probs = self._run_recognition(crop)
                except Exception as exc:
                    log.debug("Recognition failed on box %s: %s", (x1, y1, x2, y2), exc)
                    continue

                if probs is None:
                    continue

                with self._timed("rec_decode"):
                    text, confidence = self._decode_ctc(probs)
                if confidence < self.rec_threshold or not text:
                    continue

                results.append(OcrDetection(
                    frame_time=frame_time,
                    box=(x1, y1, x2, y2),
                    text=text,
                    angle=angle,
                    confidence=confidence,
                ))

        return results
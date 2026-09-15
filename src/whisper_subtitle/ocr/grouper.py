"""Group frame-level OCR detections into time-ranged events."""

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from whisper_subtitle.models import OcrDetection, OcrEvent

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase and strip whitespace for comparison purposes only."""
    return _WS_RE.sub("", text.lower())


def _text_similarity(a: str, b: str) -> float:
    """Similarity between two strings, in [0, 1]. 1 = identical."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _OpenEvent:
    """A group of matching detections still being accumulated."""

    first: OcrDetection
    last_time: float
    detections: list[OcrDetection] = field(default_factory=list)

    def add(self, det: OcrDetection) -> None:
        self.detections.append(det)
        self.last_time = det.frame_time

    def close(self) -> OcrEvent:
        all_dets = [self.first, *self.detections]
        confidences = [d.confidence for d in all_dets]
        return OcrEvent(
            start=all_dets[0].frame_time,
            end=all_dets[-1].frame_time,
            box=self.first.box,
            text=self.first.text,
            frame_count=len(all_dets),
            confidence=sum(confidences) / len(confidences),
        )


def group_detections(
    detections: list[OcrDetection],
    *,
    text_similarity: float = 0.8,
    box_iou: float = 0.5,
    max_gap_seconds: float = 1.5,
) -> list[OcrEvent]:
    """Group frame-level detections into time-ranged events.

    Two detections are merged into the same event if:
      - their text is similar (>= text_similarity), AND
      - their boxes overlap (IoU >= box_iou), AND
      - the time gap since the event's last detection <= max_gap_seconds

    Detections are processed in chronological order regardless of the
    order they arrive in. Output is sorted by event start time.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: (d.frame_time, d.box[0]))

    open_events: list[_OpenEvent] = []
    closed_events: list[OcrEvent] = []

    for det in sorted_dets:
        match = _find_match(det, open_events, text_similarity, box_iou)
        if match is not None:
            match.add(det)
        else:
            open_events.append(_OpenEvent(first=det, last_time=det.frame_time))

        # Close events whose most recent detection is now too far in the past.
        still_open: list[_OpenEvent] = []
        for ev in open_events:
            if det.frame_time - ev.last_time > max_gap_seconds:
                closed_events.append(ev.close())
            else:
                still_open.append(ev)
        open_events = still_open

    for ev in open_events:
        closed_events.append(ev.close())

    closed_events.sort(key=lambda e: e.start)
    log.debug(
        "Grouped %d detections into %d events",
        len(detections), len(closed_events),
    )
    return closed_events


def _find_match(
    det: OcrDetection,
    open_events: list[_OpenEvent],
    text_similarity: float,
    box_iou: float,
) -> _OpenEvent | None:
    """Find the open event this detection best belongs to, if any."""
    best: _OpenEvent | None = None
    best_score = 0.0

    for ev in open_events:
        sim = _text_similarity(det.text, ev.first.text)
        if sim < text_similarity:
            continue
        if _iou(det.box, ev.first.box) < box_iou:
            continue
        if sim > best_score:
            best_score = sim
            best = ev

    return best

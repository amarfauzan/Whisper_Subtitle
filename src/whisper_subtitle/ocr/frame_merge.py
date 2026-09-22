"""Per-frame spatial merge of OCR detections.

For each frame, pick the detection with the longest text as the anchor.
Find every detection reachable by transitive adjacency from the anchor
(side-by-side or stacked). Merge them into one detection:
    - same row → join with space
    - different rows → join with newline, top-to-bottom

Everything else in the frame is dropped. Result: one detection per frame.

Runs before the grouper so that multi-box subtitles collapse to a single
detection per frame, which the grouper then tracks cleanly across time.
"""

import logging
from collections import defaultdict

from whisper_subtitle.models import OcrDetection

log = logging.getLogger(__name__)


# Gap tolerance as a fraction of the shorter box's height.
# 1.0 handles OCR jitter (a two-line subtitle's gap can vary 20-30px
# between frames even when the text is identical).
_GAP_TOLERANCE = 1.0

# Minimum overlap ratios to consider two boxes "adjacent".
_SAME_ROW_Y_OVERLAP = 0.3
_STACKED_X_OVERLAP = 0.5

# Two detections on the same row if their y-centers differ by less than
# this fraction of the shorter height.
_ROW_Y_TOLERANCE = 0.5


def merge_frame_detections(
    detections: list[OcrDetection],
) -> list[OcrDetection]:
    """Reduce a list of detections to one per frame timestamp."""
    by_frame: dict[float, list[OcrDetection]] = defaultdict(list)
    for d in detections:
        by_frame[d.frame_time].append(d)

    merged: list[OcrDetection] = []
    dropped = 0

    for frame_time in sorted(by_frame):
        frame_dets = by_frame[frame_time]
        anchor_idx = max(
            range(len(frame_dets)),
            key=lambda i: len(frame_dets[i].text),
        )
        neighbors = _find_neighbors(anchor_idx, frame_dets)
        group = [frame_dets[anchor_idx]] + [frame_dets[i] for i in neighbors]

        dropped += len(frame_dets) - len(group)
        merged.append(_merge_group(group))

    if dropped:
        log.debug(
            "Frame merge: %d detections → %d (dropped %d)",
            len(detections), len(merged), dropped,
        )
    return merged


# ---------------------------------------------------------------- internals


def _is_adjacent(a: OcrDetection, b: OcrDetection) -> bool:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1
    ref_h = min(ah, bh)
    if ref_h <= 0:
        return False

    y_overlap = min(ay2, by2) - max(ay1, by1)
    x_overlap = min(ax2, bx2) - max(ax1, bx1)

    # Same row: overlapping vertically, small horizontal gap
    if y_overlap >= _SAME_ROW_Y_OVERLAP * ref_h:
        gap = max(ax1 - bx2, bx1 - ax2, 0)
        if gap <= _GAP_TOLERANCE * ref_h:
            return True

    # Stacked: overlapping horizontally, small vertical gap
    if x_overlap >= _STACKED_X_OVERLAP * min(aw, bw):
        gap = max(ay1 - by2, by1 - ay2, 0)
        if gap <= _GAP_TOLERANCE * ref_h:
            return True

    return False


def _find_neighbors(
    anchor_idx: int,
    frame_dets: list[OcrDetection],
) -> set[int]:
    """Transitive adjacency from the anchor within a single frame."""
    neighbors: set[int] = set()
    frontier = {anchor_idx}

    while frontier:
        new: set[int] = set()
        for i in range(len(frame_dets)):
            if i in neighbors or i == anchor_idx:
                continue
            for j in frontier:
                if _is_adjacent(frame_dets[i], frame_dets[j]):
                    new.add(i)
                    break
        if not new:
            break
        neighbors |= new
        frontier = new

    return neighbors


def _merge_group(group: list[OcrDetection]) -> OcrDetection:
    """Combine a group of same-frame detections into one."""
    group = sorted(
        group,
        key=lambda d: ((d.box[1] + d.box[3]) / 2, (d.box[0] + d.box[2]) / 2),
    )

    # Group into rows by y-center proximity
    rows: list[list[OcrDetection]] = []
    for d in group:
        y_c = (d.box[1] + d.box[3]) / 2
        h = d.box[3] - d.box[1]
        placed = False
        for row in rows:
            ref = row[0]
            ref_y_c = (ref.box[1] + ref.box[3]) / 2
            ref_h = ref.box[3] - ref.box[1]
            if abs(y_c - ref_y_c) <= _ROW_Y_TOLERANCE * min(h, ref_h):
                row.append(d)
                placed = True
                break
        if not placed:
            rows.append([d])

    # Sort each row by x, join cells with space, rows with newline
    lines: list[str] = []
    for row in rows:
        row.sort(key=lambda d: (d.box[0] + d.box[2]) / 2)
        lines.append(" ".join(d.text for d in row))
    text = "\n".join(lines)

    x1 = min(d.box[0] for d in group)
    y1 = min(d.box[1] for d in group)
    x2 = max(d.box[2] for d in group)
    y2 = max(d.box[3] for d in group)
    confidence = sum(d.confidence for d in group) / len(group)
    frame_time = group[0].frame_time
    angle = group[0].angle

    return OcrDetection(
        frame_time=frame_time,
        box=(x1, y1, x2, y2),
        text=text,
        angle=angle,
        confidence=confidence,
    )
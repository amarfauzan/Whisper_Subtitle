"""Filter grouped OCR events by duration, box size, and text content.

The filter's job is to drop events that are not subtitles: logos,
signs, labels, decoration, OCR artifacts. It does NOT try to fix bad
grouping — that's the grouper's job. It just decides keep/drop per event.

Rules are applied in order from cheapest to most expensive. Every drop
is categorized so the log can show a summary that helps tuning.
"""

import logging
import re

from whisper_subtitle.config import OcrFilterConfig
from whisper_subtitle.models import OcrEvent, Subtitle

log = logging.getLogger(__name__)


def filter_events(
    events: list[OcrEvent],
    cfg: OcrFilterConfig,
    *,
    frame_height: int,
) -> list[OcrEvent]:
    """Return the subset of events that pass every filter rule.

    Rules, in order:
      1. duration in [min_duration, max_duration]
      2. box height / frame_height in
         [min_box_height_ratio, max_box_height_ratio]
      3. len(text.strip()) >= min_text_length
      4. text (case-insensitive) not in blocklist_texts
      5. text does not fully match any blocklist_pattern
    """
    if frame_height <= 0:
        raise ValueError(f"frame_height must be positive, got {frame_height}")

    compiled_patterns = [re.compile(p) for p in cfg.blocklist_patterns]
    blocklist_set = {t.lower() for t in cfg.blocklist_texts}

    kept: list[OcrEvent] = []
    dropped: dict[str, int] = {}

    for ev in events:
        reason = _reject_reason(
            ev, cfg, frame_height, compiled_patterns, blocklist_set
        )
        if reason is None:
            kept.append(ev)
        else:
            dropped[reason] = dropped.get(reason, 0) + 1

    if dropped:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
        log.info(
            "Filter: kept %d/%d (dropped: %s)",
            len(kept), len(events), summary,
        )
    else:
        log.info("Filter: kept %d/%d (nothing dropped)", len(kept), len(events))

    return kept


def events_to_subtitles(events: list[OcrEvent]) -> list[Subtitle]:
    """Convert surviving events to Subtitle objects.

    The event's start/end are used directly. The merger (when we build
    it) will decide differently for the merged mode.
    """
    return [
        Subtitle(start=ev.start, end=ev.end, text=ev.text)
        for ev in events
    ]


def _reject_reason(
    ev: OcrEvent,
    cfg: OcrFilterConfig,
    frame_height: int,
    compiled_patterns: list[re.Pattern],
    blocklist_set: set[str],
) -> str | None:
    """Return a rejection reason string, or None if the event should be kept."""
    if ev.confidence < cfg.min_confidence:
        return "low_confidence"

    if ev.duration < cfg.min_duration:
        return "too_short"
    if ev.duration > cfg.max_duration:
        return "too_long"

    box_h = ev.box[3] - ev.box[1]
    ratio = box_h / frame_height
    if ratio < cfg.min_box_height_ratio:
        return "box_too_small"
    if ratio > cfg.max_box_height_ratio:
        return "box_too_large"

    text = ev.text.strip()
    if len(text) < cfg.min_text_length:
        return "text_too_short"

    if text.lower() in blocklist_set:
        return "blocklisted"

    for pattern in compiled_patterns:
        if pattern.fullmatch(text):
            return "pattern_match"

    return None
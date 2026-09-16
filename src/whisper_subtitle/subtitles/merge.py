"""Merge whisper subtitles with OCR events into one combined subtitle list.

Whisper and OCR are two independent interpretations of what is happening
on screen. They should only merge when they're talking about the same
content:

  - Overlap with high similarity -> whisper text is used, no hint.
  - Overlap with moderate similarity -> whisper text is used, OCR text
    is attached as `source_hint` for downstream translation.
  - Overlap with low similarity -> different content. Whisper wins, OCR
    event is discarded (it would produce an overlapping subtitle).
  - No overlap, fill_gaps enabled -> OCR event becomes its own subtitle.

Whisper always wins when both sources have content for the same time.
OCR only contributes where whisper is silent.
"""

import logging
import re
from dataclasses import replace
from difflib import SequenceMatcher

from whisper_subtitle.config import MergeConfig
from whisper_subtitle.models import OcrEvent, Subtitle

log = logging.getLogger(__name__)


_WS_RE = re.compile(r"\s+")

# Leading bracketed prefix: "[SIGN]", "(웃음)", "[일단무시]"
_BRACKET_PREFIX_RE = re.compile(r"^\s*[\(\[]\s*[^\)\]]{1,20}\s*[\)\]]\s*")

# Leading speaker prefix: "미나미어머니*", "ZENA#", "Minami:", "원이·"
_SPEAKER_PREFIX_RE = re.compile(r"^\s*[가-힣A-Za-z]{1,15}\s*[*#:·]\s*")


def merge_whisper_ocr(
    whisper_subs: list[Subtitle],
    ocr_events: list[OcrEvent],
    cfg: MergeConfig,
) -> list[Subtitle]:
    """Combine whisper subtitles and OCR events into one list.

    Args:
        whisper_subs: subtitles from the whisper path, already on the
            global timeline.
        ocr_events: OCR events that passed all filters, already on the
            global timeline.
        cfg: merge configuration.

    Returns:
        Combined subtitles, sorted by start time. Whisper entries may
        carry a `source_hint` when the OCR reading was moderately
        different (0.5 <= similarity < 0.9).
    """
    if not whisper_subs and not ocr_events:
        return []
    if not ocr_events:
        return list(whisper_subs)
    if not whisper_subs:
        return _ocr_events_to_subtitles(ocr_events, cfg) if cfg.fill_gaps else []

    claimed: set[int] = set()
    merged: list[Subtitle] = []

    for sub in whisper_subs:
        match_idx, similarity = _find_best_ocr_match(sub, ocr_events, claimed, cfg)
        if match_idx is None:
            merged.append(sub)
            continue

        claimed.add(match_idx)
        ocr_text = ocr_events[match_idx].text

        if similarity >= cfg.agree_threshold:
            merged.append(sub)
        elif similarity >= cfg.hint_threshold:
            merged.append(replace(sub, source_hint=ocr_text))
        else:
            # Different content — whisper wins, OCR event consumed.
            merged.append(sub)

    if cfg.fill_gaps:
        for i, ev in enumerate(ocr_events):
            if i in claimed:
                continue
            if _overlaps_any_whisper(ev, whisper_subs, cfg.overlap_threshold):
                continue
            if ev.duration < cfg.gap_fill_min_duration:
                continue
            text = cfg.gap_fill_prefix + ev.text if cfg.gap_fill_prefix else ev.text
            merged.append(Subtitle(start=ev.start, end=ev.end, text=text))

    merged.sort(key=lambda s: s.start)

    log.info(
        "Merged %d whisper + %d OCR events -> %d subtitles "
        "(%d OCR as hints, %d OCR as gap-fills)",
        len(whisper_subs),
        len(ocr_events),
        len(merged),
        sum(1 for s in merged if s.source_hint is not None),
        len(merged) - len(whisper_subs),
    )
    return merged


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------


def _find_best_ocr_match(
    sub: Subtitle,
    ocr_events: list[OcrEvent],
    claimed: set[int],
    cfg: MergeConfig,
) -> tuple[int | None, float]:
    """Find the OCR event that best matches this whisper subtitle.

    Prefers higher time overlap; ties broken by higher text similarity.
    Returns (index, similarity) or (None, 0.0) if nothing qualifies.
    """
    best_idx: int | None = None
    best_overlap = 0.0
    best_similarity = 0.0

    for i, ev in enumerate(ocr_events):
        if i in claimed:
            continue
        overlap = _overlap_ratio(sub.start, sub.end, ev.start, ev.end)
        if overlap < cfg.overlap_threshold:
            continue
        similarity = _best_similarity(sub.text, ev.text)
        if overlap > best_overlap or (
            overlap == best_overlap and similarity > best_similarity
        ):
            best_idx = i
            best_overlap = overlap
            best_similarity = similarity

    if best_idx is None:
        return None, 0.0
    return best_idx, best_similarity


def _overlaps_any_whisper(
    ev: OcrEvent,
    whisper_subs: list[Subtitle],
    threshold: float,
) -> bool:
    for sub in whisper_subs:
        if _overlap_ratio(sub.start, sub.end, ev.start, ev.end) >= threshold:
            return True
    return False


def _overlap_ratio(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if overlap <= 0.0:
        return 0.0
    shorter = min(a_end - a_start, b_end - b_start)
    if shorter <= 0.0:
        return 0.0
    return overlap / shorter


# ----------------------------------------------------------------------
# Text similarity
# ----------------------------------------------------------------------


def _normalize(text: str) -> str:
    return _WS_RE.sub("", text.lower())


def _strip_ocr_prefix(text: str) -> str:
    """Strip a leading speaker or bracketed prefix from OCR text.

    OCR often reads on-screen speaker labels merged with the subtitle,
    e.g. "미나미어머니*지금까지정말많은..." or "[SIGN] TOKYO". We strip
    these before comparison so the similarity score reflects the actual
    subtitle text.
    """
    text = _BRACKET_PREFIX_RE.sub("", text)
    text = _SPEAKER_PREFIX_RE.sub("", text)
    return text


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _best_similarity(whisper_text: str, ocr_text: str) -> float:
    """Best similarity across raw and prefix-stripped OCR text."""
    raw = _similarity(whisper_text, ocr_text)
    stripped = _strip_ocr_prefix(ocr_text)
    if stripped and stripped != ocr_text:
        return max(raw, _similarity(whisper_text, stripped))
    return raw


# ----------------------------------------------------------------------
# OCR-only path (no whisper available)
# ----------------------------------------------------------------------


def _ocr_events_to_subtitles(
    events: list[OcrEvent],
    cfg: MergeConfig,
) -> list[Subtitle]:
    kept = [ev for ev in events if ev.duration >= cfg.gap_fill_min_duration]
    kept.sort(key=lambda e: e.start)
    prefix = cfg.gap_fill_prefix
    return [
        Subtitle(
            start=ev.start,
            end=ev.end,
            text=(prefix + ev.text) if prefix else ev.text,
        )
        for ev in kept
    ]

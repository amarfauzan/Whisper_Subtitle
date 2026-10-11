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

    Whisper provides the dialogue and its timing. OCR is used three ways:

      - OCR reading that matches the overlapping whisper cue (similarity
        >= `unrelated_threshold`) -> attached as `source_hint` when the
        readings differ, so translation sees both.
      - OCR text that overlaps speech but says something different ->
        that is on-screen information (venue notices, product/place
        cards), not dialogue. It is kept as its own cue instead of being
        thrown away, when `keep_unrelated_ocr` is set.
      - OCR text in a speech gap -> gap-fill, if `fill_gaps` is enabled.

    Effect text (short bracketed captions, laughter/punctuation) and
    near-duplicates of a neighbouring cue are dropped.

    Args:
        whisper_subs: subtitles from the whisper path, on the global timeline.
        ocr_events: OCR events that passed all filters, on the global timeline.
        cfg: merge configuration.

    Returns:
        Combined subtitles, sorted by start time. Whisper entries may
        carry a `source_hint` when OCR overlapped them; OCR entries may
        overlap a whisper cue on purpose (on-screen text).
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

        # Attach the OCR reading as a hint for the LLM to use at
        # translation time. On agreement (similarity >= agree_threshold)
        # the OCR text is redundant and would only add noise, so skip it.
        if similarity >= cfg.agree_threshold:
            merged.append(sub)
        else:
            merged.append(replace(sub, source_hint=ocr_text))

    if cfg.fill_gaps:
        for i, ev in enumerate(ocr_events):
            if i in claimed:
                continue
            if ev.duration < cfg.gap_fill_min_duration:
                continue

            # An OCR reading that matches the speech it overlaps is the
            # same dialogue (it just lost the claim contest).
            overlap_sim = _best_overlap_similarity(
                ev, whisper_subs, cfg.overlap_threshold
            )
            # Burn-in captions can lag or lead the audio: also reject a
            # re-timed reading of neighbouring speech.
            if _adjacent_speech_similarity(ev, whisper_subs) >= _ADJACENT_SIMILARITY:
                continue

            if overlap_sim is not None:
                if not cfg.keep_unrelated_ocr:
                    continue
                if overlap_sim >= cfg.unrelated_threshold:
                    continue
                # Speech is playing, so only genuine on-screen
                # information survives; short fragments are caption noise.
                if not _looks_like_information(ev.text, cfg.info_min_chars):
                    continue
            elif not _looks_like_info_text(ev.text):
                continue

            text = cfg.gap_fill_prefix + ev.text if cfg.gap_fill_prefix else ev.text
            if _is_redundant(text, merged, ev.start):
                continue
            merged.append(Subtitle(start=ev.start, end=ev.end, text=text))

    merged.sort(key=lambda s: s.start)

    log.info(
        "Merged %d whisper + %d OCR events -> %d subtitles "
        "(%d hints, %d on-screen text cues)",
        len(whisper_subs),
        len(ocr_events),
        len(merged),
        sum(1 for s in merged if s.source_hint is not None),
        len(merged) - len(whisper_subs),
    )
    return merged


# Effect text: laughter runs, punctuation-only, and short bracketed
# captions like "((소스발견))" that are decoration, not information.
_EFFECT_ONLY_RE = re.compile(r"^[\s~!?.,…ㅋㅎㅠㅜ\(\)\[\]\{\}]+$")
_WRAPPED_RE = re.compile(r"^[\(\[]\s*(.*?)\s*[\)\]]$", re.DOTALL)

# How close (in seconds) an OCR event must be to speech before we treat it
# as a possible reading of that speech rather than a gap-fill.
_ADJACENT_WINDOW_SECONDS = 1.5
# Similarity at which adjacent speech counts as the same utterance.
_ADJACENT_SIMILARITY = 0.5
# Text similarity at which two nearby cues are treated as duplicates.
_DUPLICATE_SIMILARITY = 0.8
# How far apart two OCR cues may be and still count as duplicates.
_DUPLICATE_WINDOW_SECONDS = 5.0


def _looks_like_information(text: str, min_chars: int) -> bool:
    """True when OCR text next to speech looks like real on-screen info.

    Dialogue captions appear as short unspaced fragments next to the
    speech they transcribe, so require either a full line of text or a
    digit (prices, dates, counts, channel numbers).
    """
    if not _looks_like_info_text(text):
        return False
    if re.search(r"[0-9]", text):
        return True
    return len(_normalize(text)) >= min_chars


def _adjacent_speech_similarity(
    ev: OcrEvent,
    whisper_subs: list[Subtitle],
) -> float:
    """Best similarity to speech starting/ending just before or after."""
    best = 0.0
    for sub in whisper_subs:
        gap = max(sub.start - ev.end, ev.start - sub.end, 0.0)
        if gap > _ADJACENT_WINDOW_SECONDS:
            continue
        best = max(best, _best_similarity(sub.text, ev.text))
    return best


def _looks_like_info_text(text: str) -> bool:
    """True when OCR text looks like on-screen information worth keeping."""
    stripped = text.strip()
    if len(_normalize(stripped)) < 2:
        return False
    if _EFFECT_ONLY_RE.fullmatch(stripped):
        return False
    wrapped = _WRAPPED_RE.fullmatch(stripped)
    return not (wrapped and len(_normalize(wrapped.group(1))) < 8)


def _is_redundant(text: str, existing: list[Subtitle], start: float) -> bool:
    """True when an OCR cue repeats text already emitted near this time.

    Catches exact repeats, fragments already covered by a neighbouring
    cue, incremental captions (a cue that extends an earlier one), and
    near-identical re-readings.
    """
    key = _normalize(text)
    for sub in existing:
        if abs(sub.start - start) > _DUPLICATE_WINDOW_SECONDS:
            continue
        other = _normalize(sub.text)
        if not other:
            continue
        if key == other:
            return True
        if len(key) >= 4 and key in other:
            return True
        if len(other) >= 4 and other in key:
            # Incremental caption: this cue just grows the previous one.
            return True
        if _similarity(text, sub.text) >= _DUPLICATE_SIMILARITY:
            return True
    return False


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

    Only OCR readings similar enough to be the *same speech* qualify
    (similarity >= `unrelated_threshold`). Anything less similar is
    on-screen information rather than a reading of this cue, so it is
    left unclaimed for the on-screen-text path.

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
        if similarity < cfg.unrelated_threshold:
            continue
        if overlap > best_overlap or (
            overlap == best_overlap and similarity > best_similarity
        ):
            best_idx = i
            best_overlap = overlap
            best_similarity = similarity

    if best_idx is None:
        return None, 0.0
    return best_idx, best_similarity


def _best_overlap_similarity(
    ev: OcrEvent,
    whisper_subs: list[Subtitle],
    threshold: float,
) -> float | None:
    """Best text similarity to any whisper cue this OCR event overlaps.

    Returns None when the event overlaps no whisper cue at all.
    """
    best: float | None = None
    for sub in whisper_subs:
        if _overlap_ratio(sub.start, sub.end, ev.start, ev.end) < threshold:
            continue
        similarity = _best_similarity(sub.text, ev.text)
        if best is None or similarity > best:
            best = similarity
    return best


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
    kept = [
        ev for ev in events
        if ev.duration >= cfg.gap_fill_min_duration and _looks_like_info_text(ev.text)
    ]
    kept.sort(key=lambda e: e.start)
    prefix = cfg.gap_fill_prefix

    out: list[Subtitle] = []
    for ev in kept:
        text = (prefix + ev.text) if prefix else ev.text
        if _is_redundant(text, out, ev.start):
            continue
        out.append(Subtitle(start=ev.start, end=ev.end, text=text))
    return out

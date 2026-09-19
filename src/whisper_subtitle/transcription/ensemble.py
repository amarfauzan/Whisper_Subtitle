"""Merge two ASR transcripts using V3's timing and BAT's text.

Takes large-v3's finer-grained cues as the timing skeleton. For each
V3 cue, finds overlapping BAT cues, attaches BAT's text as an alternate
reading. Where they disagree significantly, marks for LLM adjudication
(LLM call happens in a separate step).
"""

import logging
import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from whisper_subtitle.config import EnsembleConfig
from whisper_subtitle.models import WhisperSegment

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub("", text.lower())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _overlap(a: WhisperSegment, b: WhisperSegment) -> float:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    if hi <= lo:
        return 0.0
    shorter = min(a.end - a.start, b.end - b.start)
    return (hi - lo) / shorter if shorter > 0 else 0.0


@dataclass
class AdjudicationRequest:
    """A single cue where BAT and V3 disagree — sent to LLM."""

    index: int              # position in the merged list
    v3_text: str
    bat_text: str
    context_before: str     # previous 2 cues' text for context
    context_after: str


@dataclass
class MergeResult:
    cues: list[WhisperSegment]
    pending: list[AdjudicationRequest]


def merge_ensemble(
    bat: list[WhisperSegment],
    v3: list[WhisperSegment],
    cfg: EnsembleConfig,
) -> MergeResult:
    """Merge two transcripts. Returns cues + pending LLM requests.

    The pipeline calls this, then optionally resolves pending requests
    via LLM (see llm_adjudicate), then writes the final SRT.
    """
    if not v3:
        return MergeResult(cues=[], pending=[])
    if not bat:
        return MergeResult(
            cues=[v for v in v3 if not _is_junk(v, cfg)],
            pending=[],
        )

    # Step 1: assign each BAT cue to the V3 cue with maximum overlap.
    # A BAT cue is used at most once; multiple BAT cues can feed one V3 cue.
    bat_assignment: dict[int, list[WhisperSegment]] = {i: [] for i in range(len(v3))}
    for b in bat:
        # Find ALL V3 cues this BAT cue meaningfully overlaps
        overlapping = [
            (i, v) for i, v in enumerate(v3)
            if _overlap(b, v) >= cfg.overlap_threshold
        ]
        if not overlapping:
            continue

        if len(overlapping) == 1:
            bat_assignment[overlapping[0][0]].append(b)
            continue

        # Multiple V3 cues — split BAT text across them
        overlapping.sort(key=lambda x: x[1].start)
        v3_indices = [i for i, _ in overlapping]
        v3_cues = [v for _, v in overlapping]
        slices = _split_bat_across_v3(b.text, v3_cues)

        for idx, slice_text in zip(v3_indices, slices):
            if slice_text:
                bat_assignment[idx].append(replace(b, text=slice_text))

    # Step 2: for each V3 cue, build merged text and decide
    cues: list[WhisperSegment] = []
    pending: list[AdjudicationRequest] = []

    for i, v in enumerate(v3):
        if _is_junk(v, cfg):
            continue

        assigned = sorted(bat_assignment[i], key=lambda b: b.start)
        if not assigned:
            # V3-only cue — BAT missed this window
            cues.append(v)
            continue

        bat_text = " ".join(b.text.strip() for b in assigned if b.text.strip())
        if not bat_text:
            cues.append(v)
            continue

        sim = _similarity(v.text, bat_text)

        if sim >= cfg.agree_threshold:
            # High agreement — keep V3's text (usually cleaner)
            cues.append(v)
        elif sim >= cfg.hint_threshold:
            # Moderate agreement — prefer BAT's text (usually more natural)
            cues.append(replace(v, text=bat_text))
        else:
            # Real disagreement — mark for LLM
            provisional = v  # placeholder; text will be replaced if LLM resolves
            idx = len(cues)
            cues.append(provisional)
            pending.append(AdjudicationRequest(
                index=idx,
                v3_text=v.text,
                bat_text=bat_text,
                context_before=" ".join(c.text for c in cues[max(0, idx-2):idx]),
                context_after="",  # filled in below
            ))

    # Fill context_after now that we know all cues
    for req in pending:
        after_start = req.index + 1
        after_end = min(after_start + 2, len(cues))
        req.context_after = " ".join(c.text for c in cues[after_start:after_end])

    log.info(
        "Ensemble merge: %d V3 cues → %d kept, %d pending adjudication",
        len(v3), len(cues), len(pending),
    )
    return MergeResult(cues=cues, pending=pending)


def _is_junk(seg: WhisperSegment, cfg: EnsembleConfig) -> bool:
    """Drop tiny noise cues that V3 sometimes emits (single dots, stray chars)."""
    if seg.duration >= cfg.min_cue_duration:
        return False
    text = seg.text.strip()
    if len(text) >= cfg.min_cue_chars:
        return False
    return True

def _split_bat_across_v3(
    bat_text: str,
    v3_cues: list[WhisperSegment],
) -> list[str]:
    """Split BAT text into len(v3_cues) pieces.

    Uses V3 cue text lengths as a proportional guide for where to cut.
    Falls back to a single-piece result if BAT text is too short to split.
    """
    if len(v3_cues) <= 1:
        return [bat_text]

    # If BAT text is too short to meaningfully split, keep it whole
    if len(bat_text) < 4 * len(v3_cues):
        return [bat_text] + [""] * (len(v3_cues) - 1)

    weights = [max(len(v.text.strip()), 1) for v in v3_cues]
    total_weight = sum(weights)

    pieces: list[str] = []
    cursor = 0
    cumulative = 0
    for weight in weights[:-1]:
        cumulative += weight
        target_char = int(len(bat_text) * cumulative / total_weight)
        split_at = _find_split_near(bat_text, cursor, target_char)
        pieces.append(bat_text[cursor:split_at].strip())
        cursor = split_at
    pieces.append(bat_text[cursor:].strip())
    return pieces


def _find_split_near(text: str, min_pos: int, target: int) -> int:
    """Find the nearest whitespace near `target`, at or after `min_pos`.

    Returns the index just AFTER the split character so the space or
    punctuation stays with the left piece.
    """
    n = len(text)
    if target >= n - 1:
        return n
    if target <= min_pos:
        target = min_pos + 1

    # Search outward from target for a whitespace or punctuation
    for offset in range(0, n):
        for pos in (target + offset, target - offset):
            if pos <= min_pos or pos >= n:
                continue
            if text[pos] in " ,.?!·、。？！":
                return pos + 1
    return target
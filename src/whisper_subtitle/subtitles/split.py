"""Split over-long subtitles into readable cues.

Whisper-server mode never applies whisper-cli's `--max-len`/`-sow` line
splitting, so cues can come out as 100+ character, 9-second text walls.
This pass chops them at sentence, then word, then character boundaries
and distributes the original time range across the pieces, so the SRT
stays readable without re-timing the transcription.
"""

import logging
import math
import re

from whisper_subtitle.models import Subtitle

log = logging.getLogger(__name__)


# Split after sentence-ending punctuation, keeping the punctuation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!…])\s+")


def split_long_subtitles(
    subtitles: list[Subtitle],
    *,
    max_chars: int,
    max_duration: float,
    min_duration: float = 0.5,
) -> list[Subtitle]:
    """Return subtitles with over-long cues broken into smaller ones.

    A cue is split when its text is longer than `max_chars` or its
    duration is longer than `max_duration`. Pieces never go below
    `min_duration` — if the original cue is too short to hold that many
    pieces, the text is packed into fewer, larger pieces instead.

    Ordering and total time span are preserved.
    """
    if max_chars <= 0 and max_duration <= 0:
        return list(subtitles)

    out: list[Subtitle] = []
    split_count = 0
    for sub in subtitles:
        pieces = _split_one(sub, max_chars, max_duration, min_duration)
        if len(pieces) > 1:
            split_count += 1
        out.extend(pieces)

    if split_count:
        log.info(
            "Split %d over-long cue(s) (was %d, now %d)",
            split_count, len(subtitles), len(out),
        )
    return out


def _split_one(
    sub: Subtitle,
    max_chars: int,
    max_duration: float,
    min_duration: float,
) -> list[Subtitle]:
    text = sub.text.strip()
    if not text:
        return [sub]

    too_long_text = max_chars > 0 and len(text) > max_chars
    too_long_time = max_duration > 0 and sub.duration > max_duration
    if not (too_long_text or too_long_time):
        return [sub]

    # How many pieces do the text and the duration each require?
    target = 1
    if max_chars > 0:
        target = max(target, math.ceil(len(text) / max_chars))
    if max_duration > 0 and sub.duration > 0:
        target = max(target, math.ceil(sub.duration / max_duration))
    if target <= 1:
        return [sub]

    # ...but every piece must still be on screen for min_duration, so
    # there is an upper bound on how many pieces the cue can hold.
    hard_max = len(text)  # effectively "no limit"
    if min_duration > 0 and sub.duration > 0:
        hard_max = max(1, int(sub.duration / min_duration))

    chunks = _chunk_text(text, max_chars if max_chars > 0 else len(text), target)
    if len(chunks) <= 1:
        # Nothing further to split textually (e.g. one short word).
        return [sub]
    if len(chunks) > hard_max:
        # Too many pieces for the available time: merge them back so no
        # text is lost, even if a piece then exceeds the char cap.
        chunks = _pack(chunks, hard_max)
    if len(chunks) <= 1:
        return [sub]

    weights = [max(len(c), 1) for c in chunks]
    total_weight = sum(weights)
    duration = max(sub.duration, 0.0)

    pieces: list[Subtitle] = []
    cursor = sub.start
    for i, (chunk, weight) in enumerate(zip(chunks, weights)):
        if i == len(chunks) - 1:
            end = sub.end
        else:
            end = cursor + duration * (weight / total_weight)
            end = min(end, sub.end)
            if end - cursor < min_duration:
                end = min(sub.end, cursor + min_duration)
        pieces.append(
            Subtitle(
                start=round(cursor, 3),
                end=round(end, 3),
                text=chunk,
                # The hint describes the whole utterance; keep it on the
                # first piece so translation still sees it once.
                source_hint=sub.source_hint if i == 0 else None,
            )
        )
        cursor = end

    return [p for p in pieces if p.end > p.start or len(pieces) == 1]


def _chunk_text(text: str, max_chars: int, target: int = 1) -> list[str]:
    """Split text into balanced units of at most max_chars.

    Units are sentences when they fit, otherwise words (or raw
    characters for unspaced OCR text). They are then packed to a width
    that balances the target piece count, so a long cue does not end up
    with one huge piece and one tiny remainder.
    """
    units = _units(text, max_chars)
    if not units:
        return [text]

    width = max_chars
    if target > 1:
        total = sum(len(u) for u in units)
        width = min(max_chars, max(1, math.ceil(total / target)))
    return _pack_width(units, width)


def _units(text: str, max_chars: int) -> list[str]:
    """Break text into sentences, words, or character runs."""
    out: list[str] = []
    for line in text.split("\n"):
        for sentence in _SENTENCE_SPLIT_RE.split(line.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                out.append(sentence)
                continue

            words = sentence.split(" ")
            longest = max((len(w) for w in words), default=0)
            if len(words) == 1 or longest > max_chars:
                # Unspaced OCR text (or one giant token): hard split.
                out.extend(
                    sentence[i:i + max_chars]
                    for i in range(0, len(sentence), max_chars)
                )
            else:
                out.extend(words)
    return out


def _pack_width(units: list[str], max_chars: int) -> list[str]:
    """Greedily merge units into chunks no longer than max_chars."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _pack(units: list[str], target_count: int) -> list[str]:
    """Greedily merge units into at most target_count chunks."""
    if target_count <= 1:
        return [" ".join(u.strip() for u in units).strip()]

    total_chars = sum(len(u) for u in units) or 1
    per_chunk = max(1, total_chars // target_count)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if current and len(candidate) > per_chunk and len(chunks) < target_count - 1:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks

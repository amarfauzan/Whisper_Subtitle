"""Merge VAD speech boundaries with whisper's text segments."""

from whisper_subtitle.models import Subtitle, VadSegment, WhisperSegment


def merge_vad_and_whisper(
    vad_segments: list[VadSegment],
    whisper_segments: list[WhisperSegment],
    *,
    max_merge_duration: float = 2.0,
) -> list[Subtitle]:
    """Build final subtitles from VAD boundaries and whisper text.

    For each VAD speech region:
      - If it is shorter than max_merge_duration, all overlapping whisper
        text is merged into one subtitle bounded by the VAD's start/end.
      - Otherwise, whisper's own segment breaks are kept, clipped to the
        VAD's boundaries.

    Whisper segments that do not overlap any VAD region are dropped —
    VAD says no speech there, so whisper is hallucinating.

    Args:
        vad_segments: acoustic speech regions (from standalone VAD).
        whisper_segments: whisper's text segments with approximate times.
        max_merge_duration: VAD segments shorter than this are collapsed
            into a single subtitle. Default 2.0 seconds.

    Returns:
        A list of Subtitle objects, sorted by start time.
    """
    groups = _group_whisper_by_vad(vad_segments, whisper_segments)

    subtitles: list[Subtitle] = []
    for vad in vad_segments:
        group = groups[vad.index]
        if not group:
            continue
        subtitles.extend(_emit_for_vad(vad, group, max_merge_duration))

    subtitles.sort(key=lambda s: s.start)
    return subtitles


def _group_whisper_by_vad(
    vad_segments: list[VadSegment],
    whisper_segments: list[WhisperSegment],
) -> dict[int, list[WhisperSegment]]:
    """Assign each whisper segment to the VAD segment with the most overlap.

    Whisper segments with no overlap at all are dropped.
    """
    groups: dict[int, list[WhisperSegment]] = {v.index: [] for v in vad_segments}

    for w in whisper_segments:
        best_vad = None
        best_overlap = 0.0

        for v in vad_segments:
            overlap = min(w.end, v.end) - max(w.start, v.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_vad = v

        if best_vad is not None:
            groups[best_vad.index].append(w)

    # Sort each group by start time so output stays ordered within a VAD.
    for group in groups.values():
        group.sort(key=lambda w: w.start)

    return groups


def _emit_for_vad(
    vad: VadSegment,
    group: list[WhisperSegment],
    max_merge_duration: float,
) -> list[Subtitle]:
    """Turn a VAD segment and its assigned whisper segments into subtitles."""
    if vad.duration <= max_merge_duration:
        # Short speech region: collapse to one subtitle with VAD boundaries.
        text = " ".join(w.text for w in group).strip()
        if not text:
            return []
        return [Subtitle(start=vad.start, end=vad.end, text=text)]

    # Long speech region: keep whisper's breaks, clipped to VAD bounds.
    subtitles = []
    for w in group:
        start = max(w.start, vad.start)
        end = min(w.end, vad.end)
        if end <= start:
            continue  # collapsed to nothing after clipping
        subtitles.append(Subtitle(start=start, end=end, text=w.text))
    return subtitles

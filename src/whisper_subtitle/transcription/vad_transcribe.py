"""Transcribe a chunk by running whisper on each VAD segment independently.

Whisper's --vad flag concatenates speech regions before feeding the model,
which is efficient but causes cross-segment hallucination: the model can't
tell that two clips were separated in the original audio, so it echoes its
own context when uncertain.

This strategy avoids that by running whisper on each VAD segment as its own
isolated clip. The model's context is just that one segment — no cross-
segment contamination. Costs more whisper calls, but with the server
backend the model stays loaded, so the per-call overhead is minimal.
"""

import logging
import tempfile
from pathlib import Path

from whisper_subtitle.config import WhisperConfig
from whisper_subtitle.exceptions import PipelineError
from whisper_subtitle.media.audio import extract_clip
from whisper_subtitle.models import VadSegment, WhisperSegment
from whisper_subtitle.transcription.whisper_backend import WhisperBackend

log = logging.getLogger(__name__)


def transcribe_by_vad_segments(
    audio_path: Path,
    vad_segments: list[VadSegment],
    backend: WhisperBackend,
    cfg: WhisperConfig,
) -> list[WhisperSegment]:
    """Transcribe each VAD segment as an isolated clip.

    Returns combined WhisperSegment objects on the audio's timeline,
    sorted by start time.
    """
    if not vad_segments:
        log.warning("No VAD segments to transcribe: %s", audio_path.name)
        return []

    merged = _merge_close_segments(vad_segments, cfg.vad_segments.merge_gap)
    merged = [
        s for s in merged
        if s.duration >= cfg.vad_segments.min_segment_duration
    ]

    if not merged:
        log.warning(
            "No VAD segments survived merge/min-duration filters: %s",
            audio_path.name,
        )
        return []

    log.info(
        "Transcribing %d VAD segments from %s (strategy=vad_segments)",
        len(merged), audio_path.name,
    )

    all_segments: list[WhisperSegment] = []
    failures = 0

    with tempfile.TemporaryDirectory(prefix="vad_clips_") as tmp:
        tmp_dir = Path(tmp)

        for i, _ in enumerate(merged, start=1):
            clip_start, clip_end = _padded_range(
                i - 1, merged, cfg.vad_segments.padding
            )
            clip_path = tmp_dir / f"clip_{i:04d}.wav"

            try:
                extract_clip(audio_path, clip_start, clip_end, clip_path)
                result = backend.transcribe(clip_path)
            except PipelineError as exc:
                log.warning(
                    "Segment %d/%d failed (%.2f-%.2fs): %s",
                    i, len(merged), clip_start, clip_end, exc,
                )
                failures += 1
                continue

            for wseg in result.segments:
                all_segments.append(WhisperSegment(
                    start=wseg.start + clip_start,
                    end=wseg.end + clip_start,
                    text=wseg.text,
                ))

            if i % 20 == 0:
                log.debug(
                    "Transcribed %d/%d segments (%d whisper segments so far)",
                    i, len(merged), len(all_segments),
                )

    if failures:
        log.warning(
            "Transcription finished with %d/%d segment failures",
            failures, len(merged),
        )

    all_segments.sort(key=lambda s: s.start)
    log.info(
        "VAD-segment transcription complete: %d whisper segments from %d clips",
        len(all_segments), len(merged),
    )
    return all_segments


def _merge_close_segments(
    segments: list[VadSegment],
    merge_gap: float,
) -> list[VadSegment]:
    """Merge VAD segments whose gap is smaller than merge_gap."""
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.start)
    merged: list[VadSegment] = [sorted_segs[0]]

    for nxt in sorted_segs[1:]:
        prev = merged[-1]
        if nxt.start - prev.end <= merge_gap:
            merged[-1] = VadSegment(
                index=prev.index,
                start=prev.start,
                end=max(prev.end, nxt.end),
            )
        else:
            merged.append(nxt)

    return merged


def _padded_range(
    idx: int,
    merged: list[VadSegment],
    padding: float,
) -> tuple[float, float]:
    """Clip range for the segment at `idx`, padded but bounded by midpoints
    to the previous/next segments so consecutive clips never overlap."""
    seg = merged[idx]

    start = max(0.0, seg.start - padding)
    end = seg.end + padding

    if idx > 0:
        prev_end = merged[idx - 1].end
        midpoint = (prev_end + seg.start) / 2
        start = max(start, midpoint)

    if idx < len(merged) - 1:
        next_start = merged[idx + 1].start
        midpoint = (seg.end + next_start) / 2
        end = min(end, midpoint)

    return start, end
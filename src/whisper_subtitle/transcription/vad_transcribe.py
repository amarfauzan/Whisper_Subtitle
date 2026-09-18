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
import wave
from dataclasses import dataclass

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

# ----------------------------------------------------------------------
# Batched transcription
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _MappingEntry:
    """A region in the batch WAV and where it came from in the original audio."""

    batch_start: float
    batch_end: float
    original_start: float
    original_end: float
    is_speech: bool


def _build_batch_clip(
    audio_path: Path,
    segments: list[VadSegment],
    out_path: Path,
    silence_ms: int,
) -> tuple[list[_MappingEntry], float]:
    """Build a batch WAV by concatenating VAD segments with silence between them.

    Returns:
        (mapping, batch_duration). mapping has one entry per speech region
        and one per silence gap. batch_duration is the total seconds written.
    """
    with wave.open(str(audio_path), "rb") as src:
        params = src.getparams()
        framerate = params.framerate
        silence_frames = int((silence_ms / 1000) * framerate)
        silence_bytes = b"\x00" * (params.sampwidth * params.nchannels) * silence_frames

        all_frames = b""
        mapping: list[_MappingEntry] = []
        batch_pos = 0.0

        for i, seg in enumerate(segments):
            # Copy the speech region verbatim
            src.setpos(int(seg.start * framerate))
            n_frames = int((seg.end - seg.start) * framerate)
            all_frames += src.readframes(n_frames)

            duration = n_frames / framerate  # use the real (rounded) duration
            mapping.append(_MappingEntry(
                batch_start=batch_pos,
                batch_end=batch_pos + duration,
                original_start=seg.start,
                original_end=seg.start + duration,
                is_speech=True,
            ))
            batch_pos += duration

            # Silence gap between segments (not after the last)
            if i < len(segments) - 1:
                all_frames += silence_bytes
                gap_duration = silence_frames / framerate
                mapping.append(_MappingEntry(
                    batch_start=batch_pos,
                    batch_end=batch_pos + gap_duration,
                    original_start=seg.end,
                    original_end=seg.end,
                    is_speech=False,
                ))
                batch_pos += gap_duration

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(all_frames)

    return mapping, batch_pos


def _map_batch_time(t: float, mapping: list[_MappingEntry]) -> float:
    """Map a position in batch time to the corresponding time in the source audio."""
    for entry in mapping:
        if entry.batch_start <= t <= entry.batch_end:
            if entry.batch_end == entry.batch_start:
                return entry.original_start
            frac = (t - entry.batch_start) / (entry.batch_end - entry.batch_start)
            return entry.original_start + frac * (entry.original_end - entry.original_start)
    # Out of range: clamp
    if t < mapping[0].batch_start:
        return mapping[0].original_start
    return mapping[-1].original_end


def _map_whisper_segment(wseg: WhisperSegment, mapping: list[_MappingEntry]) -> WhisperSegment:
    return WhisperSegment(
        start=_map_batch_time(wseg.start, mapping),
        end=_map_batch_time(wseg.end, mapping),
        text=wseg.text,
    )


def _group_into_batches(
    segments: list[VadSegment],
    max_duration: float,
    silence_seconds: float,
) -> list[list[VadSegment]]:
    """Group segments so each batch's total duration (including silence) <= max_duration."""
    batches: list[list[VadSegment]] = []
    current: list[VadSegment] = []
    current_duration = 0.0

    for seg in segments:
        seg_duration = seg.duration
        extra = seg_duration + (silence_seconds if current else 0.0)

        if current and current_duration + extra > max_duration:
            batches.append(current)
            current = [seg]
            current_duration = seg_duration
        else:
            current.append(seg)
            current_duration += extra

    if current:
        batches.append(current)
    return batches


def _coverage_ok(
    result_segments: list[WhisperSegment],
    batch_duration: float,
    min_coverage: float,
) -> bool:
    if not result_segments:
        return False
    last_end = max(s.end for s in result_segments)
    return last_end >= batch_duration * min_coverage


def transcribe_batched_vad_segments(
    audio_path: Path,
    vad_segments: list[VadSegment],
    backend: WhisperBackend,
    cfg: WhisperConfig,
) -> list[WhisperSegment]:
    """Batch VAD segments into larger clips, transcribe each batch in one call.

    Falls back to per-segment transcription for any batch where coverage
    looks wrong (rare interactions with whisper's internal sliding window).
    """
    if not vad_segments:
        return []

    merged = _merge_close_segments(vad_segments, cfg.vad_segments.merge_gap)
    merged = [s for s in merged if s.duration >= cfg.vad_segments.min_segment_duration]
    if not merged:
        return []

    batches = _group_into_batches(
        merged,
        cfg.vad_segments.max_batch_duration,
        cfg.vad_segments.silence_ms / 1000.0,
    )

    log.info(
        "Transcribing %d VAD segments in %d batches (max %.1fs/batch)",
        len(merged), len(batches), cfg.vad_segments.max_batch_duration,
    )

    all_segments: list[WhisperSegment] = []
    fallback_count = 0

    with tempfile.TemporaryDirectory(prefix="vad_batches_") as tmp:
        tmp_dir = Path(tmp)

        for i, batch in enumerate(batches, start=1):
            batch_path = tmp_dir / f"batch_{i:04d}.wav"
            mapping, batch_duration = _build_batch_clip(
                audio_path, batch, batch_path, cfg.vad_segments.silence_ms,
            )

            try:
                result = backend.transcribe(batch_path)
            except PipelineError as exc:
                log.warning("Batch %d failed: %s — falling back", i, exc)
                all_segments.extend(
                    transcribe_by_vad_segments(audio_path, batch, backend, cfg)
                )
                fallback_count += 1
                continue

            if not _coverage_ok(
                result.segments, batch_duration, cfg.vad_segments.min_coverage,
            ):
                last = max((s.end for s in result.segments), default=0.0)
                log.warning(
                    "Batch %d under-covered (%.1fs of %.1fs) — falling back",
                    i, last, batch_duration,
                )
                all_segments.extend(
                    transcribe_by_vad_segments(audio_path, batch, backend, cfg)
                )
                fallback_count += 1
                continue

            for wseg in result.segments:
                all_segments.append(_map_whisper_segment(wseg, mapping))

            if i % 5 == 0:
                log.debug("Batched %d/%d batches", i, len(batches))

    if fallback_count:
        log.info("Batched transcription: %d/%d batches fell back to per-segment",
                 fallback_count, len(batches))

    all_segments.sort(key=lambda s: s.start)
    log.info(
        "Batched transcription complete: %d whisper segments from %d batches",
        len(all_segments), len(batches),
    )
    return all_segments


def transcribe_vad_segments(
    audio_path: Path,
    vad_segments: list[VadSegment],
    backend: WhisperBackend,
    cfg: WhisperConfig,
) -> list[WhisperSegment]:
    """Dispatch to batched or per-segment strategy based on config."""
    if cfg.vad_segments.max_batch_duration > 0:
        return transcribe_batched_vad_segments(audio_path, vad_segments, backend, cfg)
    return transcribe_by_vad_segments(audio_path, vad_segments, backend, cfg)
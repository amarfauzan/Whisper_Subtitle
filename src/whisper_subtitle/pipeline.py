"""Orchestrate the video → SRT pipeline."""

import logging
from pathlib import Path

from whisper_subtitle.config import Config
from whisper_subtitle.media.ffmpeg import Chunk, extract_audio, split_video
from whisper_subtitle.models import Subtitle
from whisper_subtitle.subtitles.merger import merge_vad_and_whisper
from whisper_subtitle.subtitles.srt import render_srt
from whisper_subtitle.subtitles.timeline import shift_timeline
from whisper_subtitle.transcription.vad import run_vad
from whisper_subtitle.transcription.whisper_runner import run_whisper

log = logging.getLogger(__name__)


def process_video(
    input_path: Path,
    cfg: Config,
    *,
    translate: bool = False,
) -> Path:
    """Run the full pipeline and return the path to the final SRT."""
    input_path = input_path.resolve()
    output_dir = input_path.parent / f"{input_path.stem}_chunks"

    log.info("Splitting %s", input_path.name)
    chunks = split_video(input_path, output_dir, cfg.chunk_length_seconds)
    log.info("Processing %d chunk(s)", len(chunks))

    subtitles: list[Subtitle] = []
    for i, chunk in enumerate(chunks, start=1):
        log.info("Chunk %d/%d: %s", i, len(chunks), chunk.path.name)
        subtitles.extend(_process_chunk(chunk, cfg))

    final_srt = output_dir / f"{input_path.stem}.srt"
    final_srt.write_text(render_srt(subtitles), encoding="utf-8")
    log.info("Wrote %s (%d subtitles)", final_srt.name, len(subtitles))

    if translate:
        _run_translation(final_srt, cfg)

    return final_srt


def _process_chunk(chunk: Chunk, cfg: Config) -> list[Subtitle]:
    """Transcribe one chunk and shift its subtitles onto the video timeline."""
    audio = extract_audio(chunk.path)
    vad_segments = run_vad(audio, cfg.vad)
    whisper_result = run_whisper(audio, cfg.whisper)

    subtitles = merge_vad_and_whisper(
        vad_segments,
        whisper_result.segments,
        max_merge_duration=cfg.vad.max_merge_duration,
    )

    if chunk.offset:
        subtitles = shift_timeline(subtitles, chunk.offset)

    return subtitles


def _run_translation(srt_path: Path, cfg: Config) -> None:
    """Optional translation step. Imported lazily so the rest of the
    pipeline works even if translation dependencies are broken."""
    from whisper_subtitle.translation.translator import translate_srt
    translate_srt(srt_path, cfg)

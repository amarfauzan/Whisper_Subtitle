"""Orchestrate the video → SRT pipeline."""

import logging
from pathlib import Path

from whisper_subtitle.config import Config
from whisper_subtitle.exceptions import PipelineError
from whisper_subtitle.media.ffmpeg import Chunk, extract_audio, split_video
from whisper_subtitle.models import Subtitle
from whisper_subtitle.ocr.engine import PaddleOcrEngine
from whisper_subtitle.ocr.filter import events_to_subtitles, filter_events
from whisper_subtitle.ocr.grouper import group_detections
from whisper_subtitle.ocr.runner import sample_and_detect
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
    if not cfg.transcription.enabled and not cfg.ocr.enabled:
        raise PipelineError(
            "Both transcription and OCR are disabled. Nothing to do."
        )

    input_path = input_path.resolve()
    output_dir = input_path.parent / f"{input_path.stem}_chunks"

    log.info("Splitting %s", input_path.name)
    chunks = split_video(input_path, output_dir, cfg.chunk_length_seconds)
    log.info("Processing %d chunk(s)", len(chunks))

    # Build the OCR engine once if OCR is enabled. Loading ONNX models
    # takes ~1s, so we don't want to do it per chunk.
    engine = _build_ocr_engine(cfg) if cfg.ocr.enabled else None

    all_subs: list[Subtitle] = []
    for i, chunk in enumerate(chunks, start=1):
        log.info("Chunk %d/%d: %s", i, len(chunks), chunk.path.name)
        all_subs.extend(_process_chunk(chunk, cfg, engine))

    final_srt = output_dir / f"{input_path.stem}.srt"
    final_srt.write_text(render_srt(all_subs), encoding="utf-8")
    log.info("Wrote %s (%d subtitles)", final_srt.name, len(all_subs))

    if translate:
        _run_translation(final_srt, cfg)

    return final_srt


def _build_ocr_engine(cfg: Config) -> PaddleOcrEngine:
    log.info("Loading OCR engine")
    return PaddleOcrEngine(
        det_model=cfg.ocr.det_model,
        rec_model=cfg.ocr.rec_model,
        dict_file=cfg.ocr.dict_file,
        providers=list(cfg.ocr.providers),
        rec_threshold=cfg.ocr.rec_confidence,
    )


def _process_chunk(
    chunk: Chunk,
    cfg: Config,
    engine: PaddleOcrEngine | None,
) -> list[Subtitle]:
    """Dispatch one chunk to whichever source(s) are enabled."""
    if cfg.transcription.enabled and cfg.ocr.enabled:
        raise PipelineError(
            "Merged mode (transcription + OCR) is not implemented yet."
        )
    if cfg.transcription.enabled:
        return _process_chunk_whisper(chunk, cfg)
    if cfg.ocr.enabled and engine is not None:
        return _process_chunk_ocr(chunk, cfg, engine)
    return []


def _process_chunk_whisper(chunk: Chunk, cfg: Config) -> list[Subtitle]:
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


def _process_chunk_ocr(
    chunk: Chunk,
    cfg: Config,
    engine: PaddleOcrEngine,
) -> list[Subtitle]:
    detections, frame_height = sample_and_detect(chunk.path, engine, cfg.ocr)
    events = group_detections(detections)
    kept = filter_events(events, cfg.ocr.filter, frame_height=frame_height)
    subtitles = events_to_subtitles(kept)

    if chunk.offset:
        subtitles = shift_timeline(subtitles, chunk.offset)

    return subtitles


def _run_translation(srt_path: Path, cfg: Config) -> None:
    """Optional translation step, lazily imported."""
    from whisper_subtitle.translation.translator import translate_srt_file

    translated = translate_srt_file(srt_path, cfg.translation)
    log.info("Translated SRT: %s", translated)
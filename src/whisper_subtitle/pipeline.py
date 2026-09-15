"""Orchestrate the video → SRT pipeline."""

import logging
from pathlib import Path

from whisper_subtitle.config import Config
from whisper_subtitle.exceptions import PipelineError
from whisper_subtitle.media.ffmpeg import Chunk, extract_audio, split_video
from whisper_subtitle.models import OcrEvent, Subtitle
from whisper_subtitle.ocr.engine import PaddleOcrEngine
from whisper_subtitle.ocr.filter import (
    events_to_subtitles,
    filter_events,
    filter_repeated_texts,
)
from whisper_subtitle.ocr.grouper import group_detections
from whisper_subtitle.ocr.runner import sample_and_detect
from whisper_subtitle.subtitles.merger import merge_vad_and_whisper
from whisper_subtitle.subtitles.srt import render_srt
from whisper_subtitle.subtitles.timeline import shift_ocr_events, shift_timeline
from whisper_subtitle.transcription.vad import run_vad
from whisper_subtitle.transcription.whisper_runner import run_whisper

log = logging.getLogger(__name__)


def process_video(input_path: Path, cfg: Config, *, translate: bool = False) -> Path:
    if not cfg.transcription.enabled and not cfg.ocr.enabled:
        raise PipelineError(
            "Both transcription and OCR are disabled. Nothing to do."
        )

    input_path = input_path.resolve()
    output_dir = input_path.parent / f"{input_path.stem}_chunks"

    log.info("Splitting %s", input_path.name)
    chunks = split_video(input_path, output_dir, cfg.chunk_length_seconds)
    log.info("Processing %d chunk(s)", len(chunks))

    if cfg.transcription.enabled and cfg.ocr.enabled:
        raise PipelineError(
            "Merged mode (transcription + OCR) is not implemented yet."
        )

    if cfg.ocr.enabled:
        all_subs = _process_all_ocr(chunks, cfg)
    else:
        all_subs = _process_all_whisper(chunks, cfg)

    final_srt = output_dir / f"{input_path.stem}.srt"
    final_srt.write_text(render_srt(all_subs), encoding="utf-8")
    log.info("Wrote %s (%d subtitles)", final_srt.name, len(all_subs))

    if translate:
        _run_translation(final_srt, cfg)

    return final_srt


def _process_all_whisper(chunks: list[Chunk], cfg: Config) -> list[Subtitle]:
    all_subs: list[Subtitle] = []
    for i, chunk in enumerate(chunks, start=1):
        log.info("Chunk %d/%d: %s", i, len(chunks), chunk.path.name)
        all_subs.extend(_process_chunk_whisper(chunk, cfg))
    return all_subs


def _process_all_ocr(chunks: list[Chunk], cfg: Config) -> list[Subtitle]:
    engine = _build_ocr_engine(cfg)

    all_events: list[OcrEvent] = []
    for i, chunk in enumerate(chunks, start=1):
        log.info("Chunk %d/%d: %s", i, len(chunks), chunk.path.name)
        all_events.extend(_ocr_chunk_events(chunk, cfg, engine))

    log.info("Running global filters across %d events", len(all_events))
    kept = filter_repeated_texts(all_events, cfg.ocr.filter.max_text_repeats)

    if cfg.ocr.llm_filter.enabled:
        if not cfg.translation.api_key:
            raise PipelineError(
                "LLM filter requires DEEPSEEK_API_KEY in .env"
            )
        from whisper_subtitle.ocr.llm_filter import filter_events_with_llm
        kept = filter_events_with_llm(
            kept,
            cfg=cfg.ocr.llm_filter,
            api_key=cfg.translation.api_key,
            base_url=cfg.translation.base_url,
        )

    return events_to_subtitles(kept)


def _build_ocr_engine(cfg: Config) -> PaddleOcrEngine:
    log.info("Loading OCR engine")
    return PaddleOcrEngine(
        det_model=cfg.ocr.det_model,
        rec_model=cfg.ocr.rec_model,
        dict_file=cfg.ocr.dict_file,
        providers=list(cfg.ocr.providers),
        rec_threshold=cfg.ocr.rec_confidence,
    )


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


def _ocr_chunk_events(
    chunk: Chunk,
    cfg: Config,
    engine: PaddleOcrEngine,
) -> list[OcrEvent]:
    """Run OCR on one chunk and return filtered events shifted to the global timeline."""
    detections, frame_height = sample_and_detect(chunk.path, engine, cfg.ocr)
    events = group_detections(detections)
    kept = filter_events(events, cfg.ocr.filter, frame_height=frame_height)
    if chunk.offset:
        kept = shift_ocr_events(kept, chunk.offset)
    return kept


def _run_translation(srt_path: Path, cfg: Config) -> None:
    """Optional translation step, lazily imported."""
    from whisper_subtitle.translation.translator import translate_srt_file

    translated = translate_srt_file(srt_path, cfg.translation)
    log.info("Translated SRT: %s", translated)
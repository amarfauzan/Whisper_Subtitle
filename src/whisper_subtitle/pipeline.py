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
from whisper_subtitle.transcription.vad_transcribe import transcribe_by_vad_segments
from whisper_subtitle.ocr.grouper import group_detections
from whisper_subtitle.ocr.runner import sample_and_detect
from whisper_subtitle.subtitles.merger import merge_vad_and_whisper
from whisper_subtitle.subtitles.srt import render_srt
from whisper_subtitle.subtitles.timeline import shift_ocr_events, shift_timeline
from whisper_subtitle.transcription.vad import run_vad
from concurrent.futures import ThreadPoolExecutor
from whisper_subtitle.transcription.whisper_backend import (
    WhisperBackend,
    make_whisper_backend,
)
from whisper_subtitle.transcription.vad_transcribe import transcribe_vad_segments

log = logging.getLogger(__name__)


def process_video(input_path: Path, cfg: Config, *, translate: bool = False) -> Path:
    if not cfg.transcription.enabled and not cfg.ocr.enabled:
        raise PipelineError("Both transcription and OCR are disabled. Nothing to do.")

    input_path = input_path.resolve()
    output_dir = input_path.parent / f"{input_path.stem}_chunks"

    log.info("Splitting %s", input_path.name)
    chunks = split_video(input_path, output_dir, cfg.chunk_length_seconds)
    log.info("Processing %d chunk(s)", len(chunks))

    if cfg.transcription.enabled and cfg.ocr.enabled:
        all_subs = _process_all_merged(chunks, cfg)
    elif cfg.ocr.enabled:
        all_subs = _process_all_ocr(chunks, cfg)
    else:
        all_subs = _process_all_whisper(chunks, cfg)


    final_srt = output_dir / f"{input_path.stem}.srt"
    final_srt.write_text(render_srt(all_subs), encoding="utf-8")
    log.info("Wrote %s (%d subtitles)", final_srt.name, len(all_subs))

    if translate:
        _run_translation(all_subs, final_srt, cfg)

    return final_srt


def _process_all_merged(chunks: list[Chunk], cfg: Config) -> list[Subtitle]:
    engine = _build_ocr_engine(cfg)

    if cfg.merge.parallel:
        log.info("Running whisper and OCR in parallel")
        with ThreadPoolExecutor(max_workers=2) as pool:
            whisper_future = pool.submit(_run_whisper_over_chunks, chunks, cfg)
            ocr_future = pool.submit(_run_ocr_over_chunks, chunks, cfg, engine)
            whisper_subs = whisper_future.result()
            ocr_events = ocr_future.result()
    else:
        log.info("Running whisper and OCR sequentially")
        whisper_subs = _run_whisper_over_chunks(chunks, cfg)
        ocr_events = _run_ocr_over_chunks(chunks, cfg, engine)

    log.info(
        "Chunk processing complete: %d whisper subs, %d OCR events",
        len(whisper_subs), len(ocr_events),
    )

    ocr_events = filter_repeated_texts(ocr_events, cfg.ocr.filter.max_text_repeats)
    ocr_events = _run_llm_filter_if_enabled(ocr_events, cfg)

    from whisper_subtitle.subtitles.merge import merge_whisper_ocr
    return merge_whisper_ocr(whisper_subs, ocr_events, cfg.merge)

def _run_whisper_over_chunks(
    chunks: list[Chunk],
    cfg: Config,
) -> list[Subtitle]:
    if cfg.whisper.ensemble.enabled:
        from whisper_subtitle.transcription.ensemble_pipeline import (
            process_chunks_with_ensemble,
        )
        return process_chunks_with_ensemble(chunks, cfg)

    with make_whisper_backend(cfg.whisper) as backend:
        all_subs: list[Subtitle] = []
        for i, chunk in enumerate(chunks, start=1):
            log.info("Whisper chunk %d/%d: %s", i, len(chunks), chunk.path.name)
            all_subs.extend(_process_chunk_whisper(chunk, cfg, backend))
    return all_subs


def _run_ocr_over_chunks(
    chunks: list[Chunk],
    cfg: Config,
    engine: PaddleOcrEngine,
) -> list[OcrEvent]:
    """Process every chunk through the OCR pipeline.

    Only runs the heavy work: inference, grouping, per-event filter.
    The repeat filter and LLM filter run once, globally, after both
    threads join — that's why they live in _process_all_merged and not
    here.
    """
    all_events: list[OcrEvent] = []
    for i, chunk in enumerate(chunks, start=1):
        log.info("OCR chunk %d/%d: %s", i, len(chunks), chunk.path.name)
        all_events.extend(_ocr_chunk_events(chunk, cfg, engine))
    return all_events


def _process_all_ocr(chunks: list[Chunk], cfg: Config) -> list[Subtitle]:
    engine = _build_ocr_engine(cfg)

    all_events: list[OcrEvent] = []
    for i, chunk in enumerate(chunks, start=1):
        log.info("Chunk %d/%d: %s", i, len(chunks), chunk.path.name)
        all_events.extend(_ocr_chunk_events(chunk, cfg, engine))

    log.info("Running global filters across %d events", len(all_events))
    kept = filter_repeated_texts(all_events, cfg.ocr.filter.max_text_repeats)
    kept = _run_llm_filter_if_enabled(kept, cfg)
    return events_to_subtitles(kept)


def _run_llm_filter_if_enabled(
    events: list[OcrEvent],
    cfg: Config,
) -> list[OcrEvent]:
    if not cfg.ocr.llm_filter.enabled:
        return events
    if not cfg.translation.api_key:
        raise PipelineError("LLM filter requires DEEPSEEK_API_KEY in .env")
    from whisper_subtitle.ocr.llm_filter import filter_events_with_llm

    return filter_events_with_llm(
        events,
        cfg=cfg.ocr.llm_filter,
        api_key=cfg.translation.api_key,
        base_url=cfg.translation.base_url,
    )


def _build_ocr_engine(cfg: Config) -> PaddleOcrEngine:
    log.info("Loading OCR engine")
    return PaddleOcrEngine(
        det_model=cfg.ocr.det_model,
        rec_model=cfg.ocr.rec_model,
        dict_file=cfg.ocr.dict_file,
        providers=list(cfg.ocr.providers),
        rec_threshold=cfg.ocr.rec_confidence,
    )





def _process_all_whisper(chunks: list[Chunk], cfg: Config) -> list[Subtitle]:
    if cfg.whisper.ensemble.enabled:
        from whisper_subtitle.transcription.ensemble_pipeline import (
            process_chunks_with_ensemble,
        )
        return process_chunks_with_ensemble(chunks, cfg)

    with make_whisper_backend(cfg.whisper) as backend:
        all_subs: list[Subtitle] = []
        for i, chunk in enumerate(chunks, start=1):
            log.info("Whisper chunk %d/%d: %s", i, len(chunks), chunk.path.name)
            all_subs.extend(_process_chunk_whisper(chunk, cfg, backend))
    return all_subs

def _process_chunk_whisper(
    chunk: Chunk,
    cfg: Config,
    backend: WhisperBackend,
) -> list[Subtitle]:
    audio = extract_audio(chunk.path)
    vad_segments = run_vad(audio, cfg.vad)

    if cfg.whisper.strategy == "vad_segments":
        whisper_segments = transcribe_vad_segments(
            audio, vad_segments, backend, cfg.whisper
        )
    else:
        whisper_segments = backend.transcribe(audio).segments

    subtitles = merge_vad_and_whisper(
        vad_segments,
        whisper_segments,
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


def _run_translation(
    subtitles: list[Subtitle],
    source_srt: Path,
    cfg: Config,
) -> None:
    """Optional translation step, lazily imported."""
    from whisper_subtitle.translation.translator import translate_subtitles

    translated = translate_subtitles(subtitles, cfg.translation, source_srt)
    log.info("Translated SRT: %s", translated)
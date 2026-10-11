"""Orchestrate the video → SRT pipeline."""

import logging
from pathlib import Path
import re

from whisper_subtitle.config import Config
from whisper_subtitle.exceptions import PipelineError
from whisper_subtitle.media.ffmpeg import Chunk, extract_audio, split_video
from whisper_subtitle.models import OcrDetection, OcrEvent, Subtitle
from whisper_subtitle.ocr.engine import PaddleOcrEngine
from whisper_subtitle.ocr.filter import (
    events_to_subtitles,
    filter_events,
    filter_repeated_texts,
    pad_event_timestamps,
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
from whisper_subtitle.ocr.frame_merge import merge_frame_detections
from whisper_subtitle.ocr.filter import find_repeated_texts
from whisper_subtitle.subtitles.merge import merge_whisper_ocr
from whisper_subtitle.persistence import (
    save_diagnostic_report,
    save_ocr_events,
    save_raw_whisper_chunk,
    save_transcription,
)
from whisper_subtitle.media.vocal_separation import separate_vocals


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

    with ThreadPoolExecutor(max_workers=2) as pool:
        whisper_future = pool.submit(_run_whisper_over_chunks, chunks, cfg)
        ocr_future = pool.submit(_run_ocr_over_chunks, chunks, cfg, engine)
        whisper_subs = whisper_future.result()
        ocr_events = ocr_future.result()

    log.info(
        "Chunk processing complete: %d whisper subs, %d OCR events",
        len(whisper_subs), len(ocr_events),
    )

    return merge_whisper_ocr(whisper_subs, ocr_events, cfg.merge)



def _run_whisper_over_chunks(
    chunks: list[Chunk],
    cfg: Config,
) -> list[Subtitle]:

    #if ensemble run this
    if cfg.whisper.ensemble.enabled:
        from whisper_subtitle.transcription.ensemble_pipeline import (
            process_chunks_with_ensemble,
        )
        return process_chunks_with_ensemble(chunks, cfg)
    
    #run whisper primary model only in server mode
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
    """Used by merged mode. Calls the same pipeline as OCR-only."""
    return _run_ocr_pipeline(chunks, cfg, engine)

def _process_all_ocr(chunks: list[Chunk], cfg: Config) -> list[Subtitle]:
    engine = _build_ocr_engine(cfg)
    kept = _run_ocr_pipeline(chunks, cfg, engine)
    return events_to_subtitles(kept)

def _run_ocr_pipeline(
    chunks: list[Chunk],
    cfg: Config,
    engine: PaddleOcrEngine,
) -> list[OcrEvent]:
    """Run the full OCR pipeline. Returns final filtered events.

    Used by both OCR-only mode and merged mode. The pipeline:

        sample per chunk
          -> blocklist
          -> counting grouper
          -> global repeat-text detection
          -> drop repeated detections
          -> frame-level spatial merge
          -> final grouper
          -> per-event filter
          -> optional LLM filter
    """
    all_dets: list[OcrDetection] = []
    counting_events: list[OcrEvent] = []
    frame_height = 0

    for i, chunk in enumerate(chunks, start=1):
        log.info("OCR chunk %d/%d: %s", i, len(chunks), chunk.path.name)
        dets, events_a, h = _ocr_chunk_pass_a(chunk, cfg, engine)
        all_dets.extend(dets)
        counting_events.extend(events_a)
        frame_height = h

    log.info(
        "Counting pass complete: %d detections, %d events",
        len(all_dets), len(counting_events),
    )

    #filter repeated text
    repeated = find_repeated_texts(
        counting_events, cfg.ocr.filter.max_text_repeats
    )
    log.info("Repeat filter: %d unique texts flagged", len(repeated))

    surviving = [d for d in all_dets if d.text not in repeated]
    log.info(
        "After repeat-text drop: %d detections (dropped %d)",
        len(surviving), len(all_dets) - len(surviving),
    )

    merged = merge_frame_detections(surviving)

    events = group_detections(merged)
    log.info("Grouper: %d events", len(events))

    kept = filter_events(events, cfg.ocr.filter, frame_height=frame_height)
    log.info("Per-event filter: %d events kept", len(kept))

    kept = _run_llm_filter_if_enabled(kept, cfg)

    # Pad timestamps to compensate for 2 fps sampling phase
    kept = pad_event_timestamps(kept, cfg.ocr.filter.timestamp_padding)
    if cfg.ocr.filter.timestamp_padding > 0:
        log.info(
            "Applied %.2fs timestamp padding to %d events",
            cfg.ocr.filter.timestamp_padding, len(kept),
        )
        # Save combined OCR output for diagnostics
    # Save combined OCR output for diagnostics
    if cfg.whisper.diagnostics.save_intermediates and chunks:
        output_dir = chunks[0].path.parent
        save_ocr_events(
            kept,
            output_dir / "ocr.json",
            chunk="all",
            frame_height=frame_height,
            stage="final",
        )

    return kept



def _ocr_chunk_pass_a(
    chunk: Chunk,
    cfg: Config,
    engine: PaddleOcrEngine,
) -> tuple[list[OcrDetection], list[OcrEvent], int]:
    """One chunk's OCR pass: sample, blocklist, group, shift to global time."""
    detections, frame_height = sample_and_detect(chunk.path, engine, cfg.ocr)

    # Blocklist filter on raw detections
    patterns = [re.compile(p) for p in cfg.ocr.filter.blocklist_patterns]
    blocklist = {t.lower() for t in cfg.ocr.filter.blocklist_texts}

    # apply the blocklist on detections- append to filered list
    filtered = []
    for d in detections:
        text = d.text.strip()
        if not text:
            continue
        if text.lower() in blocklist:
            continue
        if any(p.fullmatch(text) for p in patterns):
            continue
        filtered.append(d)

    # Counting grouper — events here feed only the repeat filter
    counting_events = group_detections(filtered)

    # Shift onto global timeline
    if chunk.offset:
        filtered = [
            OcrDetection(
                frame_time=d.frame_time + chunk.offset,
                box=d.box, text=d.text,
                angle=d.angle, confidence=d.confidence,
            )
            for d in filtered
        ]
        counting_events = shift_ocr_events(counting_events, chunk.offset)

    return filtered, counting_events, frame_height

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

def _process_chunk_whisper(chunk, cfg, backend):
    audio = extract_audio(chunk.path)

    #run with vocal remover 
    if cfg.uvr.enabled:
        audio = separate_vocals(audio, cfg.uvr)

    vad_segments = run_vad(audio, cfg.vad)

    # Clear raw accumulator for this chunk
    if hasattr(backend, "raw_responses"):
        backend.raw_responses.clear()

    if cfg.whisper.strategy == "vad_segments":
        whisper_segments = transcribe_by_vad_segments(
            audio, vad_segments, backend, cfg.whisper
        )
    else:
        whisper_segments = backend.transcribe(audio).segments

    # Existing: save parsed
    if cfg.whisper.diagnostics.save_intermediates:
        save_transcription(
            whisper_segments,
            chunk.path.parent / f"{chunk.path.stem}.primary.json",
            source="primary",
            model=cfg.whisper.model.name,
            chunk=chunk.path.name,
        )

    # NEW: save raw server responses
    if (
        cfg.whisper.diagnostics.save_raw_whisper
        and hasattr(backend, "raw_responses")
        and backend.raw_responses
    ):
        save_raw_whisper_chunk(
            backend.raw_responses,
            chunk.path.parent / f"{chunk.path.stem}.primary.raw.json",
            source="primary",
            model=cfg.whisper.model.name,
            chunk=chunk.path.name,
        )


    subtitles = merge_vad_and_whisper(
        vad_segments,
        whisper_segments,
        max_merge_duration=cfg.vad.max_merge_duration,
    )
    if chunk.offset:
        subtitles = shift_timeline(subtitles, chunk.offset)
    return subtitles


def _run_translation(
    subtitles: list[Subtitle],
    source_srt: Path,
    cfg: Config,
) -> None:
    """Optional translation step, lazily imported."""
    from whisper_subtitle.translation.translator import translate_subtitles

    translated = translate_subtitles(subtitles, cfg.translation, source_srt)
    log.info("Translated SRT: %s", translated)
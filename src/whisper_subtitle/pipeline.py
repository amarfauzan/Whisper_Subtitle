"""Orchestrate the video → SRT pipeline."""

import logging
import re
from pathlib import Path

from whisper_subtitle.config import Config
from whisper_subtitle.exceptions import PipelineError
from whisper_subtitle.media.audio import audio_duration
from whisper_subtitle.media.ffmpeg import Chunk, extract_audio, split_video
from whisper_subtitle.media.vocal_separation import separate_vocals
from whisper_subtitle.models import OcrDetection, OcrEvent, Subtitle, VadSegment
from whisper_subtitle.ocr.engine import PaddleOcrEngine
from whisper_subtitle.ocr.filter import (
    events_to_subtitles,
    filter_events,
    find_repeated_texts,
    normalize_repeat_text,
    pad_event_timestamps,
)
from whisper_subtitle.ocr.frame_merge import merge_frame_detections
from whisper_subtitle.ocr.grouper import group_detections
from whisper_subtitle.ocr.runner import sample_and_detect
from whisper_subtitle.persistence import (
    save_ocr_events,
    save_raw_whisper_chunk,
    save_transcription,
)
from whisper_subtitle.subtitles.merge import merge_whisper_ocr
from whisper_subtitle.subtitles.merger import merge_vad_and_whisper
from whisper_subtitle.subtitles.split import split_long_subtitles
from whisper_subtitle.subtitles.srt import render_srt
from whisper_subtitle.subtitles.timeline import shift_ocr_events, shift_timeline
from whisper_subtitle.transcription.vad import run_vad, silence_intervals
from whisper_subtitle.transcription.vad_transcribe import transcribe_vad_segments
from whisper_subtitle.transcription.whisper_backend import make_whisper_backend

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

    # Shape the final cues: server mode never applies whisper-cli's line
    # splitting, so over-long cues are chopped here.
    all_subs = split_long_subtitles(
        all_subs,
        max_chars=cfg.subtitles.max_chars_per_cue,
        max_duration=cfg.subtitles.max_cue_duration,
        min_duration=cfg.subtitles.min_cue_duration,
    )

    final_srt = output_dir / f"{input_path.stem}.srt"
    final_srt.write_text(render_srt(all_subs), encoding="utf-8")
    log.info("Wrote %s (%d subtitles)", final_srt.name, len(all_subs))

    if translate:
        if not all_subs:
            log.warning("No subtitles produced; skipping translation.")
        else:
            _run_translation(all_subs, final_srt, cfg)

    return final_srt


def _process_all_merged(chunks: list[Chunk], cfg: Config) -> list[Subtitle]:
    engine = _build_ocr_engine(cfg)

    # Whisper runs first so its VAD speech boundaries are available to
    # tell OCR which (silence) regions actually need scanning.
    whisper_subs, vad_by_chunk = _run_whisper_over_chunks(chunks, cfg)
    ocr_events = _run_ocr_over_chunks(chunks, cfg, engine, vad_by_chunk)

    log.info(
        "Chunk processing complete: %d whisper subs, %d OCR events",
        len(whisper_subs), len(ocr_events),
    )

    return merge_whisper_ocr(whisper_subs, ocr_events, cfg.merge)



def _run_whisper_over_chunks(
    chunks: list[Chunk],
    cfg: Config,
) -> tuple[list[Subtitle], dict[Path, list[VadSegment]]]:
    """Transcribe every chunk and return (subtitles, VAD per chunk)."""

    if cfg.whisper.ensemble.enabled:
        from whisper_subtitle.transcription.ensemble_pipeline import (
            process_chunks_with_ensemble,
        )
        return process_chunks_with_ensemble(chunks, cfg)

    with make_whisper_backend(cfg.whisper) as backend:
        all_subs: list[Subtitle] = []
        vad_by_chunk: dict[Path, list[VadSegment]] = {}
        for i, chunk in enumerate(chunks, start=1):
            log.info("Whisper chunk %d/%d: %s", i, len(chunks), chunk.path.name)
            subs, vad = _process_chunk_whisper(chunk, cfg, backend)
            all_subs.extend(subs)
            vad_by_chunk[chunk.path] = vad
    return all_subs, vad_by_chunk


def _run_ocr_over_chunks(
    chunks: list[Chunk],
    cfg: Config,
    engine: PaddleOcrEngine,
    vad_by_chunk: dict[Path, list[VadSegment]] | None = None,
) -> list[OcrEvent]:
    """Run OCR for merged mode.

    With `ocr_silence_only` (the default) and VAD available, OCR only
    scans the non-speech regions — the parts that can actually become
    gap-fills — which is both faster and free of ASR duplication.
    Otherwise the whole video is scanned.
    """
    keep_by_chunk: dict[Path, list[tuple[float, float]]] | None = None
    if cfg.merge.ocr_silence_only and vad_by_chunk is not None:
        keep_by_chunk = _silence_intervals_by_chunk(chunks, cfg, vad_by_chunk)

    return _run_ocr_pipeline(
        chunks, cfg, engine, keep_intervals_by_chunk=keep_by_chunk
    )


def _silence_intervals_by_chunk(
    chunks: list[Chunk],
    cfg: Config,
    vad_by_chunk: dict[Path, list[VadSegment]],
) -> dict[Path, list[tuple[float, float]]]:
    """Map each chunk to its non-speech time ranges (chunk-local)."""
    keep: dict[Path, list[tuple[float, float]]] = {}
    margin = cfg.merge.speech_margin_seconds
    for chunk in chunks:
        audio = extract_audio(chunk.path)
        if cfg.uvr.enabled:
            audio = separate_vocals(audio, cfg.uvr)
        duration = audio_duration(audio)
        vad = vad_by_chunk.get(chunk.path, [])
        intervals = silence_intervals(vad, duration, margin)
        keep[chunk.path] = intervals
        log.info(
            "OCR silence regions for %s: %d (%.1fs of %.1fs)",
            chunk.path.name, len(intervals),
            sum(e - s for s, e in intervals), duration,
        )
    return keep

def _process_all_ocr(chunks: list[Chunk], cfg: Config) -> list[Subtitle]:
    engine = _build_ocr_engine(cfg)
    kept = _run_ocr_pipeline(chunks, cfg, engine)
    return events_to_subtitles(kept)

def _run_ocr_pipeline(
    chunks: list[Chunk],
    cfg: Config,
    engine: PaddleOcrEngine,
    keep_intervals_by_chunk: dict[Path, list[tuple[float, float]]] | None = None,
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

    When `keep_intervals_by_chunk` is given, only those time ranges are
    sampled (merged mode scans silence only).
    """
    all_dets: list[OcrDetection] = []
    counting_events: list[OcrEvent] = []
    frame_heights: set[int] = set()

    # Compile blocklist patterns once instead of once per chunk.
    patterns = [re.compile(p) for p in cfg.ocr.filter.blocklist_patterns]
    blocklist = {t.lower() for t in cfg.ocr.filter.blocklist_texts}

    for i, chunk in enumerate(chunks, start=1):
        log.info("OCR chunk %d/%d: %s", i, len(chunks), chunk.path.name)
        intervals = None
        if keep_intervals_by_chunk is not None:
            intervals = keep_intervals_by_chunk.get(chunk.path)
        dets, events_a, h = _ocr_chunk_pass_a(
            chunk, cfg, engine, patterns, blocklist, intervals
        )
        all_dets.extend(dets)
        counting_events.extend(events_a)
        frame_heights.add(h)

    # All chunks come from one video, so heights should agree. Guard
    # against surprising resolutions instead of silently using the last.
    frame_height = max(frame_heights) if frame_heights else 0
    if len(frame_heights) > 1:
        log.warning(
            "Chunks report differing frame heights %s; using %d for "
            "box-height ratios",
            sorted(frame_heights), frame_height,
        )

    log.info(
        "Counting pass complete: %d detections, %d events",
        len(all_dets), len(counting_events),
    )

    # Filter repeated text. Comparison uses the same normalization as
    # the grouper (case-folded, whitespace-stripped) so "MINAMI" and
    # "minami " land in the same bucket. Only persistent repetitions
    # (watermarks, nametags) are flagged — burst repeats are real
    # dialogue and stay.
    repeated = find_repeated_texts(
        counting_events,
        cfg.ocr.filter.max_text_repeats,
        cfg.ocr.filter.repeat_span_seconds,
    )
    log.info("Repeat filter: %d unique texts flagged", len(repeated))

    surviving = [
        d for d in all_dets if normalize_repeat_text(d.text) not in repeated
    ]
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
    blocklist_patterns: list[re.Pattern[str]],
    blocklist: set[str],
    keep_intervals: list[tuple[float, float]] | None = None,
) -> tuple[list[OcrDetection], list[OcrEvent], int]:
    """One chunk's OCR pass: sample, blocklist, group, shift to global time."""
    detections, frame_height = sample_and_detect(
        chunk.path, engine, cfg.ocr, keep_intervals=keep_intervals
    )

    # Apply the blocklist on detections — append to filtered list
    filtered = []
    for d in detections:
        text = d.text.strip()
        if not text:
            continue
        if text.lower() in blocklist:
            continue
        if any(p.fullmatch(text) for p in blocklist_patterns):
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
    subs, _ = _run_whisper_over_chunks(chunks, cfg)
    return subs

def _process_chunk_whisper(
    chunk: Chunk,
    cfg: Config,
    backend,
) -> tuple[list[Subtitle], list[VadSegment]]:
    audio = extract_audio(chunk.path)

    # Optional vocal separation, applied before VAD so both the VAD
    # boundaries and the transcription use the vocals-only audio.
    if cfg.uvr.enabled:
        audio = separate_vocals(audio, cfg.uvr)

    vad_segments = run_vad(audio, cfg.vad)

    # Clear raw accumulator for this chunk
    if hasattr(backend, "raw_responses"):
        backend.raw_responses.clear()

    if cfg.whisper.strategy == "vad_segments":
        whisper_segments = transcribe_vad_segments(
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
    return subtitles, vad_segments


def _run_translation(
    subtitles: list[Subtitle],
    source_srt: Path,
    cfg: Config,
) -> None:
    """Optional translation step, lazily imported."""
    from whisper_subtitle.translation.translator import translate_subtitles

    translated = translate_subtitles(subtitles, cfg.translation, source_srt)
    log.info("Translated SRT: %s", translated)
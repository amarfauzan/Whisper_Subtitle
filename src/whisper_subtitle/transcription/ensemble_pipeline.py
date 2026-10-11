"""Orchestrate the ensemble whisper pipeline: batisay + large-v3.

Runs the two models sequentially — one server at a time — to keep VRAM
usage bounded. For each chunk, both models transcribe the same VAD
segments; the merge happens afterward using large-v3's timing and
batisay's text, with optional LLM adjudication for disagreements.

Sequential phases:
  1. Primary (batisay) → transcribe all chunks, cache VAD segments
  2. Secondary (large-v3) → transcribe all chunks, reuse cached VAD
  3. Merge per chunk (V3 timing + BAT text, LLM arbitrates)
"""

import logging
from dataclasses import replace
from pathlib import Path

from whisper_subtitle.config import Config
from whisper_subtitle.exceptions import PipelineError
from whisper_subtitle.media.audio import audio_duration
from whisper_subtitle.media.ffmpeg import Chunk, extract_audio
from whisper_subtitle.media.vocal_separation import separate_vocals
from whisper_subtitle.models import Subtitle, VadSegment, WhisperSegment
from whisper_subtitle.persistence import (
    save_diagnostic_report,
    save_transcription,
)
from whisper_subtitle.subtitles.timeline import shift_timeline
from whisper_subtitle.transcription.diagnostics import diagnose_transcription
from whisper_subtitle.transcription.ensemble import merge_ensemble
from whisper_subtitle.transcription.llm_ensemble import adjudicate_requests
from whisper_subtitle.transcription.vad import run_vad
from whisper_subtitle.transcription.vad_transcribe import transcribe_vad_segments
from whisper_subtitle.transcription.whisper_backend import make_whisper_backend

log = logging.getLogger(__name__)

# How far a cue may be stretched past whisper's own end to fill a VAD
# speech region. Without a cap, one short line can occupy a whole
# multi-second region.
_MAX_VAD_EXTENSION = 1.0


def process_chunks_with_ensemble(
    chunks: list[Chunk],
    cfg: Config,
) -> list[Subtitle]:
    if not cfg.whisper.ensemble.enabled:
        raise RuntimeError(
            "process_chunks_with_ensemble called but ensemble.enabled is False"
        )
    if not cfg.whisper.ensemble.secondary_model.exists():
        raise RuntimeError(
            f"Secondary model not found: {cfg.whisper.ensemble.secondary_model}"
        )

    output_dir = chunks[0].path.parent if chunks else Path(".")
    primary_cache: dict[Path, tuple[list[VadSegment], list[WhisperSegment]]] = {}
    diagnostics: list[dict] = []

    # ------------------------------------------------------------------
    # Phase 1: primary (batisay)
    # ------------------------------------------------------------------
    log.info("Ensemble phase 1/2: primary model (%s)", cfg.whisper.model.name)
    with make_whisper_backend(cfg.whisper) as primary:
        for i, chunk in enumerate(chunks, start=1):
            log.info("Primary chunk %d/%d: %s", i, len(chunks), chunk.path.name)
            audio = extract_audio(chunk.path)

            # Run vocal separation before VAD so both the VAD boundaries
            # and the transcription use the vocals-only audio. Phase 2
            # reuses this same audio (separate_vocals caches) and these
            # same VAD segments, so both models see identical input.
            if cfg.uvr.enabled:
                audio = separate_vocals(audio, cfg.uvr)

            vad = run_vad(audio, cfg.vad)
            segs = transcribe_vad_segments(audio, vad, primary, cfg.whisper)

            primary_cache[chunk.path] = (vad, segs)

            if cfg.whisper.diagnostics.save_intermediates:
                save_transcription(
                    segs,
                    output_dir / f"{chunk.path.stem}.primary.json",
                    source="primary",
                    model=cfg.whisper.model.name,
                    chunk=chunk.path.name,
                )

            if cfg.whisper.diagnostics.enabled:
                _diagnose_and_maybe_abort(
                    segs, audio, chunk, output_dir, cfg, diagnostics,
                )

    # ------------------------------------------------------------------
    # Phase 2: secondary (large-v3)
    # ------------------------------------------------------------------
    secondary_cfg = replace(
        cfg.whisper,
        model=cfg.whisper.ensemble.secondary_model,
        server_port=cfg.whisper.ensemble.secondary_port,
    )
    secondary_cache: dict[Path, list[WhisperSegment]] = {}

    log.info(
        "Ensemble phase 2/2: secondary model (%s)",
        cfg.whisper.ensemble.secondary_model.name,
    )
    with make_whisper_backend(secondary_cfg) as secondary:
        for i, chunk in enumerate(chunks, start=1):
            log.info("Secondary chunk %d/%d: %s", i, len(chunks), chunk.path.name)
            audio = extract_audio(chunk.path)

            # Vocals were already separated during phase 1; this call
            # reuses the cached output (or re-runs if missing).
            if cfg.uvr.enabled:
                audio = separate_vocals(audio, cfg.uvr)

            vad, _ = primary_cache[chunk.path]
            segs = transcribe_vad_segments(audio, vad, secondary, secondary_cfg)
            secondary_cache[chunk.path] = segs

            if cfg.whisper.diagnostics.save_intermediates:
                save_transcription(
                    segs,
                    output_dir / f"{chunk.path.stem}.secondary.json",
                    source="secondary",
                    model=cfg.whisper.ensemble.secondary_model.name,
                    chunk=chunk.path.name,
                )

    # ------------------------------------------------------------------
    # Phase 3: merge
    # ------------------------------------------------------------------
    log.info("Ensemble phase 3/3: merging per chunk")
    all_subs: list[Subtitle] = []

    for i, chunk in enumerate(chunks, start=1):
        log.info("Merge chunk %d/%d: %s", i, len(chunks), chunk.path.name)
        vad, bat_segs = primary_cache[chunk.path]
        v3_segs = secondary_cache[chunk.path]

        result = merge_ensemble(bat_segs, v3_segs, cfg.whisper.ensemble)

        if cfg.whisper.ensemble.llm_adjudicate and result.pending:
            if not cfg.translation.api_key:
                log.warning(
                    "LLM adjudication enabled but DEEPSEEK_API_KEY not set; "
                    "keeping V3 text for %d disagreements", len(result.pending),
                )
            else:
                result.cues = adjudicate_requests(
                    requests=result.pending,
                    cues=result.cues,
                    cfg=cfg.whisper.ensemble,
                    api_key=cfg.translation.api_key,
                    base_url=cfg.translation.base_url,
                )

        clipped = _clip_to_vad(result.cues, vad)
        subs = [
            Subtitle(start=c.start, end=c.end, text=c.text)
            for c in clipped if c.text.strip()
        ]
        if chunk.offset:
            subs = shift_timeline(subs, chunk.offset)
        all_subs.extend(subs)

    # Save overall diagnostics summary
    if cfg.whisper.diagnostics.enabled and diagnostics:
        save_diagnostic_report(
            {
                "chunk_count": len(chunks),
                "chunks": diagnostics,
            },
            output_dir / f"{chunks[0].path.parent.name}.diagnostics.json",
        )

    log.info(
        "Ensemble complete: %d subtitles from %d chunks",
        len(all_subs), len(chunks),
    )
    return all_subs


def _diagnose_and_maybe_abort(
    segs: list[WhisperSegment],
    audio: Path,
    chunk: Chunk,
    output_dir: Path,
    cfg: Config,
    diagnostics: list[dict],
) -> None:
    duration = audio_duration(audio)
    result = diagnose_transcription(segs, duration)

    entry = {
        "chunk": chunk.path.name,
        "verdict": result.verdict,
        "reason": result.reason,
        "warnings": result.warnings,
        "metrics": result.metrics,
    }
    diagnostics.append(entry)

    if result.verdict == "abort":
        log.error(
            "Primary %s: %s", chunk.path.name, result.reason,
        )
        save_diagnostic_report(
            entry,
            output_dir / f"{chunk.path.stem}.diagnostic.json",
        )
        if cfg.whisper.diagnostics.abort_on_loop:
            raise PipelineError(
                f"Primary transcription abort on {chunk.path.name}: "
                f"{result.reason}"
            )
    elif result.verdict == "warning":
        for w in result.warnings:
            log.warning("Primary %s: %s", chunk.path.name, w)
        save_diagnostic_report(
            entry,
            output_dir / f"{chunk.path.stem}.diagnostic.json",
        )
    else:
        log.info(
            "Primary %s OK (%d segs, %.1f%% coverage)",
            chunk.path.name,
            result.metrics["segment_count"],
            result.metrics["coverage_ratio"] * 100,
        )

def _clip_to_vad(
    cues: list[WhisperSegment],
    vad_segments: list[VadSegment],
    *,
    max_extension: float = _MAX_VAD_EXTENSION,
) -> list[WhisperSegment]:
    """Clip each cue to the VAD speech region(s) it overlaps.

    Drops cues that don't overlap any VAD segment (whisper predicted
    speech in silence). A cue that spans multiple VAD segments is
    clipped to their union, but never extended more than
    `max_extension` seconds past whisper's own end — otherwise a short
    line like "진짜 꿀이네." ends up on screen for 9 seconds.

    Cues are also prevented from running into the next cue.
    """
    result: list[WhisperSegment] = []
    for cue in cues:
        overlapping = [
            v for v in vad_segments
            if min(cue.end, v.end) > max(cue.start, v.start)
        ]
        if not overlapping:
            continue

        clip_start = min(v.start for v in overlapping)
        clip_end = max(v.end for v in overlapping)
        new_start = max(cue.start, clip_start)
        new_end = min(cue.end + max_extension, clip_end)

        if new_end - new_start < 0.1:
            continue

        result.append(replace(cue, start=new_start, end=new_end))

    # Never let one cue's displayed text bleed into the next cue.
    result.sort(key=lambda c: c.start)
    for i in range(len(result) - 1):
        if result[i].end > result[i + 1].start:
            result[i] = replace(result[i], end=result[i + 1].start)
    return [c for c in result if c.end > c.start]
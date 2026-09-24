"""Save and load pipeline intermediates.

Every stage that produces a list of segments or events can be written
to disk for later inspection, diffing, or re-processing without
re-running whisper or OCR.
"""

import json
import logging
from pathlib import Path

from whisper_subtitle.models import OcrEvent, Subtitle, WhisperSegment
from whisper_subtitle.subtitles.srt import render_srt

log = logging.getLogger(__name__)


def save_transcription(
    segments: list[WhisperSegment],
    json_path: Path,
    *,
    source: str,
    model: str,
    chunk: str,
) -> None:
    """Write whisper segments to JSON + SRT.

    JSON is canonical; SRT is a convenience for playback comparison.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "model": model,
        "chunk": chunk,
        "segment_count": len(segments),
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in segments
        ],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    srt_path = json_path.with_suffix(".srt")
    srt_path.write_text(
        render_srt([Subtitle(s.start, s.end, s.text) for s in segments]),
        encoding="utf-8",
    )
    log.debug("Saved %s (%d segments) → %s", source, len(segments), json_path.name)


def load_transcription(json_path: Path) -> list[WhisperSegment]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [
        WhisperSegment(start=s["start"], end=s["end"], text=s["text"])
        for s in data["segments"]
    ]


def save_ocr_events(
    events: list[OcrEvent],
    json_path: Path,
    *,
    chunk: str,
    frame_height: int,
    stage: str = "final",
) -> None:
    """Write OCR events to JSON + SRT."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "ocr",
        "stage": stage,
        "chunk": chunk,
        "frame_height": frame_height,
        "event_count": len(events),
        "events": [
            {
                "start": ev.start,
                "end": ev.end,
                "box": list(ev.box),
                "text": ev.text,
                "confidence": ev.confidence,
                "frame_count": ev.frame_count,
            }
            for ev in events
        ],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    srt_path = json_path.with_suffix(".srt")
    srt_path.write_text(
        render_srt([Subtitle(ev.start, ev.end, ev.text) for ev in events]),
        encoding="utf-8",
    )
    log.debug("Saved OCR %s (%d events) → %s", stage, len(events), json_path.name)


def load_ocr_events(json_path: Path) -> list[OcrEvent]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [
        OcrEvent(
            start=ev["start"],
            end=ev["end"],
            box=tuple(ev["box"]),
            text=ev["text"],
            confidence=ev["confidence"],
            frame_count=ev["frame_count"],
        )
        for ev in data["events"]
    ]


def save_diagnostic_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.debug("Saved diagnostic report → %s", path.name)
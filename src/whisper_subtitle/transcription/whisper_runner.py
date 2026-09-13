"""Run the whisper-cli and parse its output."""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from whisper_subtitle.config import WhisperConfig
from whisper_subtitle.exceptions import WhisperError
from whisper_subtitle.models import WhisperSegment

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhisperResult:
    """Paths to the files whisper produced, plus the parsed segments."""

    segments: list[WhisperSegment]
    txt_path: Path
    json_path: Path
    srt_path: Path


def parse_whisper_json(json_path: Path) -> list[WhisperSegment]:
    """Parse a whisper.cpp JSON output file into WhisperSegment objects.

    whisper.cpp emits both `offsets` (milliseconds) and `timestamps`
    (HH:MM:SS,mmm strings) per segment. We prefer `offsets` — it's already
    numeric, so no parsing or rounding is involved.
    """
    if not json_path.exists():
        raise WhisperError(f"Whisper JSON not found: {json_path}")

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WhisperError(f"Invalid whisper JSON in {json_path}: {exc}") from exc

    raw_segments = data.get("transcription") or []
    if not raw_segments:
        raise WhisperError(f"No segments in whisper JSON: {json_path}")

    segments = []
    for item in raw_segments:
        try:
            start_ms = item["offsets"]["from"]
            end_ms = item["offsets"]["to"]
        except KeyError as exc:
            raise WhisperError(
                f"Whisper segment missing field {exc}: {item}"
            ) from exc

        text = item.get("text", "").strip()
        if not text:
            continue

        segments.append(
            WhisperSegment(
                start=start_ms / 1000.0,
                end=end_ms / 1000.0,
                text=text,
            )
        )

    if not segments:
        raise WhisperError(f"All whisper segments were empty: {json_path}")

    return segments


def run_whisper(audio_path: Path, cfg: WhisperConfig) -> WhisperResult:
    """Run whisper-cli on an audio file and return the parsed result.

    Writes three files next to the audio file (same stem):
        <stem>.txt, <stem>.json, <stem>.srt

    If those files already exist, skips the run and reuses them. This is
    what makes the pipeline resumable across long videos.
    """
    if not audio_path.exists():
        raise WhisperError(f"Audio file not found: {audio_path}")

    output_base = audio_path.with_suffix("")
    txt_path = output_base.with_suffix(".txt")
    json_path = output_base.with_suffix(".json")
    srt_path = output_base.with_suffix(".srt")

    if txt_path.exists() and json_path.exists() and srt_path.exists():
        log.info("Whisper output already exists, skipping: %s", audio_path.name)
        return WhisperResult(
            segments=parse_whisper_json(json_path),
            txt_path=txt_path,
            json_path=json_path,
            srt_path=srt_path,
        )

    command = [
        str(cfg.cli),
        "-m", str(cfg.model),
        "-l", cfg.language,
        "-bs", str(cfg.beam_size),
        "-bo", str(cfg.best_of),
        "-otxt",
        "-oj",
        "-osrt",
        "--vad",
        "-vm", str(cfg.vad_model),
        "-sow",  # split on word boundaries
        "--max-len", str(cfg.max_line_length),
        "-of", str(output_base),
        str(audio_path),
    ]

    log.info("Running whisper on %s", audio_path.name)
    log.debug("Whisper command: %s", " ".join(command))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise WhisperError(
            f"Whisper failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )

    for path in (txt_path, json_path, srt_path):
        if not path.exists():
            raise WhisperError(f"Whisper did not produce: {path}")

    return WhisperResult(
        segments=parse_whisper_json(json_path),
        txt_path=txt_path,
        json_path=json_path,
        srt_path=srt_path,
    )

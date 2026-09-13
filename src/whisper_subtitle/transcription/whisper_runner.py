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
    """Parsed segments plus the path to the JSON they came from."""

    segments: list[WhisperSegment]
    json_path: Path


def parse_whisper_json(json_path: Path) -> list[WhisperSegment]:
    """Parse a whisper.cpp JSON output file into WhisperSegment objects.

    Returns an empty list if whisper ran cleanly but produced no segments
    (for example, on an audio track with no speech). Raises WhisperError
    only if the file is missing, unreadable, or malformed.
    """
    if not json_path.exists():
        raise WhisperError(f"Whisper JSON not found: {json_path}")

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WhisperError(f"Invalid whisper JSON in {json_path}: {exc}") from exc

    raw_segments = data.get("transcription") or []

    segments: list[WhisperSegment] = []
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

    return segments


def run_whisper(audio_path: Path, cfg: WhisperConfig) -> WhisperResult:
    """Run whisper-cli on an audio file and return the parsed result.

    Produces one file next to the audio: <stem>.json.
    If it already exists, skips the run and reuses it — this is what makes
    the pipeline resumable across long videos.
    """
    if not audio_path.exists():
        raise WhisperError(f"Audio file not found: {audio_path}")

    output_base = audio_path.with_suffix("")
    json_path = output_base.with_suffix(".json")

    if json_path.exists():
        log.info("Whisper output already exists, skipping: %s", audio_path.name)
        return WhisperResult(
            segments=parse_whisper_json(json_path),
            json_path=json_path,
        )

    command = [
        str(cfg.cli),
        "-m", str(cfg.model),
        "-l", cfg.language,
        "-bs", str(cfg.beam_size),
        "-bo", str(cfg.best_of),
        "-oj",  # JSON only — SRT and TXT are derived downstream
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

    if not json_path.exists():
        raise WhisperError(f"Whisper did not produce: {json_path}")

    return WhisperResult(
        segments=parse_whisper_json(json_path),
        json_path=json_path,
    )
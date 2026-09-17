"""Run whisper-vad-speech-segments and parse its output."""

import logging
import re
import subprocess
from pathlib import Path

from whisper_subtitle.config import VadConfig
from whisper_subtitle.exceptions import VadError
from whisper_subtitle.models import VadSegment

log = logging.getLogger(__name__)


# Matches lines like:
#   VAD segment 42: start = 599.48, end = 600.67
_SEGMENT_RE = re.compile(
    r"VAD segment\s+(\d+):\s+"
    r"start\s*=\s*([\d.]+),\s*"
    r"end\s*=\s*([\d.]+)"
)

# Matches the summary line whisper-vad-speech-segments always prints:
#   whisper_vad_segments_from_probs: Final speech segments after filtering: 12
_COMPLETION_RE = re.compile(
    r"Final speech segments after filtering:\s*\d+"
)


def parse_vad_output(text: str) -> list[VadSegment]:
    """Parse the stderr output of whisper-vad-speech-segments.

    Returns VadSegment objects sorted by index.

    Raises VadError only if the executable did not produce a recognizable
    completion line — that means it likely crashed before finishing. A
    clean run that found zero segments returns an empty list.
    """
    segments = [
        VadSegment(
            index=int(m.group(1)),
            start=float(m.group(2)),
            end=float(m.group(3)),
        )
        for m in _SEGMENT_RE.finditer(text)
    ]

    if segments:
        return segments

    # No segments found. Did VAD actually complete?
    if _COMPLETION_RE.search(text):
        return []  # valid: no speech in this audio

    raise VadError(
        "VAD executable produced no segments and no completion line. "
        "It probably crashed — check the stderr above."
    )

def run_vad(audio_path: Path, cfg: VadConfig) -> list[VadSegment]:
    """Run the VAD executable and return the parsed segments."""
    if not audio_path.exists():
        raise VadError(f"Audio file not found: {audio_path}")

    command = [
        str(cfg.exe),
        "-f", str(audio_path),
        "-vm", str(cfg.model),
        "-vmsd", str(cfg.max_speech_duration),
    ]

    log.debug("Running VAD: %s", " ".join(command))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # whisper-vad-speech-segments writes segment info to stderr.
    if result.returncode != 0:
        raise VadError(
            f"VAD failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )

    return parse_vad_output(result.stderr)

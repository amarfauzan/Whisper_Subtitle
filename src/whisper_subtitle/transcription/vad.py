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


def parse_vad_output(text: str) -> list[VadSegment]:
    """Parse the stderr output of whisper-vad-speech-segments.

    Returns VadSegment objects sorted by index. Raises VadError if no
    segments are found — an empty result almost always means the binary
    failed silently, not that the audio was silent.
    """
    segments = [
        VadSegment(
            index=int(m.group(1)),
            start=float(m.group(2)),
            end=float(m.group(3)),
        )
        for m in _SEGMENT_RE.finditer(text)
    ]

    if not segments:
        raise VadError("No VAD segments found in the executable output.")

    return segments


def run_vad(audio_path: Path, cfg: VadConfig) -> list[VadSegment]:
    """Run the VAD executable and return the parsed segments."""
    if not audio_path.exists():
        raise VadError(f"Audio file not found: {audio_path}")

    command = [
        str(cfg.exe),
        "-f", str(audio_path),
        "-vm", str(cfg.model),
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

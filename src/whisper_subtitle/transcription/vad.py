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
        check=False,
    )

    # whisper-vad-speech-segments writes segment info to stderr.
    if result.returncode != 0:
        raise VadError(
            f"VAD failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )

    return parse_vad_output(result.stderr)


def silence_intervals(
    segments: list[VadSegment],
    duration: float,
    margin: float = 0.0,
) -> list[tuple[float, float]]:
    """Return the non-speech time ranges of ``[0, duration]``.

    Each speech segment is expanded by ``margin`` on both sides (so a
    caption at the very edge of speech is not half-sampled) and clamped
    to the audio bounds. Overlapping expanded segments are merged, then
    the gaps between them are returned.

    With no speech at all the whole duration is returned; with continuous
    speech an empty list is returned.
    """
    if not segments:
        return [(0.0, duration)]

    expanded: list[tuple[float, float]] = []
    for seg in segments:
        start = max(0.0, seg.start - margin)
        end = min(duration, seg.end + margin)
        if end > start:
            expanded.append((start, end))
    expanded.sort()

    merged: list[list[float]] = []
    for start, end in expanded:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps

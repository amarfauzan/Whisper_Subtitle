"""Slice a WAV file to a time range using the stdlib wave module.

Assumes 16-bit PCM, which is what our ffmpeg audio extraction produces.
Used by the VAD-segment transcription strategy to feed clean,
speech-only clips to the whisper backend.
"""

import logging
import wave
from pathlib import Path

from whisper_subtitle.exceptions import PipelineError

log = logging.getLogger(__name__)


def extract_clip(
    source_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_path: Path,
) -> None:
    """Write a WAV clip from source_path to output_path covering
    [start_seconds, end_seconds).

    Silently clamps to the source's actual bounds — no error if the
    requested range extends past the end of the file.

    Raises:
        PipelineError if source_path can't be read as a WAV.
    """
    if end_seconds <= start_seconds:
        raise PipelineError(
            f"extract_clip: end ({end_seconds}) must be > start ({start_seconds})"
        )

    try:
        with wave.open(str(source_path), "rb") as src:
            framerate = src.getframerate()
            nchannels = src.getnchannels()
            sampwidth = src.getsampwidth()
            nframes = src.getnframes()

            start_frame = max(0, int(start_seconds * framerate))
            end_frame = min(nframes, int(end_seconds * framerate))

            if end_frame <= start_frame:
                raise PipelineError(
                    f"extract_clip: empty range "
                    f"[{start_seconds}, {end_seconds}) in {source_path.name}"
                )

            src.setpos(start_frame)
            frames = src.readframes(end_frame - start_frame)
    except (wave.Error, OSError) as exc:
        raise PipelineError(
            f"extract_clip: failed to read {source_path}: {exc}"
        ) from exc
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as dst:
        dst.setnchannels(nchannels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)
        dst.writeframes(frames)

    log.debug(
        "Extracted clip: %.3fs → %.3fs (%d frames)",
        start_seconds, end_seconds, end_frame - start_frame,
    )
"""Thin wrappers around ffmpeg and ffprobe subprocesses."""

import csv
import logging
import subprocess
from pathlib import Path

from whisper_subtitle.exceptions import FFmpegError
from whisper_subtitle.models import Chunk

log = logging.getLogger(__name__)


def _run(command: list[str], *, description: str) -> None:
    """Run a subprocess and raise FFmpegError on non-zero exit."""
    log.debug("%s: %s", description, " ".join(command))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise FFmpegError(
            f"{description} failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )


def split_video(
    input_path: Path,
    output_dir: Path,
    chunk_seconds: int,
) -> list[Chunk]:
    """Split a video into stream-copy chunks of ~chunk_seconds each.

    Chunks land in output_dir as <stem>_000.mp4, <stem>_001.mp4, ...
    A segments.csv records each chunk's real start time in the original.

    If chunks + segments.csv already exist, reuses them (resume support).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    csv_path = output_dir / "segments.csv"
    pattern = output_dir / f"{stem}_%03d.mp4"

    existing = sorted(output_dir.glob(f"{stem}_*.mp4"))
    if existing and csv_path.exists():
        log.info("Reusing %d existing chunks", len(existing))
        return _read_chunks_from_csv(csv_path, output_dir)

    if existing and not csv_path.exists():
        raise FFmpegError(
            f"Found {len(existing)} chunks in {output_dir} but no segments.csv. "
            f"Delete the folder and re-run."
        )

    command = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-map", "0",
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-segment_list", str(csv_path),
        "-segment_list_type", "csv",
        "-reset_timestamps", "1",
        str(pattern),
    ]
    _run(command, description="ffmpeg split")

    if not csv_path.exists():
        raise FFmpegError(f"ffmpeg did not produce {csv_path.name}")

    chunks = _read_chunks_from_csv(csv_path, output_dir)
    log.info("Created %d chunks", len(chunks))
    return chunks


def _read_chunks_from_csv(csv_path: Path, output_dir: Path) -> list[Chunk]:
    """Read ffmpeg's segment_list CSV into Chunk objects.

    CSV format (per row): filename,start_time,end_time
    start_time/end_time are relative to the ORIGINAL video, not the chunk.
    """
    chunks: list[Chunk] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            filename = Path(row[0]).name
            offset = float(row[1])
            chunks.append(Chunk(path=output_dir / filename, offset=offset))
    if not chunks:
        raise FFmpegError(f"No entries in {csv_path}")
    return chunks


def extract_audio(video_path: Path) -> Path:
    """Extract mono 16 kHz PCM audio from a video into a .wav next to it.

    Skips if the .wav already exists (resume support).
    """
    audio_path = video_path.with_suffix(".wav")

    if audio_path.exists():
        log.info("Audio already exists, skipping: %s", audio_path.name)
        return audio_path

    command = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path),
    ]
    _run(command, description=f"ffmpeg extract audio from {video_path.name}")

    if not audio_path.exists():
        raise FFmpegError(f"ffmpeg did not produce {audio_path.name}")

    return audio_path

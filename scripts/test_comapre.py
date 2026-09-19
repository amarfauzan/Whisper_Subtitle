"""Compare batisay and large-v3 transcription on the same audio.

Runs both models on a short extract using identical VAD segments,
writes two SRT files, and prints an interleaved view for eyeballing.

Usage:
    uv run python scripts/compare_models.py video.mp4 \
        --batisay path/to/batisay.bin \
        --large-v3 path/to/large-v3.bin \
        --max-seconds 180

Note: stops any running whisper-server on the target ports first.
"""

import argparse
import logging
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from whisper_subtitle.config import load_config
from whisper_subtitle.models import Subtitle
from whisper_subtitle.subtitles.srt import render_srt
from whisper_subtitle.transcription.vad import run_vad
from whisper_subtitle.transcription.vad_transcribe import transcribe_vad_segments
from whisper_subtitle.transcription.whisper_backend import make_whisper_backend


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compare")


def extract_audio_chunk(video: Path, seconds: float, out: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video), "-t", str(seconds),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out),
        ],
        check=True,
        capture_output=True,
    )


def transcribe_with(
    audio: Path,
    vad_segments,
    cfg,
    model_path: Path,
    port: int,
    label: str,
):
    log.info("=" * 70)
    log.info("%s — %s", label, model_path.name)
    log.info("=" * 70)
    model_cfg = replace(cfg.whisper, model=model_path, server_port=port)
    t0 = time.perf_counter()
    with make_whisper_backend(model_cfg) as backend:
        segments = transcribe_vad_segments(audio, vad_segments, backend, model_cfg)
    log.info("%s: %d segments in %.1fs", label, len(segments), time.perf_counter() - t0)
    return segments


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--batisay", type=Path, required=True)
    p.add_argument("--large-v3", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=180.0)
    p.add_argument("--batisay-port", type=int, default=8080)
    p.add_argument("--large-v3-port", type=int, default=8081)
    args = p.parse_args()

    # Clean any orphaned servers
    subprocess.run(["pkill", "-f", "whisper-server"], capture_output=True)

    cfg = load_config()
    out_dir = args.video.parent / f"{args.video.stem}_cmp"
    out_dir.mkdir(exist_ok=True)

    audio = out_dir / "extract.wav"
    log.info("Extracting first %.0fs", args.max_seconds)
    extract_audio_chunk(args.video, args.max_seconds, audio)

    log.info("Running VAD (shared between both models)")
    vad_segments = run_vad(audio, cfg.vad)
    log.info("VAD: %d segments", len(vad_segments))

    batisay_segs = transcribe_with(
        audio, vad_segments, cfg, args.batisay, args.batisay_port, "BATISAY"
    )
    large_segs = transcribe_with(
        audio, vad_segments, cfg, args.large_v3, args.large_v3_port, "LARGE-V3"
    )

    batisay_srt = out_dir / "batisay.srt"
    large_srt = out_dir / "large_v3.srt"
    batisay_srt.write_text(
        render_srt([Subtitle(s.start, s.end, s.text) for s in batisay_segs]),
        encoding="utf-8",
    )
    large_srt.write_text(
        render_srt([Subtitle(s.start, s.end, s.text) for s in large_segs]),
        encoding="utf-8",
    )
    log.info("Wrote: %s", batisay_srt)
    log.info("Wrote: %s", large_srt)

    # Interleaved view
    print()
    print("=" * 100)
    print("INTERLEAVED — same audio, same VAD, two models")
    print("=" * 100)
    combined = (
        [(s.start, s.end, "BAT", s.text) for s in batisay_segs]
        + [(s.start, s.end, "V3", s.text) for s in large_segs]
    )
    combined.sort(key=lambda x: (x[0], x[2]))
    for start, end, tag, text in combined:
        print(f"{start:7.2f} → {end:7.2f}  {tag:<3}  {end-start:5.2f}s  {text}")
        # Also save the interleaved output to a file for later review
    interleaved_path = out_dir / "interleaved.txt"
    with interleaved_path.open("w", encoding="utf-8") as f:
        for start, end, tag, text in combined:
            f.write(f"{start:7.2f} → {end:7.2f}  {tag:<3}  {end-start:5.2f}s  {text}\n")
    log.info("Interleaved output saved: %s", interleaved_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
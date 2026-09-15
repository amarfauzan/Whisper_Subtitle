"""Temporary smoke test: run the OCR runner on a short video.

Usage:
    uv run python scripts/smoke_runner.py path/to/video.mp4 [seconds]
"""

import sys
from pathlib import Path

from whisper_subtitle.config import load_config
from whisper_subtitle.ocr.engine import PaddleOcrEngine
from whisper_subtitle.ocr.runner import sample_and_detect


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("Usage: uv run python scripts/smoke_runner.py <video> [max_seconds]")
        return 1

    video_path = Path(sys.argv[1]).expanduser().resolve()
    max_seconds = float(sys.argv[2]) if len(sys.argv) == 3 else None

    if not video_path.exists():
        print(f"Not found: {video_path}")
        return 1

    cfg = load_config(Path("config/ocr_only.yaml"))

    # If user gave a max duration, make a trimmed copy so we don't sit
    # here for 30 minutes on a 1-hour video.
    if max_seconds is not None:
        import subprocess
        trimmed = video_path.with_name(video_path.stem + "_trim.mp4")
        print(f"Trimming to {max_seconds}s → {trimmed.name}")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-t", str(max_seconds),
                "-c", "copy",
                str(trimmed),
            ],
            check=True,
            capture_output=True,
        )
        video_path = trimmed

    print(f"Video:      {video_path.name}")
    print(f"Sample fps: {cfg.ocr.sample_fps}")
    print()

    print("Loading engine...")
    engine = PaddleOcrEngine(
        det_model=cfg.ocr.det_model,
        rec_model=cfg.ocr.rec_model,
        dict_file=cfg.ocr.dict_file,
        providers=list(cfg.ocr.providers),
        rec_threshold=cfg.ocr.rec_confidence,
    )

    print("Running OCR...")
    detections, frame_h = sample_and_detect(video_path, engine, cfg.ocr)

    print()
    print(f"Frame height: {frame_h}px")
    print(f"Total detections: {len(detections)}")
    print()

    # Show first 20 in order.
    for i, det in enumerate(detections[:20], start=1):
        print(
            f"  [{i:>3}] t={det.frame_time:6.2f}s  "
            f"conf={det.confidence:.3f}  "
            f"box=({det.box[0]:>4},{det.box[1]:>4},{det.box[2]:>4},{det.box[3]:>4})  "
            f"text={det.text!r}"
        )
    if len(detections) > 20:
        print(f"  ... and {len(detections) - 20} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Temporary smoke test: OCR → grouper → filter on a video.

Usage:
    uv run python scripts/smoke_filter.py path/to/video.mp4
"""

import sys
from pathlib import Path

from whisper_subtitle.config import load_config
from whisper_subtitle.ocr.engine import PaddleOcrEngine
from whisper_subtitle.ocr.filter import events_to_subtitles, filter_events
from whisper_subtitle.ocr.grouper import group_detections
from whisper_subtitle.ocr.runner import sample_and_detect


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/smoke_filter.py <video>")
        return 1

    video_path = Path(sys.argv[1]).expanduser().resolve()
    if not video_path.exists():
        print(f"Not found: {video_path}")
        return 1

    cfg = load_config(Path("config/ocr_only.yaml"))

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
    print(f"  → {len(detections)} raw detections (frame height {frame_h}px)")

    print("Grouping...")
    events = group_detections(detections)
    print(f"  → {len(events)} events")

    print("Filtering...")
    kept = filter_events(events, cfg.ocr.filter, frame_height=frame_h)
    print(f"  → {len(kept)} events kept")
    print()

    print(f"Surviving events ({len(kept)}):")
    for i, ev in enumerate(kept, start=1):
        print(
            f"  [{i:>3}] "
            f"{ev.start:6.2f} → {ev.end:6.2f}s "
            f"(dur {ev.duration:5.2f}s, "
            f"conf {ev.confidence:.2f})  "
            f"text={ev.text!r}"
        )

    subtitles = events_to_subtitles(kept)
    print()
    print(f"Would produce {len(subtitles)} subtitles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

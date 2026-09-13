import argparse
import logging
import sys
from pathlib import Path

from whisper_subtitle.config import load_config
from whisper_subtitle.exceptions import PipelineError
from whisper_subtitle.logging_setup import setup_logging
from whisper_subtitle.pipeline import process_video

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="whisper-subtitle",
        description="Video → SRT transcription pipeline",
    )
    p.add_argument("video", type=Path, help="Input video file")
    p.add_argument("--translate", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    if not args.video.exists():
        log.error("File not found: %s", args.video)
        return 1

    try:
        cfg = load_config(args.config)
        final_srt = process_video(args.video, cfg, translate=args.translate)
    except PipelineError as exc:
        log.error("Pipeline failed: %s", exc)
        return 2

    print(f"Done: {final_srt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

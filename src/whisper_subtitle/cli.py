import argparse
import logging
import sys
import time
from pathlib import Path

from whisper_subtitle.config import load_config, with_mode
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
    p.add_argument(
        "--mode",
        choices=["whisper", "ocr", "merged"],
        default=None,
        help="Run only whisper, only OCR, or merge both. "
             "Overrides the enabled flags in the config.",
    )
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

    start = time.perf_counter()
    exit_code = 0

    try:
        cfg = load_config(args.config)
        if args.mode is not None:
            cfg = with_mode(cfg, args.mode)
        final_srt = process_video(args.video, cfg, translate=args.translate)
        print(f"Done: {final_srt}")
    except PipelineError as exc:
        log.error("Pipeline failed: %s", exc)
        exit_code = 2
    finally:
        elapsed = time.perf_counter() - start
        print(f"Total time: {_format_duration(elapsed)}")

    return exit_code


def _format_duration(seconds: float) -> str:
    """Format seconds as 'Xm Y.YYs' for long durations, or 'Y.YYs' for short."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.2f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"
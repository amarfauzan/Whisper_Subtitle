import logging
from pathlib import Path

from whisper_subtitle.config import Config

log = logging.getLogger(__name__)


def process_video(
    input_path: Path,
    cfg: Config,
    *,
    translate: bool = False,
) -> Path:
    """Run the full pipeline. Returns the path to the final SRT."""
    log.info("Processing %s", input_path)
    log.info("Chunk length: %ds", cfg.chunk_length_seconds)
    log.info("Translate: %s", translate)

    # TODO: split, extract audio, VAD, whisper, merge, combine
    final_srt = input_path.with_suffix(".srt")
    log.info("Done: %s", final_srt)
    return final_srt

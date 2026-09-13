"""Shift subtitle timestamps onto the original video timeline."""

from whisper_subtitle.models import Subtitle


def shift_timeline(
    subtitles: list[Subtitle],
    offset_seconds: float,
) -> list[Subtitle]:
    """Return new Subtitle objects with timestamps shifted by offset_seconds.

    Chunks are split with -reset_timestamps 1, so each chunk starts at 0.0.
    To place a chunk's subtitles back on the original video timeline, we
    add the chunk's real start time to every timestamp.
    """
    if offset_seconds == 0:
        return list(subtitles)

    return [
        Subtitle(
            start=sub.start + offset_seconds,
            end=sub.end + offset_seconds,
            text=sub.text,
        )
        for sub in subtitles
    ]

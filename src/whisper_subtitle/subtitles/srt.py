"""Pure SRT timestamp and formatting utilities."""

from whisper_subtitle.models import Subtitle


def timestamp_to_ms(ts: str) -> int:
    """Convert an SRT timestamp 'HH:MM:SS,mmm' to milliseconds."""
    hours, minutes, rest = ts.split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis)
    )


def ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to an SRT timestamp 'HH:MM:SS,mmm'."""
    if ms < 0:
        raise ValueError(f"Negative timestamp: {ms}")
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def seconds_to_timestamp(seconds: float) -> str:
    """Convert float seconds to an SRT timestamp 'HH:MM:SS,mmm'."""
    if seconds < 0:
        raise ValueError(f"Negative timestamp: {seconds}")
    return ms_to_timestamp(round(seconds * 1000))


def render_srt(subtitles: list[Subtitle]) -> str:
    """Render a list of Subtitle objects to SRT text."""
    blocks = []
    for i, sub in enumerate(subtitles, start=1):
        blocks.append(
            f"{i}\n"
            f"{seconds_to_timestamp(sub.start)} --> "
            f"{seconds_to_timestamp(sub.end)}\n"
            f"{sub.text}"
        )
    return "\n\n".join(blocks) + "\n"
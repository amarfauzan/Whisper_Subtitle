"""Core data types shared across the pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VadSegment:
    """A speech interval detected by the VAD."""

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class WhisperSegment:
    """A transcript segment produced by Whisper."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Subtitle:
    """A finalized subtitle entry."""

    start: float
    end: float
    text: str
    source_hint: str | None = None    # ← new

    @property
    def duration(self) -> float:
        return self.end - self.start

@dataclass(frozen=True)
class Chunk:
    """A video chunk plus its start time in the original video."""

    path: Path
    offset: float

@dataclass(frozen=True)
class OcrDetection:
    """A text detection in a single frame."""

    frame_time: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    text: str
    angle: float
    confidence: float = 1.0


@dataclass(frozen=True)
class OcrEvent:
    """A deduplicated group of detections spanning a time range."""

    start: float
    end: float
    box: tuple[int, int, int, int]
    text: str
    frame_count: int
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start
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

    @property
    def duration(self) -> float:
        return self.end - self.start

@dataclass(frozen=True)
class Chunk:
    """A video chunk plus its start time in the original video."""

    path: Path
    offset: float
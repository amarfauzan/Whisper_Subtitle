"""Core data types shared across the pipeline."""

from dataclasses import dataclass


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
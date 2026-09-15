class PipelineError(Exception):
    """Base for all pipeline failures."""


class FFmpegError(PipelineError):
    """Raised when ffmpeg or ffprobe fail."""


class WhisperError(PipelineError):
    """Raised when the whisper CLI fails."""


class VadError(PipelineError):
    """Raised when the VAD executable fails."""


class TranslationError(PipelineError):
    """Raised when translation fails."""

class OcrError(PipelineError):
    """Raised when OCR fails."""
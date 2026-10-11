"""Quality diagnostics for a chunk's transcription.

Runs after primary (batisay) transcription to catch hallucination loops
before we spend time on secondary (large-v3), OCR, merge, and translation.

Hard aborts only on clear hallucination loops (repeated text). Empty
output is a warning, not an abort — a silent music segment is legitimate.
Everything else is a warning — content-specific anomalies that may be
legitimate (e.g., music segments with no dialogue).
"""

import logging
from dataclasses import dataclass

from whisper_subtitle.models import WhisperSegment

log = logging.getLogger(__name__)


# Hallucination thresholds — must match to abort
_MAX_CONSECUTIVE_REPEAT = 5    # same text 5+ times in a row
_MAX_ALTERNATING_RUN = 8       # A/B/A/B/A/B/A/B sequence

# Statistical warning thresholds
_MIN_COVERAGE_RATIO = 0.05
_MAX_COVERAGE_RATIO = 0.95
_MIN_CHARS_PER_MINUTE = 100
_MAX_CHARS_PER_MINUTE = 2000
_MAX_SINGLE_SEGMENT_DURATION = 15.0


@dataclass
class DiagnosticResult:
    verdict: str                    # "ok" | "warning" | "abort"
    reason: str | None
    warnings: list[str]
    metrics: dict


def diagnose_transcription(
    segments: list[WhisperSegment],
    chunk_duration: float,
) -> DiagnosticResult:
    """Analyze one chunk's transcription. See module docstring."""
    metrics = _compute_metrics(segments, chunk_duration)

    if not segments:
        return DiagnosticResult(
            verdict="warning",
            reason="No segments produced.",
            warnings=["Empty transcription — nothing was produced for this chunk."],
            metrics=metrics,
        )

    if metrics["longest_consecutive_repeat"] >= _MAX_CONSECUTIVE_REPEAT:
        n = metrics["longest_consecutive_repeat"]
        return DiagnosticResult(
            verdict="abort",
            reason=f"Hallucination loop: same text repeated {n} times in a row.",
            warnings=[],
            metrics=metrics,
        )

    if metrics["longest_alternating_run"] >= _MAX_ALTERNATING_RUN:
        n = metrics["longest_alternating_run"]
        return DiagnosticResult(
            verdict="abort",
            reason=f"Hallucination loop: A/B alternation of length {n}.",
            warnings=[],
            metrics=metrics,
        )

    warnings: list[str] = []
    if metrics["coverage_ratio"] < _MIN_COVERAGE_RATIO:
        warnings.append(
            f"Low coverage ({metrics['coverage_ratio']:.1%}) — "
            f"whisper may have missed most audio."
        )
    if metrics["coverage_ratio"] > _MAX_COVERAGE_RATIO:
        warnings.append(
            f"High coverage ({metrics['coverage_ratio']:.1%}) — "
            f"possible hallucination through silence."
        )
    if metrics["chars_per_minute"] < _MIN_CHARS_PER_MINUTE:
        warnings.append(
            f"Low text density ({metrics['chars_per_minute']:.0f} chars/min)."
        )
    if metrics["chars_per_minute"] > _MAX_CHARS_PER_MINUTE:
        warnings.append(
            f"High text density ({metrics['chars_per_minute']:.0f} chars/min)."
        )
    if metrics["max_segment_duration"] > _MAX_SINGLE_SEGMENT_DURATION:
        warnings.append(
            f"Long segment ({metrics['max_segment_duration']:.1f}s) — "
            f"will produce a text-wall cue."
        )

    return DiagnosticResult(
        verdict="warning" if warnings else "ok",
        reason=None,
        warnings=warnings,
        metrics=metrics,
    )


def _compute_metrics(
    segments: list[WhisperSegment],
    chunk_duration: float,
) -> dict:
    if not segments:
        return {
            "segment_count": 0,
            "chunk_duration": chunk_duration,
            "total_speech_seconds": 0.0,
            "coverage_ratio": 0.0,
            "total_chars": 0,
            "chars_per_minute": 0.0,
            "max_segment_duration": 0.0,
            "longest_consecutive_repeat": 0,
            "longest_alternating_run": 0,
        }

    total_speech = sum(s.duration for s in segments)
    total_chars = sum(len(s.text.strip()) for s in segments)
    speech_minutes = total_speech / 60.0

    return {
        "segment_count": len(segments),
        "chunk_duration": chunk_duration,
        "total_speech_seconds": round(total_speech, 3),
        "coverage_ratio": round(
            total_speech / chunk_duration if chunk_duration > 0 else 0.0, 4
        ),
        "total_chars": total_chars,
        "chars_per_minute": round(
            total_chars / speech_minutes if speech_minutes > 0 else 0.0, 1
        ),
        "max_segment_duration": round(
            max(s.duration for s in segments), 3
        ),
        "longest_consecutive_repeat": _longest_consecutive_run(segments),
        "longest_alternating_run": _longest_alternating_run(segments),
    }


def _longest_consecutive_run(segments: list[WhisperSegment]) -> int:
    if not segments:
        return 0
    best = 1
    current = 1
    for i in range(1, len(segments)):
        if segments[i].text == segments[i - 1].text:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _longest_alternating_run(segments: list[WhisperSegment]) -> int:
    n = len(segments)
    if n < 2:
        return 0
    best = 0
    for i in range(n - 1):
        a = segments[i].text
        b = segments[i + 1].text
        if a == b:
            continue
        length = 2
        for j in range(i + 2, n):
            expected = a if (j - i) % 2 == 0 else b
            if segments[j].text == expected:
                length += 1
            else:
                break
        best = max(best, length)
    return best
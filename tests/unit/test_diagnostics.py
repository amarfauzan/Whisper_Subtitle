
from whisper_subtitle.models import WhisperSegment
from whisper_subtitle.transcription.diagnostics import (
    _longest_alternating_run,
    _longest_consecutive_run,
    diagnose_transcription,
)


def seg(start, end, text):
    return WhisperSegment(start=start, end=end, text=text)


class TestConsecutiveRun:
    def test_empty(self):
        assert _longest_consecutive_run([]) == 0

    def test_no_repeats(self):
        segs = [seg(i, i + 1, f"t{i}") for i in range(5)]
        assert _longest_consecutive_run(segs) == 1

    def test_run_of_three(self):
        segs = [
            seg(0, 1, "a"),
            seg(1, 2, "b"), seg(2, 3, "b"), seg(3, 4, "b"),
            seg(4, 5, "c"),
        ]
        assert _longest_consecutive_run(segs) == 3

    def test_run_at_start(self):
        segs = [seg(i, i + 1, "x") for i in range(5)] + [seg(5, 6, "y")]
        assert _longest_consecutive_run(segs) == 5


class TestAlternatingRun:
    def test_empty(self):
        assert _longest_alternating_run([]) == 0

    def test_single(self):
        assert _longest_alternating_run([seg(0, 1, "a")]) == 0

    def test_simple_alternation(self):
        segs = [seg(i, i + 1, "a" if i % 2 == 0 else "b") for i in range(10)]
        assert _longest_alternating_run(segs) == 10

    def test_breaks_on_repeat(self):
        segs = [
            seg(0, 1, "a"), seg(1, 2, "b"),
            seg(2, 3, "b"),   # break
        ]
        assert _longest_alternating_run(segs) == 2


class TestDiagnose:
    def test_empty_is_warning(self):
        """Empty transcription is a warning, not an abort — silence is valid."""
        result = diagnose_transcription([], 600.0)
        assert result.verdict == "warning"
        assert any("empty" in w.lower() for w in result.warnings)

    def test_consecutive_loop_aborts(self):
        segs = [seg(i, i + 1, "감사합니다") for i in range(6)]
        result = diagnose_transcription(segs, 600.0)
        assert result.verdict == "abort"
        assert "consecutive" in result.reason.lower() or "loop" in result.reason.lower()

    def test_alternating_loop_aborts(self):
        segs = [
            seg(i, i + 1, "a" if i % 2 == 0 else "b")
            for i in range(10)
        ]
        result = diagnose_transcription(segs, 600.0)
        assert result.verdict == "abort"

    def test_normal_content_ok(self):
        segs = [seg(i * 2, i * 2 + 1.5, f"문장{i}") for i in range(20)]
        result = diagnose_transcription(segs, 600.0)
        assert result.verdict in ("ok", "warning")
        assert result.verdict != "abort"

    def test_low_coverage_warns(self):
        segs = [seg(0, 1, "short")]
        result = diagnose_transcription(segs, 600.0)
        assert result.verdict == "warning"
        assert any("coverage" in w.lower() for w in result.warnings)

    def test_long_segment_warns(self):
        segs = [seg(0, 20, "very long cue")]  # 20s > 15s threshold
        result = diagnose_transcription(segs, 30.0)
        assert any("long" in w.lower() for w in result.warnings)

    def test_metrics_populated(self):
        segs = [seg(i, i + 1, "x" * 10) for i in range(5)]
        result = diagnose_transcription(segs, 10.0)
        assert result.metrics["segment_count"] == 5
        assert result.metrics["total_chars"] == 50
        assert result.metrics["coverage_ratio"] == 0.5
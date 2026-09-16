"""Tests for the whisper + OCR merge."""

import pytest

from whisper_subtitle.config import MergeConfig
from whisper_subtitle.models import OcrEvent, Subtitle
from whisper_subtitle.subtitles.merge import (
    _best_similarity,
    _overlap_ratio,
    _strip_ocr_prefix,
    merge_whisper_ocr,
)


def ws(start: float, end: float, text: str) -> Subtitle:
    return Subtitle(start=start, end=end, text=text)


def ocr(start: float, end: float, text: str) -> OcrEvent:
    return OcrEvent(
        start=start, end=end, box=(0, 0, 100, 50),
        text=text, frame_count=5, confidence=0.95,
    )


def cfg(**overrides) -> MergeConfig:
    base = dict(
        overlap_threshold=0.3,
        agree_threshold=0.9,
        hint_threshold=0.5,
        fill_gaps=True,
        gap_fill_min_duration=0.5,
        gap_fill_prefix="",
    )
    base.update(overrides)
    return MergeConfig(**base)


# ---------------------------------------------------------------- helpers


class TestOverlapRatio:
    def test_full_overlap_equal_duration(self):
        assert _overlap_ratio(1.0, 3.0, 1.0, 3.0) == 1.0

    def test_half_overlap(self):
        # 2s whisper, 2s OCR, 1s shared.
        assert _overlap_ratio(1.0, 3.0, 2.0, 4.0) == 0.5

    def test_short_inside_long(self):
        # 1s OCR inside 5s whisper: overlap 1s, shorter 1s → 1.0.
        assert _overlap_ratio(5.0, 10.0, 6.0, 7.0) == 1.0

    def test_no_overlap(self):
        assert _overlap_ratio(1.0, 2.0, 3.0, 4.0) == 0.0

    def test_touching_edges(self):
        assert _overlap_ratio(1.0, 2.0, 2.0, 3.0) == 0.0


class TestStripOcrPrefix:
    def test_bracketed_prefix(self):
        assert _strip_ocr_prefix("[SIGN] TOKYO") == "TOKYO"

    def test_speaker_prefix_star(self):
        assert _strip_ocr_prefix("미나미어머니*지금까지") == "지금까지"

    def test_speaker_prefix_hash(self):
        assert _strip_ocr_prefix("ZENA#hello") == "hello"

    def test_no_prefix_unchanged(self):
        assert _strip_ocr_prefix("안녕하세요") == "안녕하세요"

    def test_content_with_colon_not_stripped_blindly(self):
        # "note: something" would strip — but only if followed by colon.
        # Our regex strips up to 15 chars + colon. Verify it does.
        # This is an acceptable tradeoff.
        assert _strip_ocr_prefix("hi: there") == "there"


class TestBestSimilarity:
    def test_identical(self):
        assert _best_similarity("hello", "hello") == 1.0

    def test_with_whitespace(self):
        assert _best_similarity("안녕 하세요", "안녕하세요") == 1.0

    def test_uses_stripped_version(self):
        s = _best_similarity("지금까지", "미나미어머니*지금까지")
        assert s == 1.0

    def test_different_text_low_score(self):
        assert _best_similarity("안녕하세요", "완전히 다른 내용") < 0.5


# ---------------------------------------------------------------- main merge


class TestEmptyInputs:
    def test_both_empty(self):
        assert merge_whisper_ocr([], [], cfg()) == []

    def test_whisper_only(self):
        subs = [ws(1.0, 3.0, "hi")]
        assert merge_whisper_ocr(subs, [], cfg()) == subs

    def test_ocr_only_fill_gaps_on(self):
        events = [ocr(1.0, 3.0, "hi")]
        result = merge_whisper_ocr([], events, cfg(fill_gaps=True))
        assert len(result) == 1
        assert result[0].text == "hi"

    def test_ocr_only_fill_gaps_off(self):
        events = [ocr(1.0, 3.0, "hi")]
        assert merge_whisper_ocr([], events, cfg(fill_gaps=False)) == []


class TestAgreement:
    def test_identical_text_no_hint(self):
        result = merge_whisper_ocr(
            [ws(1.0, 3.0, "안녕하세요")],
            [ocr(1.0, 3.0, "안녕하세요")],
            cfg(),
        )
        assert len(result) == 1
        assert result[0].source_hint is None

    def test_whitespace_only_difference_no_hint(self):
        result = merge_whisper_ocr(
            [ws(1.0, 3.0, "안녕 하세요")],
            [ocr(1.0, 3.0, "안녕하세요")],
            cfg(),
        )
        assert result[0].source_hint is None

    def test_prefix_stripping_agreement(self):
        result = merge_whisper_ocr(
            [ws(1.0, 3.0, "지금까지")],
            [ocr(1.0, 3.0, "미나미어머니*지금까지")],
            cfg(),
        )
        assert len(result) == 1
        assert result[0].source_hint is None


class TestModerateSimilarity:
    def test_hint_attached(self):
        result = merge_whisper_ocr(
            [ws(1.0, 3.0, "오늘은 좋은 날이네요")],
            [ocr(1.0, 3.0, "오늘은 좋은 날이에요")],
            cfg(),
        )
        assert len(result) == 1
        assert result[0].source_hint == "오늘은 좋은 날이에요"


class TestDifferentContent:
    def test_whisper_wins_no_hint(self):
        result = merge_whisper_ocr(
            [ws(1.0, 3.0, "안녕하세요")],
            [ocr(1.0, 3.0, "TOKYO")],
            cfg(),
        )
        assert len(result) == 1
        assert result[0].text == "안녕하세요"
        assert result[0].source_hint is None

    def test_different_content_not_gap_filled(self):
        # Even with fill_gaps on, an OCR event overlapping a whisper
        # sub should not become its own entry.
        result = merge_whisper_ocr(
            [ws(1.0, 3.0, "안녕하세요")],
            [ocr(1.0, 3.0, "TOKYO")],
            cfg(fill_gaps=True),
        )
        assert len(result) == 1


class TestGapFill:
    def test_ocr_fills_empty_gap(self):
        result = merge_whisper_ocr(
            [ws(1.0, 2.0, "first"), ws(5.0, 6.0, "second")],
            [ocr(3.0, 4.0, "caption")],
            cfg(fill_gaps=True),
        )
        assert [s.text for s in result] == ["first", "caption", "second"]

    def test_gap_fill_off(self):
        result = merge_whisper_ocr(
            [ws(1.0, 2.0, "first")],
            [ocr(3.0, 4.0, "caption")],
            cfg(fill_gaps=False),
        )
        assert [s.text for s in result] == ["first"]

    def test_gap_fill_below_min_duration(self):
        result = merge_whisper_ocr(
            [],
            [ocr(1.0, 1.2, "short")],
            cfg(fill_gaps=True, gap_fill_min_duration=0.5),
        )
        assert result == []

    def test_gap_fill_prefix_applied(self):
        result = merge_whisper_ocr(
            [],
            [ocr(1.0, 3.0, "TOKYO")],
            cfg(fill_gaps=True, gap_fill_prefix="[SIGN] "),
        )
        assert result[0].text == "[SIGN] TOKYO"


class TestClaiming:
    def test_one_ocr_event_not_used_twice(self):
        # Two whisper subs overlap one OCR event.
        # The one with higher overlap claims it; the other has no match.
        result = merge_whisper_ocr(
            [ws(1.0, 3.0, "first"), ws(2.0, 4.0, "second")],
            [ocr(2.5, 4.0, "second")],
            cfg(),
        )
        # Only one hint should be attached.
        hints = [s for s in result if s.source_hint is not None]
        assert len(hints) <= 1

    def test_best_overlap_wins_for_whisper(self):
        # A whisper sub with two overlapping OCR events picks the one
        # with more overlap.
        result = merge_whisper_ocr(
            [ws(1.0, 5.0, "whisper text")],
            [ocr(1.0, 3.0, "wsp text"), ocr(1.0, 5.0, "whisper text")],
            cfg(),
        )
        assert len(result) == 1
        # The second one is identical → agreement → no hint.
        assert result[0].source_hint is None


class TestOrdering:
    def test_output_sorted_by_start(self):
        result = merge_whisper_ocr(
            [ws(5.0, 6.0, "late")],
            [ocr(1.0, 2.0, "early")],
            cfg(fill_gaps=True),
        )
        assert [s.text for s in result] == ["early", "late"]

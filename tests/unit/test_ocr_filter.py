"""Tests for the OCR event filter."""

import re

import pytest

from whisper_subtitle.config import OcrFilterConfig
from whisper_subtitle.models import OcrEvent
from whisper_subtitle.ocr.filter import (
    _reject_reason,
    events_to_subtitles,
    filter_events,
)


FRAME_H = 1080  # 1080p reference for box-height ratio math


def make_event(
    *,
    start: float = 1.0,
    end: float = 3.0,
    text: str = "안녕하세요",
    box: tuple[int, int, int, int] = (100, 800, 500, 860),  # 60px tall
    confidence: float = 0.9,
    frame_count: int = 5,
) -> OcrEvent:
    return OcrEvent(
        start=start,
        end=end,
        box=box,
        text=text,
        frame_count=frame_count,
        confidence=confidence,
    )


def make_cfg(**overrides) -> OcrFilterConfig:
    base = dict(
        min_duration=0.5,
        max_duration=8.0,
        min_box_height_ratio=0.02,   # 21.6 px on 1080
        max_box_height_ratio=0.15,   # 162 px on 1080
        min_text_length=2,
        max_text_repeats=8,  
        min_confidence=0.75,
        blocklist_texts=(),
        blocklist_patterns=("^[!?.…\\s\\(\\)\\[\\]\\{\\}]+$", "^[0-9\\s]+$"),
    )
    base.update(overrides)
    return OcrFilterConfig(**base)


def reason_for(event, cfg, frame_height=FRAME_H):
    """Helper: run the internal single-event reason check."""
    compiled = [re.compile(p) for p in cfg.blocklist_patterns]
    blocklist = {t.lower() for t in cfg.blocklist_texts}
    return _reject_reason(event, cfg, frame_height, compiled, blocklist)


# ---------------------------------------------------------------- validation


class TestInputValidation:
    def test_empty_input(self):
        assert filter_events([], make_cfg(), frame_height=FRAME_H) == []

    def test_zero_frame_height_raises(self):
        with pytest.raises(ValueError):
            filter_events([], make_cfg(), frame_height=0)

    def test_negative_frame_height_raises(self):
        with pytest.raises(ValueError):
            filter_events([], make_cfg(), frame_height=-1)


# ---------------------------------------------------------------- confidence


class TestConfidence:
    def test_below_threshold_dropped(self):
        events = [make_event(confidence=0.5)]
        assert filter_events(events, make_cfg(), frame_height=FRAME_H) == []

    def test_at_threshold_kept(self):
        events = [make_event(confidence=0.75)]
        assert len(filter_events(events, make_cfg(), frame_height=FRAME_H)) == 1

    def test_above_threshold_kept(self):
        events = [make_event(confidence=0.95)]
        assert len(filter_events(events, make_cfg(), frame_height=FRAME_H)) == 1


# ---------------------------------------------------------------- duration


class TestDuration:
    def test_too_short_dropped(self):
        events = [make_event(start=1.0, end=1.2)]  # 0.2s
        assert filter_events(events, make_cfg(), frame_height=FRAME_H) == []

    def test_at_min_kept(self):
        events = [make_event(start=1.0, end=1.5)]  # 0.5s
        assert len(filter_events(events, make_cfg(), frame_height=FRAME_H)) == 1

    def test_in_range_kept(self):
        events = [make_event(start=1.0, end=4.0)]  # 3s
        assert len(filter_events(events, make_cfg(), frame_height=FRAME_H)) == 1

    def test_at_max_kept(self):
        events = [make_event(start=0.0, end=8.0)]  # exactly 8s
        assert len(filter_events(events, make_cfg(), frame_height=FRAME_H)) == 1

    def test_too_long_dropped(self):
        events = [make_event(start=0.0, end=10.0)]  # 10s
        assert filter_events(events, make_cfg(), frame_height=FRAME_H) == []


# ---------------------------------------------------------------- box height


class TestBoxHeight:
    def test_too_small_dropped(self):
        # 10px on 1080 → ratio 0.009 < 0.02
        events = [make_event(box=(100, 800, 500, 810))]
        assert filter_events(events, make_cfg(), frame_height=FRAME_H) == []

    def test_too_large_dropped(self):
        # 200px on 1080 → ratio 0.185 > 0.15
        events = [make_event(box=(100, 700, 500, 900))]
        assert filter_events(events, make_cfg(), frame_height=FRAME_H) == []

    def test_in_range_kept(self):
        # 60px on 1080 → ratio 0.056
        events = [make_event(box=(100, 800, 500, 860))]
        assert len(filter_events(events, make_cfg(), frame_height=FRAME_H)) == 1

    def test_scales_with_smaller_frame(self):
        # 60px on 720 → ratio 0.083, still in range.
        events = [make_event(box=(100, 500, 500, 560))]
        assert len(filter_events(events, make_cfg(), frame_height=720)) == 1

    def test_scales_with_larger_frame(self):
        # 30px on 2160 → ratio 0.014, below 0.02, dropped.
        events = [make_event(box=(100, 800, 500, 830))]
        assert filter_events(events, make_cfg(), frame_height=2160) == []


# ---------------------------------------------------------------- text length


class TestTextLength:
    def test_single_char_dropped(self):
        events = [make_event(text="R")]
        assert filter_events(events, make_cfg(), frame_height=FRAME_H) == []

    def test_two_chars_kept(self):
        events = [make_event(text="응?")]
        assert len(filter_events(events, make_cfg(), frame_height=FRAME_H)) == 1

    def test_whitespace_only_dropped(self):
        events = [make_event(text="   ")]
        assert filter_events(events, make_cfg(), frame_height=FRAME_H) == []

    def test_surrounding_whitespace_stripped_before_length_check(self):
        # " a " → "a" → len 1, dropped
        events = [make_event(text=" a ")]
        assert filter_events(events, make_cfg(), frame_height=FRAME_H) == []


# ---------------------------------------------------------------- blocklist


class TestBlocklistTexts:
    def test_exact_match_dropped(self):
        events = [make_event(text="웃음")]
        cfg = make_cfg(blocklist_texts=("웃음",))
        assert filter_events(events, cfg, frame_height=FRAME_H) == []

    def test_not_in_list_kept(self):
        events = [make_event(text="웃음")]
        cfg = make_cfg(blocklist_texts=("두근",))
        assert len(filter_events(events, cfg, frame_height=FRAME_H)) == 1

    def test_case_insensitive(self):
        events = [make_event(text="LOL")]
        cfg = make_cfg(blocklist_texts=("lol",))
        assert filter_events(events, cfg, frame_height=FRAME_H) == []


# ---------------------------------------------------------------- patterns


class TestBlocklistPatterns:
    def test_punctuation_only_dropped(self):
        events = [make_event(text="!!!")]
        assert filter_events(events, make_cfg(), frame_height=FRAME_H) == []

    def test_punctuation_with_brackets_dropped(self):
        events = [make_event(text="(!?)")]
        cfg = make_cfg(
            blocklist_patterns=("^[!?.…\\s\\(\\)\\[\\]\\{\\}]+$",)
        )
        assert filter_events(events, cfg, frame_height=FRAME_H) == []

    def test_digits_only_dropped(self):
        events = [make_event(text="12345")]
        assert filter_events(events, make_cfg(), frame_height=FRAME_H) == []

    def test_short_ascii_alphanumeric_dropped(self):
        events = [make_event(text="Ch(5")]
        cfg = make_cfg(
            blocklist_patterns=("^[A-Za-z0-9\\(\\)\\s]{1,4}$",)
        )
        assert filter_events(events, cfg, frame_height=FRAME_H) == []

    def test_korean_text_kept(self):
        events = [make_event(text="안녕하세요")]
        assert len(filter_events(events, make_cfg(), frame_height=FRAME_H)) == 1

    def test_mixed_text_with_number_kept(self):
        # "5시간" contains a digit but isn't digits-only.
        events = [make_event(text="5시간")]
        assert len(filter_events(events, make_cfg(), frame_height=FRAME_H)) == 1


# ---------------------------------------------------------------- multiple


class TestMultipleEvents:
    def test_mixed_input_keeps_only_valid(self):
        events = [
            make_event(start=0.0, end=2.0, text="안녕하세요"),
            make_event(start=3.0, end=3.2, text="네"),                   # too short
            make_event(start=5.0, end=20.0, text="세탁소"),              # too long
            make_event(start=21.0, end=23.0, text="!!!"),                # pattern
            make_event(start=24.0, end=26.0, text="내일 봐요"),           # keep
            make_event(start=27.0, end=28.0, text="x", confidence=0.5),  # low conf
        ]
        result = filter_events(events, make_cfg(), frame_height=FRAME_H)
        assert [e.text for e in result] == ["안녕하세요", "내일 봐요"]

    def test_preserves_input_order(self):
        events = [
            make_event(start=5.0, end=7.0, text="second"),
            make_event(start=0.0, end=2.0, text="first"),
        ]
        result = filter_events(events, make_cfg(), frame_height=FRAME_H)
        assert [e.text for e in result] == ["second", "first"]


# ---------------------------------------------------------------- reasons


class TestRejectionReasons:
    def test_low_confidence(self):
        assert reason_for(make_event(confidence=0.5), make_cfg()) == "low_confidence"

    def test_too_short(self):
        assert reason_for(make_event(start=1.0, end=1.1), make_cfg()) == "too_short"

    def test_too_long(self):
        assert reason_for(make_event(start=0.0, end=20.0), make_cfg()) == "too_long"

    def test_box_too_small(self):
        e = make_event(box=(100, 800, 500, 810))
        assert reason_for(e, make_cfg()) == "box_too_small"

    def test_box_too_large(self):
        e = make_event(box=(100, 700, 500, 900))
        assert reason_for(e, make_cfg()) == "box_too_large"

    def test_text_too_short(self):
        assert reason_for(make_event(text="X"), make_cfg()) == "text_too_short"

    def test_blocklisted(self):
        cfg = make_cfg(blocklist_texts=("웃음",))
        assert reason_for(make_event(text="웃음"), cfg) == "blocklisted"

    def test_pattern_match(self):
        assert reason_for(make_event(text="!!!"), make_cfg()) == "pattern_match"

    def test_valid_event_returns_none(self):
        assert reason_for(make_event(), make_cfg()) is None

    def test_confidence_checked_before_duration(self):
        # Fails both. Confidence must fire first.
        e = make_event(start=1.0, end=1.05, confidence=0.3)
        assert reason_for(e, make_cfg()) == "low_confidence"


# ---------------------------------------------------------------- conversion


class TestEventsToSubtitles:
    def test_empty(self):
        assert events_to_subtitles([]) == []

    def test_single(self):
        events = [make_event(start=1.0, end=2.5, text="안녕")]
        subs = events_to_subtitles(events)
        assert len(subs) == 1
        assert subs[0].start == 1.0
        assert subs[0].end == 2.5
        assert subs[0].text == "안녕"

    def test_multiple_preserve_order(self):
        events = [
            make_event(start=1.0, end=2.0, text="first"),
            make_event(start=3.0, end=4.0, text="second"),
        ]
        subs = events_to_subtitles(events)
        assert [s.text for s in subs] == ["first", "second"]

from whisper_subtitle.ocr.filter import filter_repeated_texts


class TestRepeatedTexts:
    def test_drops_text_appearing_more_than_max(self):
        events = [make_event(text="Sports Day") for _ in range(10)]
        assert filter_repeated_texts(events, max_repeats=8) == []

    def test_keeps_text_at_or_below_max(self):
        events = [make_event(text="Sports Day") for _ in range(8)]
        assert len(filter_repeated_texts(events, max_repeats=8)) == 8

    def test_keeps_text_below_max(self):
        events = [make_event(text="Sports Day") for _ in range(3)]
        assert len(filter_repeated_texts(events, max_repeats=8)) == 3

    def test_different_texts_counted_separately(self):
        events = (
            [make_event(text="Sports Day") for _ in range(10)]
            + [make_event(text="안녕하세요")]
        )
        result = filter_repeated_texts(events, max_repeats=8)
        assert len(result) == 1
        assert result[0].text == "안녕하세요"

    def test_case_insensitive(self):
        events = (
            [make_event(text="sports day") for _ in range(5)]
            + [make_event(text="SPORTS DAY") for _ in range(5)]
        )
        assert filter_repeated_texts(events, max_repeats=8) == []

    def test_whitespace_ignored(self):
        events = (
            [make_event(text="sportsday") for _ in range(5)]
            + [make_event(text="sports day") for _ in range(5)]
        )
        assert filter_repeated_texts(events, max_repeats=8) == []

    def test_substring_is_separate_bucket(self):
        # The whole phrase "minami" repeated → dropped.
        # But "hi, my name is minami" is a different bucket → kept.
        events = (
            [make_event(text="minami") for _ in range(15)]
            + [make_event(text="hi, my name is minami") for _ in range(3)]
        )
        result = filter_repeated_texts(events, max_repeats=8)
        assert len(result) == 3
        assert all("hi, my name" in e.text for e in result)

    def test_empty_input(self):
        assert filter_repeated_texts([], max_repeats=8) == []

    def test_zero_disables(self):
        events = [make_event(text="Sports Day") for _ in range(100)]
        assert len(filter_repeated_texts(events, max_repeats=0)) == 100

    def test_negative_disables(self):
        events = [make_event(text="Sports Day") for _ in range(100)]
        assert len(filter_repeated_texts(events, max_repeats=-1)) == 100
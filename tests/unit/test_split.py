"""Tests for final-SRT cue splitting."""

import itertools

from whisper_subtitle.models import Subtitle
from whisper_subtitle.subtitles.split import split_long_subtitles


def sub(start: float, end: float, text: str) -> Subtitle:
    return Subtitle(start=start, end=end, text=text)


class TestNoSplit:
    def test_short_cue_untouched(self):
        cues = [sub(1.0, 3.0, "안녕하세요")]
        assert split_long_subtitles(
            cues, max_chars=40, max_duration=6.0
        ) == cues

    def test_long_duration_with_short_text_kept(self):
        # Nothing to split textually; the VAD-extension cap in the
        # ensemble path is what keeps cues like this short.
        result = split_long_subtitles(
            [sub(0.0, 9.0, "진짜 꿀이네.")], max_chars=40, max_duration=6.0
        )
        assert len(result) == 1

    def test_long_duration_with_long_text_split(self):
        text = "1991년 때는 바야흐로 때는 바야흐로 1991년 와 1991년은 이제 바야흐로를 붙여야 되는"
        result = split_long_subtitles(
            [sub(0.0, 9.0, text)], max_chars=40, max_duration=6.0
        )
        assert len(result) >= 2
        assert all(c.duration <= 6.0 for c in result)

    def test_empty_text_untouched(self):
        cues = [sub(0.0, 3.0, "   ")]
        assert split_long_subtitles(cues, max_chars=10, max_duration=2.0) == cues

    def test_disabled_when_both_limits_off(self):
        cues = [sub(0.0, 20.0, "가" * 200)]
        assert split_long_subtitles(cues, max_chars=0, max_duration=0) == cues


class TestSplitText:
    def test_splits_on_sentence_boundary(self):
        text = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
        result = split_long_subtitles(
            [sub(0.0, 9.0, text)], max_chars=20, max_duration=99.0
        )
        assert len(result) > 1
        assert all(len(c.text) <= 20 for c in result)
        # No text lost, order preserved.
        joined = "".join(c.text.replace(" ", "") for c in result)
        assert joined == text.replace(" ", "")

    def test_splits_unspaced_ocr_text_by_chars(self):
        text = "가" * 75
        result = split_long_subtitles(
            [sub(0.0, 9.0, text)], max_chars=30, max_duration=99.0
        )
        assert len(result) == 3
        assert all(len(c.text) <= 30 for c in result)

    def test_time_is_distributed_proportionally(self):
        text = "짧은 말. " + "긴 문장이 계속 이어집니다 " * 3
        result = split_long_subtitles(
            [sub(10.0, 20.0, text)], max_chars=20, max_duration=99.0
        )
        assert result[0].start == 10.0
        assert result[-1].end == 20.0
        # Cues are contiguous and in order.
        for a, b in itertools.pairwise(result):
            assert a.end == b.start
            assert b.start >= a.start

    def test_pieces_respect_min_duration(self):
        # 2s available and min_duration 0.5 -> at most 4 pieces.
        text = "가나다라 " * 40
        result = split_long_subtitles(
            [sub(0.0, 2.0, text)],
            max_chars=10,
            max_duration=99.0,
            min_duration=0.5,
        )
        assert len(result) <= 4
        assert all(c.duration >= 0.5 for c in result)

    def test_source_hint_kept_on_first_piece(self):
        cue = Subtitle(start=0.0, end=6.0, text="가나다라마바사 " * 4, source_hint="힌트")
        result = split_long_subtitles(
            [cue], max_chars=15, max_duration=99.0, min_duration=0.5
        )
        assert result[0].source_hint == "힌트"
        assert all(c.source_hint is None for c in result[1:])

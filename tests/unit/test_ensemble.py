"""Tests for the ensemble merge module."""

from dataclasses import replace

import pytest

from whisper_subtitle.config import EnsembleConfig
from whisper_subtitle.models import VadSegment, WhisperSegment
from whisper_subtitle.transcription.ensemble import (
    _is_junk,
    _overlap,
    _similarity,
    merge_ensemble,
    _split_bat_across_v3,
)



def ws(start: float, end: float, text: str) -> WhisperSegment:
    return WhisperSegment(start=start, end=end, text=text)


def cfg(**overrides) -> EnsembleConfig:
    base = dict(
        enabled=True,
        secondary_model=__import__("pathlib").Path("/fake/v3.bin"),
        secondary_port=8081,
        overlap_threshold=0.4,
        agree_threshold=0.85,
        hint_threshold=0.5,
        min_cue_duration=0.5,
        min_cue_chars=2,
        llm_adjudicate=True,
        llm_batch_size=50,
    )
    base.update(overrides)
    return EnsembleConfig(**base)


# ---------------------------------------------------------------- helpers


class TestSimilarity:
    def test_identical(self):
        assert _similarity("안녕하세요", "안녕하세요") == 1.0

    def test_whitespace_ignored(self):
        assert _similarity("안녕 하세요", "안녕하세요") == 1.0

    def test_case_insensitive(self):
        assert _similarity("HELLO", "hello") == 1.0

    def test_completely_different(self):
        assert _similarity("안녕", "저기요") < 0.5


class TestOverlap:
    def test_full_overlap(self):
        a = ws(1.0, 3.0, "a")
        b = ws(1.0, 3.0, "b")
        assert _overlap(a, b) == 1.0

    def test_partial_overlap(self):
        a = ws(1.0, 3.0, "a")
        b = ws(2.0, 4.0, "b")
        assert _overlap(a, b) == pytest.approx(0.5)

    def test_no_overlap(self):
        a = ws(1.0, 2.0, "a")
        b = ws(3.0, 4.0, "b")
        assert _overlap(a, b) == 0.0

    def test_short_inside_long(self):
        # 1s short inside 5s long
        a = ws(1.0, 6.0, "a")
        b = ws(2.0, 3.0, "b")
        assert _overlap(a, b) == 1.0


class TestJunkFilter:
    def test_short_dot_dropped(self):
        seg = ws(1.0, 1.2, ".")
        assert _is_junk(seg, cfg())

    def test_short_single_char_dropped(self):
        seg = ws(1.0, 1.2, "ㅋ")
        assert _is_junk(seg, cfg())

    def test_short_but_valid_text_kept(self):
        # 0.7s cue with "네!" — not junk
        seg = ws(1.0, 1.7, "네!")
        assert not _is_junk(seg, cfg())

    def test_long_enough_kept(self):
        seg = ws(1.0, 2.0, ".")
        # duration >= 0.5 → kept even if text is short
        assert not _is_junk(seg, cfg())

    def test_long_text_short_duration_kept(self):
        # short cue but enough chars
        seg = ws(1.0, 1.3, "안녕")
        assert not _is_junk(seg, cfg())


# ---------------------------------------------------------------- merge


class TestEmptyInputs:
    def test_both_empty(self):
        result = merge_ensemble([], [], cfg())
        assert result.cues == []
        assert result.pending == []

    def test_v3_only(self):
        v3 = [ws(1.0, 3.0, "안녕하세요")]
        result = merge_ensemble([], v3, cfg())
        assert len(result.cues) == 1
        assert result.cues[0].text == "안녕하세요"
        assert result.pending == []

    def test_bat_only(self):
        # No V3 → no timing skeleton → nothing merged
        bat = [ws(1.0, 3.0, "안녕하세요")]
        result = merge_ensemble(bat, [], cfg())
        assert result.cues == []


class TestAgreement:
    def test_identical_texts_kept_as_v3(self):
        bat = [ws(1.0, 3.0, "안녕하세요")]
        v3 = [ws(1.0, 3.0, "안녕하세요")]
        result = merge_ensemble(bat, v3, cfg())
        assert len(result.cues) == 1
        assert result.cues[0].text == "안녕하세요"
        assert result.pending == []

    def test_high_similarity_keeps_v3(self):
        bat = [ws(1.0, 3.0, "안녕하세요 오늘은 좋은 날이네요")]
        v3 = [ws(1.0, 3.0, "안녕하세요 오늘은 좋은 날이에요")]
        result = merge_ensemble(bat, v3, cfg())
        assert len(result.cues) == 1
        assert result.cues[0].text == v3[0].text
        assert result.pending == []


class TestModerateAgreement:
    def test_hint_range_uses_bat_text(self):
        # Similarity between agree (0.85) and hint (0.5) → use BAT
        bat = [ws(1.0, 3.0, "안녕하세요 반갑습니다")]
        v3 = [ws(1.0, 3.0, "안녕하세여 반가워요")]
        result = merge_ensemble(bat, v3, cfg())
        assert len(result.cues) == 1
        assert result.cues[0].text == "안녕하세요 반갑습니다"
        assert result.pending == []

    def test_uses_v3_timing(self):
        bat = [ws(10.0, 20.0, "안녕하세요 반갑습니다")]
        v3 = [ws(10.0, 13.0, "안녕하세여 반가워요")]
        result = merge_ensemble(bat, v3, cfg())
        # V3 timing preserved, BAT text used
        assert result.cues[0].start == 10.0
        assert result.cues[0].end == 13.0
        assert result.cues[0].text == "안녕하세요 반갑습니다"


class TestDisagreement:
    def test_low_similarity_marked_pending(self):
        bat = [ws(1.0, 3.0, "안녕하세요")]
        v3 = [ws(1.0, 3.0, "완전히 다른 말")]
        result = merge_ensemble(bat, v3, cfg())
        assert len(result.cues) == 1
        assert len(result.pending) == 1

    def test_pending_contains_both_readings(self):
        bat = [ws(1.0, 3.0, "안녕하세요")]
        v3 = [ws(1.0, 3.0, "완전히 다른 말")]
        result = merge_ensemble(bat, v3, cfg())
        req = result.pending[0]
        assert req.bat_text == "안녕하세요"
        assert req.v3_text == "완전히 다른 말"

    def test_pending_has_context(self):
        bat = [
            ws(0.0, 1.0, "before"),
            ws(1.0, 3.0, "안녕하세요"),
            ws(3.0, 4.0, "after"),
        ]
        v3 = [
            ws(0.0, 1.0, "before"),
            ws(1.0, 3.0, "완전히 다른 말"),
            ws(3.0, 4.0, "after"),
        ]
        result = merge_ensemble(bat, v3, cfg())
        req = result.pending[0]
        assert "before" in req.context_before or "before" in req.context_after
        # after is filled by the merge function
        assert req.context_after


class TestAssignment:
    def test_bat_assigned_to_best_overlap(self):
        bat = [ws(5.0, 7.0, "text")]
        v3 = [
            ws(0.0, 3.0, "first"),
            ws(5.0, 7.0, "second"),    # this one gets the BAT cue
            ws(8.0, 10.0, "third"),
        ]
        result = merge_ensemble(bat, v3, cfg())
        assert len(result.cues) == 3
        # only the middle cue had a BAT match

    def test_bat_not_reused_across_v3(self):
        # One BAT cue overlaps two V3 cues equally
        # It should only be assigned to one (highest overlap)
        bat = [ws(2.0, 4.0, "text")]
        v3 = [
            ws(0.0, 3.0, "first"),     # overlap 1s
            ws(3.0, 5.0, "second"),    # overlap 1s (tie)
        ]
        result = merge_ensemble(bat, v3, cfg())
        # At most one cue gets the BAT text
        texts = [c.text for c in result.cues]
        assert texts.count("text") <= 1

    def test_multiple_bat_per_v3(self):
        # Two BAT cues both map to one V3 cue → concatenated text
        bat = [
            ws(1.0, 2.0, "안녕"),
            ws(2.0, 3.0, "하세요"),
        ]
        v3 = [ws(1.0, 3.0, "안녕하세요")]
        result = merge_ensemble(bat, v3, cfg())
        # BAT text should be joined
        assert "안녕" in result.cues[0].text


class TestJunkDropped:
    def test_short_dot_cue_removed(self):
        bat = []
        v3 = [
            ws(1.0, 1.2, "."),          # junk
            ws(2.0, 4.0, "안녕하세요"),
        ]
        result = merge_ensemble(bat, v3, cfg())
        assert len(result.cues) == 1
        assert result.cues[0].text == "안녕하세요"

    def test_valid_short_cue_kept(self):
        v3 = [ws(1.0, 1.7, "네!")]
        result = merge_ensemble([], v3, cfg())
        assert len(result.cues) == 1


class TestOrdering:
    def test_cues_keep_v3_order(self):
        bat = []
        v3 = [
            ws(1.0, 2.0, "first"),
            ws(2.0, 3.0, "second"),
            ws(3.0, 4.0, "third"),
        ]
        result = merge_ensemble(bat, v3, cfg())
        texts = [c.text for c in result.cues]
        assert texts == ["first", "second", "third"]


class TestMultipleMixed:
    def test_mixed_agree_hint_disagree(self):
        bat = [
            ws(0.0, 1.0, "same"),                # agree with v3
            ws(1.0, 3.0, "안녕하세요 반갑습니다"),  # hint range
            ws(4.0, 6.0, "안녕"),                  # disagree
        ]
        v3 = [
            ws(0.0, 1.0, "same"),
            ws(1.0, 3.0, "안녕하세여 반가워요"),
            ws(4.0, 6.0, "완전히 다른 말"),
        ]
        result = merge_ensemble(bat, v3, cfg())
        assert len(result.cues) == 3
        assert len(result.pending) == 1
        assert result.pending[0].index == 2

class TestSplitBatAcrossV3:
    def test_single_v3_cue(self):
        bat_text = "안녕 비쥬얼즈의 정답은 짜란"
        v3 = [WhisperSegment(0.0, 3.0, "안녕")]
        assert _split_bat_across_v3(bat_text, v3) == [bat_text]

    def test_three_v3_cues_splits(self):
        bat_text = "안녕 비쥬얼즈의 정답은 짜란"
        v3 = [
            WhisperSegment(0.0, 1.0, "안녕"),
            WhisperSegment(1.0, 2.0, "비쥬얼즈의 정답은"),
            WhisperSegment(2.0, 3.0, "짜란"),
        ]
        pieces = _split_bat_across_v3(bat_text, v3)
        assert len(pieces) == 3
        # Recombined pieces should equal original (minus whitespace)
        recombined = "".join(pieces).replace(" ", "")
        original = bat_text.replace(" ", "")
        assert recombined == original

    def test_short_bat_text_not_split(self):
        bat_text = "응"
        v3 = [
            WhisperSegment(0.0, 1.0, "네"),
            WhisperSegment(1.0, 2.0, "네"),
        ]
        pieces = _split_bat_across_v3(bat_text, v3)
        assert pieces[0] == "응"

    def test_proportional_to_v3_text_length(self):
        # First V3 cue is much longer text than second
        bat_text = "안녕하세요 반갑습니다 오늘은 좋은 날이네요"
        v3 = [
            WhisperSegment(0.0, 2.0, "안녕하세요 반갑습니다"),  # 10 chars
            WhisperSegment(2.0, 3.0, "오늘은"),                # 3 chars
        ]
        pieces = _split_bat_across_v3(bat_text, v3)
        # First piece should be much longer than second
        assert len(pieces[0]) > len(pieces[1])


class TestMultiV3Assignment:
    def test_bat_spans_three_v3_cues(self):
        bat = [WhisperSegment(0.0, 3.0, "안녕 비쥬얼즈의 정답은 짜란")]
        v3 = [
            WhisperSegment(0.0, 1.0, "안녕"),
            WhisperSegment(1.0, 2.0, "비쥬얼즈의 정답은"),
            WhisperSegment(2.0, 3.0, "짜란"),
        ]
        result = merge_ensemble(bat, v3, cfg())
        # Should have 3 cues, not 1
        assert len(result.cues) == 3
        # All three should contain text (no empty cues)
        assert all(c.text.strip() for c in result.cues)

from whisper_subtitle.transcription.ensemble_pipeline import _clip_to_vad


class TestClipToVad:
    def test_single_vad_clip(self):
        cues = [WhisperSegment(1.0, 5.0, "text")]  # spans wider than VAD
        vads = [VadSegment(0, 2.0, 4.0)]
        result = _clip_to_vad(cues, vads)
        assert len(result) == 1
        assert result[0].start == 2.0
        assert result[0].end == 4.0

    def test_inside_vad_unchanged(self):
        cues = [WhisperSegment(2.5, 3.5, "text")]
        vads = [VadSegment(0, 2.0, 4.0)]
        result = _clip_to_vad(cues, vads)
        assert result[0].start == 2.5
        assert result[0].end == 3.5

    def test_no_vad_overlap_dropped(self):
        cues = [WhisperSegment(5.0, 6.0, "hallucination")]
        vads = [VadSegment(0, 1.0, 2.0)]
        assert _clip_to_vad(cues, vads) == []

    def test_cue_spanning_two_vads(self):
        # Cue spans a silence gap between two VAD segments
        cues = [WhisperSegment(1.0, 5.0, "long")]
        vads = [VadSegment(0, 0.5, 2.0), VadSegment(1, 4.0, 6.0)]
        result = _clip_to_vad(cues, vads)
        assert len(result) == 1
        # Clipped to the union: 0.5 to 6.0, intersected with cue's 1.0-5.0
        assert result[0].start == 1.0
        assert result[0].end == 5.0
"""Tests for ensemble cue clipping against VAD regions."""

from whisper_subtitle.models import VadSegment, WhisperSegment
from whisper_subtitle.transcription.ensemble_pipeline import _clip_to_vad


def cue(start: float, end: float, text: str = "말") -> WhisperSegment:
    return WhisperSegment(start=start, end=end, text=text)


def vad(index: int, start: float, end: float) -> VadSegment:
    return VadSegment(index=index, start=start, end=end)


class TestClipToVad:
    def test_cue_without_vad_overlap_dropped(self):
        assert _clip_to_vad([cue(1.0, 2.0)], [vad(0, 10.0, 12.0)]) == []

    def test_cue_inside_vad_extends_at_most_one_second(self):
        result = _clip_to_vad([cue(10.5, 11.5)], [vad(0, 10.0, 20.0)])
        assert len(result) == 1
        assert result[0].start == 10.5
        assert result[0].end == 12.5  # capped at cue.end + 1.0

    def test_short_cue_not_stretched_across_whole_vad(self):
        # A 1.2s line inside a 10s speech region must not hold the screen
        # for the whole region.
        result = _clip_to_vad([cue(10.0, 11.2, "진짜 꿀이네.")], [vad(0, 10.0, 20.0)])
        assert len(result) == 1
        assert result[0].end <= 11.2 + 1.0 + 1e-9

    def test_extension_is_capped(self):
        result = _clip_to_vad(
            [cue(10.0, 11.0)], [vad(0, 10.0, 18.0)], max_extension=0.5
        )
        assert result[0].end == 11.5

    def test_cues_never_bleed_into_the_next_cue(self):
        result = _clip_to_vad(
            [cue(10.0, 11.0, "첫번째"), cue(11.2, 12.0, "두번째")],
            [vad(0, 10.0, 20.0)],
            max_extension=5.0,
        )
        assert len(result) == 2
        assert result[0].end <= result[1].start

    def test_cue_spanning_multiple_vads_clipped_to_union(self):
        result = _clip_to_vad(
            [cue(10.0, 15.0)],
            [vad(0, 9.5, 12.0), vad(1, 13.0, 16.0)],
            max_extension=2.0,
        )
        assert len(result) == 1
        assert result[0].start == 10.0
        assert result[0].end == 16.0  # union end, below cue.end + 2.0

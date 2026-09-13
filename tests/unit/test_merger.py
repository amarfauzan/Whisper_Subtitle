from whisper_subtitle.models import VadSegment, WhisperSegment
from whisper_subtitle.subtitles.merger import merge_vad_and_whisper


def v(index, start, end):
    return VadSegment(index=index, start=start, end=end)


def w(start, end, text):
    return WhisperSegment(start=start, end=end, text=text)


class TestEmptyInputs:
    def test_no_vad_no_whisper(self):
        assert merge_vad_and_whisper([], []) == []

    def test_whisper_without_vad_returns_empty(self):
        # No VAD regions means no speech — whisper is dropped.
        assert merge_vad_and_whisper([], [w(0, 1, "hi")]) == []

    def test_vad_without_whisper_returns_empty(self):
        assert merge_vad_and_whisper([v(0, 0, 5)], []) == []


class TestShortVadMerges:
    def test_single_whisper_in_short_vad(self):
        subs = merge_vad_and_whisper(
            [v(0, 1.0, 2.0)],
            [w(1.1, 1.9, "안녕")],
            max_merge_duration=5.0,
        )
        assert len(subs) == 1
        assert subs[0].start == 1.0
        assert subs[0].end == 2.0
        assert subs[0].text == "안녕"

    def test_two_whisper_segments_merge_into_one(self):
        subs = merge_vad_and_whisper(
            [v(0, 0.0, 2.0)],
            [w(0.1, 0.9, "안녕"), w(1.0, 1.9, "하세요")],
            max_merge_duration=5.0,
        )
        assert len(subs) == 1
        assert subs[0].start == 0.0
        assert subs[0].end == 2.0
        assert subs[0].text == "안녕 하세요"


class TestLongVadSplits:
    def test_whisper_breaks_are_preserved(self):
        subs = merge_vad_and_whisper(
            [v(0, 0.0, 10.0)],
            [w(0.5, 2.0, "first"), w(3.0, 5.0, "second"), w(6.0, 9.0, "third")],
            max_merge_duration=2.0,
        )
        assert len(subs) == 3
        assert [s.text for s in subs] == ["first", "second", "third"]
        assert subs[0].start == 0.5
        assert subs[1].start == 3.0
        assert subs[2].end == 9.0

    def test_clips_whisper_to_vad_boundaries(self):
        # whisper spills outside VAD on both sides; we clip.
        subs = merge_vad_and_whisper(
            [v(0, 2.0, 5.0)],
            [w(1.0, 6.0, "spills")],
            max_merge_duration=1.0,
        )
        assert len(subs) == 1
        assert subs[0].start == 2.0
        assert subs[0].end == 5.0


class TestGrouping:
    def test_whisper_assigned_to_vad_with_max_overlap(self):
        # whisper 3.0-5.0 overlaps vad1 (2-4) by 1s and vad2 (4-6) by 1s.
        # Tie goes to the earlier VAD (strict >), so it lands in vad0's group.
        subs = merge_vad_and_whisper(
            [v(0, 2.0, 4.0), v(1, 4.0, 6.0)],
            [w(3.0, 5.0, "text")],
            max_merge_duration=1.0,
        )
        texts = [s.text for s in subs]
        assert texts == ["text"]  # appears once, not twice

    def test_whisper_with_no_vad_overlap_is_dropped(self):
        subs = merge_vad_and_whisper(
            [v(0, 10.0, 12.0)],
            [w(0.0, 1.0, "hallucination")],
            max_merge_duration=1.0,
        )
        assert subs == []


class TestSorting:
    def test_output_is_sorted_by_start(self):
        # Whisper segments given out of order.
        subs = merge_vad_and_whisper(
            [v(0, 0.0, 20.0)],
            [w(10.0, 11.0, "third"), w(2.0, 3.0, "first"), w(5.0, 6.0, "second")],
            max_merge_duration=1.0,
        )
        assert [s.text for s in subs] == ["first", "second", "third"]

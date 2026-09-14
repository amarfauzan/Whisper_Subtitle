import pytest

from whisper_subtitle.models import Subtitle
from whisper_subtitle.subtitles.srt import (
    ms_to_timestamp,
    render_srt,
    seconds_to_timestamp,
    timestamp_to_ms,
)


from whisper_subtitle.subtitles.srt import parse_srt, render_srt


class TestParseSrt:
    def test_empty(self):
        assert parse_srt("") == []
        assert parse_srt("   \n\n  ") == []

    def test_single(self):
        text = (
            "1\n"
            "00:00:01,000 --> 00:00:02,500\n"
            "hello\n"
        )
        subs = parse_srt(text)
        assert len(subs) == 1
        assert subs[0].start == 1.0
        assert subs[0].end == 2.5
        assert subs[0].text == "hello"

    def test_multiple(self):
        text = (
            "1\n"
            "00:00:01,000 --> 00:00:02,500\n"
            "first\n"
            "\n"
            "2\n"
            "00:00:03,000 --> 00:00:04,000\n"
            "second\n"
        )
        subs = parse_srt(text)
        assert len(subs) == 2
        assert subs[1].text == "second"

    def test_roundtrip(self):
        original = [
            Subtitle(start=1.0, end=2.5, text="hello"),
            Subtitle(start=3.0, end=4.0, text="world"),
        ]
        assert parse_srt(render_srt(original)) == original

    def test_windows_line_endings(self):
        text = "1\r\n00:00:01,000 --> 00:00:02,500\r\nhi\r\n"
        subs = parse_srt(text)
        assert len(subs) == 1
        assert subs[0].text == "hi"

    def test_skips_malformed_blocks(self):
        text = (
            "1\n"
            "00:00:01,000 --> 00:00:02,500\n"
            "good\n"
            "\n"
            "garbage block\n"
            "no arrow here\n"
            "\n"
            "2\n"
            "00:00:03,000 --> 00:00:04,000\n"
            "also good\n"
        )
        subs = parse_srt(text)
        assert len(subs) == 2

class TestTimestampToMs:
    def test_zero(self):
        assert timestamp_to_ms("00:00:00,000") == 0

    def test_simple(self):
        assert timestamp_to_ms("00:01:23,456") == 83_456

    def test_hours(self):
        assert timestamp_to_ms("01:02:03,004") == 3_723_004


class TestMsToTimestamp:
    def test_zero(self):
        assert ms_to_timestamp(0) == "00:00:00,000"

    def test_simple(self):
        assert ms_to_timestamp(83_456) == "00:01:23,456"

    def test_hours(self):
        assert ms_to_timestamp(3_723_004) == "01:02:03,004"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            ms_to_timestamp(-1)


class TestTimestampRoundTrip:
    @pytest.mark.parametrize(
        "ts",
        [
            "00:00:00,000",
            "00:00:00,001",
            "00:01:23,456",
            "01:02:03,004",
            "23:59:59,999",
        ],
    )
    def test_roundtrip(self, ts):
        assert ms_to_timestamp(timestamp_to_ms(ts)) == ts


class TestSecondsToTimestamp:
    def test_zero(self):
        assert seconds_to_timestamp(0.0) == "00:00:00,000"

    def test_one_point_five(self):
        assert seconds_to_timestamp(1.5) == "00:00:01,500"

    def test_rounds_to_next_second(self):
        # 1.9999 s should become 2.000 s, not "00:00:01,1000"
        assert seconds_to_timestamp(1.9999) == "00:00:02,000"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            seconds_to_timestamp(-0.1)


class TestRenderSrt:
    def test_empty(self):
        assert render_srt([]) == "\n"

    def test_single(self):
        subs = [Subtitle(start=1.0, end=2.5, text="hello")]
        expected = (
            "1\n"
            "00:00:01,000 --> 00:00:02,500\n"
            "hello\n"
        )
        assert render_srt(subs) == expected

    def test_multiple(self):
        subs = [
            Subtitle(start=0.0, end=1.0, text="first"),
            Subtitle(start=1.5, end=2.5, text="second"),
        ]
        expected = (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "first\n"
            "\n"
            "2\n"
            "00:00:01,500 --> 00:00:02,500\n"
            "second\n"
        )
        assert render_srt(subs) == expected
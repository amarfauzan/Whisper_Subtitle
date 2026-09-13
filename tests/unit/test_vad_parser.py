import pytest

from whisper_subtitle.exceptions import VadError
from whisper_subtitle.transcription.vad import parse_vad_output


SAMPLE_OUTPUT = """\
whisper_vad_segments_from_probs: found 3 segments
VAD segment 0: start = 0.48, end = 2.34
VAD segment 1: start = 2.88, end = 5.12
VAD segment 2: start = 6.05, end = 8.99
"""


def test_parses_all_segments():
    segments = parse_vad_output(SAMPLE_OUTPUT)
    assert len(segments) == 3


def test_index_values():
    segments = parse_vad_output(SAMPLE_OUTPUT)
    assert [s.index for s in segments] == [0, 1, 2]


def test_start_and_end_values():
    segments = parse_vad_output(SAMPLE_OUTPUT)
    assert segments[0].start == 0.48
    assert segments[0].end == 2.34
    assert segments[2].start == 6.05
    assert segments[2].end == 8.99


def test_duration_property():
    segments = parse_vad_output(SAMPLE_OUTPUT)
    assert segments[0].duration == pytest.approx(1.86)
    assert segments[1].duration == pytest.approx(2.24)


def test_ignores_unrelated_lines():
    text = "some noise\nVAD segment 0: start = 1.0, end = 2.0\nmore noise\n"
    segments = parse_vad_output(text)
    assert len(segments) == 1
    assert segments[0].start == 1.0


def test_empty_output_raises():
    with pytest.raises(VadError):
        parse_vad_output("")


def test_output_without_segments_raises():
    with pytest.raises(VadError):
        parse_vad_output("whisper_vad_segments_from_probs: found 0 segments\n")

import pytest

from whisper_subtitle.models import Subtitle
from whisper_subtitle.subtitles.timeline import shift_timeline


def test_empty_list():
    assert shift_timeline([], 10.0) == []


def test_zero_offset_returns_equal_subs():
    subs = [Subtitle(start=1.0, end=2.0, text="hi")]
    result = shift_timeline(subs, 0.0)
    assert result == subs


def test_zero_offset_returns_a_copy_not_the_same_list():
    subs = [Subtitle(start=1.0, end=2.0, text="hi")]
    result = shift_timeline(subs, 0.0)
    result.append(Subtitle(start=5.0, end=6.0, text="new"))
    assert len(subs) == 1  # original untouched


def test_positive_offset_shifts_everything():
    subs = [
        Subtitle(start=0.0, end=1.0, text="first"),
        Subtitle(start=1.5, end=2.5, text="second"),
    ]
    result = shift_timeline(subs, 600.0)
    assert result[0].start == 600.0
    assert result[0].end == 601.0
    assert result[1].start == 601.5
    assert result[1].end == 602.5


def test_fractional_offset():
    subs = [Subtitle(start=0.0, end=1.0, text="hi")]
    result = shift_timeline(subs, 599.84)
    assert result[0].start == pytest.approx(599.84)
    assert result[0].end == pytest.approx(600.84)


def test_text_is_preserved():
    subs = [Subtitle(start=0.0, end=1.0, text="안녕 하세요")]
    result = shift_timeline(subs, 5.0)
    assert result[0].text == "안녕 하세요"


def test_original_list_is_not_mutated():
    original = Subtitle(start=1.0, end=2.0, text="hi")
    subs = [original]
    shift_timeline(subs, 100.0)
    assert subs[0] is original
    assert subs[0].start == 1.0

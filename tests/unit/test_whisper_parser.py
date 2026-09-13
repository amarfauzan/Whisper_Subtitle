import json

import pytest

from whisper_subtitle.exceptions import WhisperError
from whisper_subtitle.transcription.whisper_runner import parse_whisper_json


def write_json(tmp_path, payload):
    path = tmp_path / "output.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_segment(offset_from, offset_to, text):
    return {
        "timestamps": {"from": "00:00:00,000", "to": "00:00:00,000"},
        "offsets": {"from": offset_from, "to": offset_to},
        "text": text,
    }


def test_parses_single_segment(tmp_path):
    path = write_json(tmp_path, {
        "transcription": [make_segment(0, 1500, " 안녕하세요")]
    })
    segments = parse_whisper_json(path)
    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert segments[0].end == 1.5
    assert segments[0].text == "안녕하세요"


def test_parses_multiple_segments(tmp_path):
    path = write_json(tmp_path, {
        "transcription": [
            make_segment(0, 1500, " 안녕하세요"),
            make_segment(1500, 3200, " 반갑습니다"),
            make_segment(3200, 5000, " 오늘은 좋은 날이네요"),
        ]
    })
    segments = parse_whisper_json(path)
    assert len(segments) == 3
    assert [s.start for s in segments] == [0.0, 1.5, 3.2]
    assert [s.end for s in segments] == [1.5, 3.2, 5.0]


def test_strips_leading_and_trailing_spaces(tmp_path):
    path = write_json(tmp_path, {
        "transcription": [make_segment(0, 1000, "   hello world   ")]
    })
    segments = parse_whisper_json(path)
    assert segments[0].text == "hello world"


def test_skips_whitespace_only_segments(tmp_path):
    path = write_json(tmp_path, {
        "transcription": [
            make_segment(0, 500, " "),
            make_segment(500, 1000, "hello"),
            make_segment(1000, 1500, ""),
        ]
    })
    segments = parse_whisper_json(path)
    assert len(segments) == 1
    assert segments[0].text == "hello"


def test_missing_file_raises(tmp_path):
    with pytest.raises(WhisperError):
        parse_whisper_json(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("this is not json", encoding="utf-8")
    with pytest.raises(WhisperError):
        parse_whisper_json(path)






def test_missing_offsets_raises(tmp_path):
    path = write_json(tmp_path, {
        "transcription": [{"timestamps": {}, "text": "hello"}]
    })
    with pytest.raises(WhisperError):
        parse_whisper_json(path)

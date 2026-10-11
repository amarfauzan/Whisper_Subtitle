
from whisper_subtitle.models import OcrEvent, WhisperSegment
from whisper_subtitle.persistence import (
    load_ocr_events,
    load_transcription,
    save_ocr_events,
    save_transcription,
)


def test_transcription_roundtrip(tmp_path):
    segs = [
        WhisperSegment(0.5, 2.5, "안녕하세요"),
        WhisperSegment(3.0, 4.5, "반갑습니다"),
    ]
    path = tmp_path / "primary.json"
    save_transcription(segs, path, source="primary", model="test", chunk="x")
    loaded = load_transcription(path)
    assert loaded == segs
    assert path.with_suffix(".srt").exists()


def test_ocr_roundtrip(tmp_path):
    events = [
        OcrEvent(1.0, 3.0, (10, 20, 100, 50), "TOKYO", confidence=0.95, frame_count=4),
    ]
    path = tmp_path / "ocr.json"
    save_ocr_events(events, path, chunk="x", frame_height=720)
    loaded = load_ocr_events(path)
    assert loaded == events
    assert path.with_suffix(".srt").exists()
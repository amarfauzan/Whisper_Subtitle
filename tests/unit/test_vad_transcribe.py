"""Tests for VAD-segment transcription."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from whisper_subtitle.config import VadSegmentsConfig, WhisperConfig
from whisper_subtitle.exceptions import PipelineError
from whisper_subtitle.models import VadSegment, WhisperSegment
from whisper_subtitle.transcription.vad_transcribe import (
    _merge_close_segments,
    _padded_range,
    transcribe_by_vad_segments,
)


def make_cfg(**overrides) -> WhisperConfig:
    base = dict(
        mode="cli",
        cli=Path("/fake/cli"),
        server_exe=Path("/fake/server"),
        model=Path("/fake/model.bin"),
        vad_model=Path("/fake/vad.bin"),
        language="ko",
        beam_size=3,
        best_of=3,
        max_line_length=30,
        server_host="127.0.0.1",
        server_port=8080,
        server_threads=8,
        server_startup_timeout=60.0,
        server_request_timeout=300.0,
        strategy="vad_segments",
        vad_segments=VadSegmentsConfig(
            padding=0.15,
            merge_gap=0.5,
            min_segment_duration=0.3,
        ),
    )
    base.update(overrides)
    return WhisperConfig(**base)


def vs(idx, start, end):
    return VadSegment(index=idx, start=start, end=end)


def ws(start, end, text):
    return WhisperSegment(start=start, end=end, text=text)


# ---------------------------------------------------------------- merging


class TestMergeCloseSegments:
    def test_empty(self):
        assert _merge_close_segments([], 0.5) == []

    def test_single(self):
        s = vs(0, 1.0, 2.0)
        assert _merge_close_segments([s], 0.5) == [s]

    def test_separate_unchanged(self):
        segs = [vs(0, 1.0, 2.0), vs(1, 5.0, 6.0)]
        assert len(_merge_close_segments(segs, 0.5)) == 2

    def test_close_merged(self):
        segs = [vs(0, 1.0, 2.0), vs(1, 2.3, 3.5)]
        result = _merge_close_segments(segs, 0.5)
        assert len(result) == 1
        assert result[0].start == 1.0
        assert result[0].end == 3.5

    def test_chain_merge(self):
        segs = [vs(0, 1.0, 2.0), vs(1, 2.3, 3.5), vs(2, 3.7, 4.5)]
        result = _merge_close_segments(segs, 0.5)
        assert len(result) == 1
        assert result[0].end == 4.5

    def test_unsorted_input(self):
        segs = [vs(1, 5.0, 6.0), vs(0, 1.0, 2.0)]
        result = _merge_close_segments(segs, 0.5)
        assert result[0].start == 1.0
        assert result[1].start == 5.0


# ---------------------------------------------------------------- padding


class TestPaddedRange:
    def test_single(self):
        segs = [vs(0, 5.0, 6.0)]
        start, end = _padded_range(0, segs, padding=0.15)
        assert start == pytest.approx(4.85)
        assert end == pytest.approx(6.15)

    def test_clamped_at_zero(self):
        segs = [vs(0, 0.05, 1.0)]
        start, _ = _padded_range(0, segs, padding=0.15)
        assert start == 0.0

    def test_bounded_by_previous_midpoint(self):
        segs = [vs(0, 1.0, 2.0), vs(1, 2.1, 3.0)]
        start, _ = _padded_range(1, segs, padding=0.15)
        # prev_end=2.0, seg.start=2.1 → midpoint=2.05
        assert start == pytest.approx(2.05)

    def test_bounded_by_next_midpoint(self):
        segs = [vs(0, 1.0, 2.0), vs(1, 3.0, 4.0)]
        _, end = _padded_range(0, segs, padding=0.15)
        # seg.end=2.0, next_start=3.0 → midpoint=2.5
        assert end == pytest.approx(2.15)


# ---------------------------------------------------------------- main


def _stub_extract(monkeypatch):
    """Replace extract_clip with a version that writes a dummy file."""

    def fake(src, start, end, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"dummy")

    monkeypatch.setattr(
        "whisper_subtitle.transcription.vad_transcribe.extract_clip",
        fake,
    )


class TestTranscribeByVadSegments:
    def test_empty_vad(self, tmp_path, monkeypatch):
        _stub_extract(monkeypatch)
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        backend = MagicMock()

        result = transcribe_by_vad_segments(audio, [], backend, make_cfg())

        assert result == []
        backend.transcribe.assert_not_called()

    def test_all_below_min_duration(self, tmp_path, monkeypatch):
        _stub_extract(monkeypatch)
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        backend = MagicMock()
        segs = [vs(0, 0.0, 0.1), vs(1, 1.0, 1.2)]  # both < 0.3s

        result = transcribe_by_vad_segments(audio, segs, backend, make_cfg())

        assert result == []
        backend.transcribe.assert_not_called()

    def test_single_segment(self, tmp_path, monkeypatch):
        _stub_extract(monkeypatch)
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        backend = MagicMock()
        backend.transcribe.return_value.segments = [ws(0.5, 1.0, "hello")]

        segs = [vs(0, 2.0, 4.0)]
        result = transcribe_by_vad_segments(audio, segs, backend, make_cfg())

        assert len(result) == 1
        # Clip starts at 2.0 - 0.15 = 1.85
        # Whisper segment at 0.5-1.0 → 2.35-2.85
        assert result[0].start == pytest.approx(2.35)
        assert result[0].end == pytest.approx(2.85)
        assert result[0].text == "hello"

    def test_offsets_applied_per_clip(self, tmp_path, monkeypatch):
        _stub_extract(monkeypatch)
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        backend = MagicMock()
        backend.transcribe.return_value.segments = [ws(0.0, 1.0, "hi")]

        segs = [vs(0, 5.0, 6.0), vs(1, 20.0, 21.0)]
        result = transcribe_by_vad_segments(audio, segs, backend, make_cfg())

        assert len(result) == 2
        assert result[0].start == pytest.approx(4.85)
        assert result[1].start == pytest.approx(19.85)

    def test_backend_failure_skips_clip(self, tmp_path, monkeypatch):
        _stub_extract(monkeypatch)
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        backend = MagicMock()
        backend.transcribe.side_effect = [
            PipelineError("server error"),
            MagicMock(segments=[ws(0.0, 1.0, "ok")]),
        ]

        segs = [vs(0, 5.0, 6.0), vs(1, 20.0, 21.0)]
        result = transcribe_by_vad_segments(audio, segs, backend, make_cfg())

        assert len(result) == 1
        assert result[0].text == "ok"

    def test_results_sorted(self, tmp_path, monkeypatch):
        _stub_extract(monkeypatch)
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        backend = MagicMock()
        backend.transcribe.return_value.segments = [
            ws(2.0, 3.0, "second"),
            ws(0.0, 1.0, "first"),
        ]

        result = transcribe_by_vad_segments(
            audio, [vs(0, 5.0, 6.0)], backend, make_cfg()
        )

        assert [s.text for s in result] == ["first", "second"]

    def test_merges_close_vads_before_transcribing(self, tmp_path, monkeypatch):
        _stub_extract(monkeypatch)
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        backend = MagicMock()
        backend.transcribe.return_value.segments = [ws(0.0, 1.0, "x")]

        # Two VADs 0.3s apart, merge_gap=0.5 → merged into one clip
        segs = [vs(0, 5.0, 6.0), vs(1, 6.3, 7.0)]
        transcribe_by_vad_segments(audio, segs, backend, make_cfg())

        assert backend.transcribe.call_count == 1
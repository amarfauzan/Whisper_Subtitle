"""Tests for VAD-segment transcription."""

from dataclasses import replace
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

from whisper_subtitle.transcription.vad_transcribe import (
    _build_batch_clip,
    _coverage_ok,
    _group_into_batches,
    _map_batch_time,
    _map_whisper_segment,
    _MappingEntry,
    transcribe_batched_vad_segments,
    transcribe_vad_segments,
)

import wave


FRAMERATE = 16000


def make_wav(path: Path, seconds: float) -> None:
    """Create a silent 16-bit mono WAV of the given duration."""
    frames = int(seconds * FRAMERATE)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(FRAMERATE)
        f.writeframes(b"\x00\x00" * frames)

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
            max_batch_duration=0.0,     # ← new
            silence_ms=200,             # ← new
            min_coverage=0.9,           # ← new
        ),
    )
    base.update(overrides)
    return WhisperConfig(**base)


class TestGroupIntoBatches:
    def test_empty(self):
        assert _group_into_batches([], 30.0, 0.2) == []

    def test_single_batch(self):
        segs = [vs(0, 0, 3.0), vs(1, 3.5, 6.0)]
        batches = _group_into_batches(segs, 30.0, 0.2)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_splits_when_over_max(self):
        segs = [vs(i, i * 10, i * 10 + 8.0) for i in range(4)]
        batches = _group_into_batches(segs, 20.0, 0.2)
        # Each seg is 8s. 8+0.2+8+0.2+8 = 24.6 > 20, so 2 per batch
        assert len(batches) == 2
        assert all(len(b) <= 2 for b in batches)

    def test_single_oversized_segment_gets_own_batch(self):
        segs = [vs(0, 0, 50.0), vs(1, 60.0, 62.0)]
        batches = _group_into_batches(segs, 30.0, 0.2)
        assert len(batches) == 2
        assert len(batches[0]) == 1  # oversized alone
        assert len(batches[1]) == 1


class TestBuildBatchClip:
    def test_single_segment(self, tmp_path):
        audio = tmp_path / "src.wav"
        make_wav(audio, 10.0)
        out = tmp_path / "batch.wav"
        mapping, duration = _build_batch_clip(
            audio, [vs(0, 2.0, 5.0)], out, silence_ms=200,
        )
        assert abs(duration - 3.0) < 0.01
        assert len(mapping) == 1
        assert mapping[0].is_speech
        assert mapping[0].original_start == 2.0
        assert mapping[0].original_end == 5.0

    def test_two_segments_with_silence(self, tmp_path):
        audio = tmp_path / "src.wav"
        make_wav(audio, 10.0)
        out = tmp_path / "batch.wav"
        segs = [vs(0, 1.0, 3.0), vs(1, 5.0, 7.0)]
        mapping, duration = _build_batch_clip(audio, segs, out, silence_ms=200)
        # 2s + 0.2s + 2s = 4.2s
        assert abs(duration - 4.2) < 0.05
        assert len(mapping) == 3
        assert mapping[0].is_speech
        assert not mapping[1].is_speech
        assert mapping[2].is_speech


class TestMapping:
    def test_inside_speech(self):
        mapping = [_MappingEntry(0.0, 2.0, 10.0, 12.0, True)]
        assert _map_batch_time(0.0, mapping) == 10.0
        assert _map_batch_time(1.0, mapping) == 11.0
        assert _map_batch_time(2.0, mapping) == 12.0

    def test_inside_silence(self):
        mapping = [
            _MappingEntry(0.0, 2.0, 10.0, 12.0, True),
            _MappingEntry(2.0, 2.2, 12.0, 12.0, False),  # silence gap
            _MappingEntry(2.2, 4.0, 15.0, 17.0, True),
        ]
        # Position inside the silence gap
        assert _map_batch_time(2.1, mapping) == 12.0  # clamps to boundary

    def test_out_of_range_clamps(self):
        mapping = [_MappingEntry(0.0, 2.0, 10.0, 12.0, True)]
        assert _map_batch_time(-1.0, mapping) == 10.0
        assert _map_batch_time(5.0, mapping) == 12.0

    def test_map_whisper_segment(self):
        mapping = [_MappingEntry(0.0, 3.0, 100.0, 103.0, True)]
        wseg = WhisperSegment(start=0.5, end=2.5, text="hi")
        mapped = _map_whisper_segment(wseg, mapping)
        assert mapped.start == 100.5
        assert mapped.end == 102.5
        assert mapped.text == "hi"


class TestCoverage:
    def test_full_coverage(self):
        segs = [WhisperSegment(start=0.0, end=10.0, text="x")]
        assert _coverage_ok(segs, 10.0, 0.9)

    def test_under_coverage(self):
        segs = [WhisperSegment(start=0.0, end=5.0, text="x")]
        assert not _coverage_ok(segs, 10.0, 0.9)

    def test_empty(self):
        assert not _coverage_ok([], 10.0, 0.9)


class TestBatchedTranscription:
    def test_dispatches_to_batched_when_enabled(self, tmp_path, monkeypatch):
        audio = tmp_path / "src.wav"
        make_wav(audio, 10.0)
        backend = MagicMock()
        backend.transcribe.return_value.segments = [ws(0.0, 3.0, "x")]
        cfg = make_cfg()
        cfg = replace(cfg, vad_segments=replace(
            cfg.vad_segments, max_batch_duration=30.0,
        ))
        segs = [vs(0, 1.0, 4.0)]
        transcribe_vad_segments(audio, segs, backend, cfg)
        assert backend.transcribe.call_count == 1

    def test_dispatches_to_per_segment_when_disabled(self, tmp_path, monkeypatch):
        audio = tmp_path / "src.wav"
        make_wav(audio, 10.0)
        backend = MagicMock()
        backend.transcribe.return_value.segments = [ws(0.0, 3.0, "x")]
        cfg = make_cfg()  # max_batch_duration defaults to 0
        segs = [vs(0, 1.0, 4.0)]
        transcribe_vad_segments(audio, segs, backend, cfg)
        assert backend.transcribe.call_count == 1

    def test_fallback_on_under_coverage(self, tmp_path, monkeypatch):
        audio = tmp_path / "src.wav"
        make_wav(audio, 20.0)
        backend = MagicMock()
        # First call (batch) returns too-short output
        # Subsequent calls (fallback per-segment) return normal output
        backend.transcribe.side_effect = [
            MagicMock(segments=[ws(0.0, 1.0, "short")]),  # under-covered
            MagicMock(segments=[ws(0.0, 3.0, "ok1")]),
            MagicMock(segments=[ws(0.0, 3.0, "ok2")]),
        ]
        cfg = make_cfg()
        cfg = replace(cfg, vad_segments=replace(
            cfg.vad_segments, max_batch_duration=30.0, min_coverage=0.9,
        ))
        segs = [vs(0, 1.0, 5.0), vs(1, 6.0, 10.0)]

        result = transcribe_batched_vad_segments(audio, segs, backend, cfg)
        # Batch failed → fell back to 2 per-segment calls
        assert backend.transcribe.call_count == 3
        assert any("ok1" in s.text for s in result)




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
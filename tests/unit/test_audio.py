import wave
from pathlib import Path

import pytest

from whisper_subtitle.exceptions import PipelineError
from whisper_subtitle.media.audio import extract_clip


FRAMERATE = 16000


def make_wav(path: Path, seconds: float) -> None:
    """Create a silent 16-bit mono WAV of the given duration."""
    frames = int(seconds * FRAMERATE)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(FRAMERATE)
        f.writeframes(b"\x00\x00" * frames)


def read_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


class TestExtractClip:
    def test_extracts_correct_duration(self, tmp_path):
        src = tmp_path / "src.wav"
        make_wav(src, 10.0)
        dst = tmp_path / "clip.wav"

        extract_clip(src, 2.0, 5.0, dst)

        assert dst.exists()
        assert abs(read_duration(dst) - 3.0) < 0.01

    def test_start_at_zero(self, tmp_path):
        src = tmp_path / "src.wav"
        make_wav(src, 5.0)
        dst = tmp_path / "clip.wav"

        extract_clip(src, 0.0, 2.0, dst)

        assert abs(read_duration(dst) - 2.0) < 0.01

    def test_extends_past_end_is_clamped(self, tmp_path):
        src = tmp_path / "src.wav"
        make_wav(src, 5.0)
        dst = tmp_path / "clip.wav"

        extract_clip(src, 4.0, 100.0, dst)

        assert abs(read_duration(dst) - 1.0) < 0.01

    def test_negative_start_is_clamped_to_zero(self, tmp_path):
        src = tmp_path / "src.wav"
        make_wav(src, 5.0)
        dst = tmp_path / "clip.wav"

        extract_clip(src, -1.0, 2.0, dst)

        assert abs(read_duration(dst) - 2.0) < 0.01

    def test_end_before_start_raises(self, tmp_path):
        src = tmp_path / "src.wav"
        make_wav(src, 5.0)
        dst = tmp_path / "clip.wav"

        with pytest.raises(PipelineError):
            extract_clip(src, 3.0, 1.0, dst)

    def test_empty_range_past_file_end_raises(self, tmp_path):
        src = tmp_path / "src.wav"
        make_wav(src, 5.0)
        dst = tmp_path / "clip.wav"

        with pytest.raises(PipelineError):
            extract_clip(src, 10.0, 12.0, dst)

    def test_missing_source_raises(self, tmp_path):
        dst = tmp_path / "clip.wav"

        with pytest.raises(PipelineError):
            extract_clip(tmp_path / "nope.wav", 0.0, 1.0, dst)

    def test_creates_parent_directory(self, tmp_path):
        src = tmp_path / "src.wav"
        make_wav(src, 5.0)
        dst = tmp_path / "nested" / "dir" / "clip.wav"

        extract_clip(src, 0.0, 1.0, dst)

        assert dst.exists()

    def test_output_format_matches_source(self, tmp_path):
        src = tmp_path / "src.wav"
        make_wav(src, 5.0)
        dst = tmp_path / "clip.wav"

        extract_clip(src, 1.0, 2.0, dst)

        with wave.open(str(dst), "rb") as f:
            assert f.getnchannels() == 1
            assert f.getsampwidth() == 2
            assert f.getframerate() == FRAMERATE

    def test_original_file_unchanged(self, tmp_path):
        src = tmp_path / "src.wav"
        make_wav(src, 5.0)
        original_size = src.stat().st_size
        dst = tmp_path / "clip.wav"

        extract_clip(src, 1.0, 2.0, dst)

        assert src.stat().st_size == original_size
from pathlib import Path

import pytest

from whisper_subtitle.exceptions import FFmpegError
from whisper_subtitle.media.ffmpeg import _read_chunks_from_csv


def test_parses_standard_csv(tmp_path):
    csv_path = tmp_path / "segments.csv"
    csv_path.write_text(
        "video_000.mp4,0.000000,600.000000\n"
        "video_001.mp4,600.000000,1200.000000\n"
        "video_002.mp4,1200.000000,1800.000000\n",
        encoding="utf-8",
    )

    chunks = _read_chunks_from_csv(csv_path, tmp_path)

    assert len(chunks) == 3
    assert chunks[0].path == tmp_path / "video_000.mp4"
    assert chunks[0].offset == 0.0
    assert chunks[1].offset == 600.0
    assert chunks[2].offset == 1200.0


def test_uses_basename_of_paths_with_directories(tmp_path):
    csv_path = tmp_path / "segments.csv"
    csv_path.write_text(
        "/some/absolute/path/video_000.mp4,0.0,600.0\n",
        encoding="utf-8",
    )

    chunks = _read_chunks_from_csv(csv_path, tmp_path)

    assert chunks[0].path == tmp_path / "video_000.mp4"


def test_fractional_offsets(tmp_path):
    csv_path = tmp_path / "segments.csv"
    csv_path.write_text(
        "video_000.mp4,0.000000,599.840000\n"
        "video_001.mp4,599.840000,1199.680000\n",
        encoding="utf-8",
    )

    chunks = _read_chunks_from_csv(csv_path, tmp_path)

    assert chunks[1].offset == pytest.approx(599.84)


def test_skips_short_rows(tmp_path):
    csv_path = tmp_path / "segments.csv"
    csv_path.write_text(
        "video_000.mp4,0.0,600.0\n"
        "malformed\n"
        "video_001.mp4,600.0,1200.0\n",
        encoding="utf-8",
    )

    chunks = _read_chunks_from_csv(csv_path, tmp_path)

    assert len(chunks) == 2


def test_empty_csv_raises(tmp_path):
    csv_path = tmp_path / "segments.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(FFmpegError):
        _read_chunks_from_csv(csv_path, tmp_path)

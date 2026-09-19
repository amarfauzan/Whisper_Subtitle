"""End-to-end test: run the full pipeline on a tiny fixture video.

Requires ffmpeg + whisper-cli + the VAD binary to be installed and
configured in .env. Skipped if the fixture is missing.
"""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from whisper_subtitle.config import load_config
from whisper_subtitle.pipeline import process_video


FIXTURE = Path(__file__).parent.parent / "fixtures" / "clip_3s.mp4"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture video not present")
def test_pipeline_runs_end_to_end(tmp_path):
    local_video = tmp_path / "clip.mp4"
    shutil.copy(FIXTURE, local_video)

    # load_config() returns the full Config (top-level).
    # default_whisper_config() returns only WhisperConfig — wrong type here.
    cfg = load_config()
    cfg = replace(cfg, chunk_length_seconds=2)

    result = process_video(local_video, cfg)

    assert result.exists()
    assert result.suffix == ".srt"

    text = result.read_text(encoding="utf-8")
    if text.strip():
        assert " --> " in text
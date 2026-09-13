"""End-to-end test: run the full pipeline on a tiny fixture video.

Requires ffmpeg + whisper-cli + the VAD binary to be installed and
configured in .env. Skipped if the fixture is missing.
"""

from pathlib import Path

import pytest

from whisper_subtitle.config import load_config
from whisper_subtitle.pipeline import process_video

FIXTURE = Path(__file__).parent.parent / "fixtures" / "clip_3s.mp4"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture video not present")
def test_pipeline_runs_end_to_end(tmp_path):
    # Copy the fixture into tmp_path so the pipeline writes there, not in tests/
    import shutil
    local_video = tmp_path / "clip.mp4"
    shutil.copy(FIXTURE, local_video)

    cfg = load_config()

    # Run with a tiny chunk length so we exercise the split + merge path.
    # Config is frozen, so build a modified copy.
    from dataclasses import replace
    cfg = replace(cfg, chunk_length_seconds=2)

    result = process_video(local_video, cfg)

    assert result.exists()
    assert result.suffix == ".srt"
    text = result.read_text(encoding="utf-8")
    # Even if whisper produces nothing, the file should exist and be valid.
    # We assert structure, not content.
    if text.strip():
        assert " --> " in text

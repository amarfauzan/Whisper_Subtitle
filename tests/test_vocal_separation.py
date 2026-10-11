"""Tests for the vocal separation module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper_subtitle.config import UvrConfig
from whisper_subtitle.exceptions import PipelineError
from whisper_subtitle.media.vocal_separation import separate_vocals


def make_cfg(**overrides) -> UvrConfig:
    base = dict(
        enabled=True,
        model="UVR_MDXNET_Main.onnx",
        output_single_stem="Vocals",
        model_dir="",
    )
    base.update(overrides)
    return UvrConfig(**base)


def test_missing_input_raises(tmp_path):
    with pytest.raises(PipelineError, match="not found"):
        separate_vocals(tmp_path / "nope.wav", make_cfg())


def test_existing_output_is_reused(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    vocals = tmp_path / "test_vocals.wav"
    vocals.write_bytes(b"fake vocals")

    # Vocals file is newer (just written), so should be reused.
    # Patch Separator to raise if called — proves we skipped separation.
    with patch(
        "whisper_subtitle.media.vocal_separation.Separator",
        side_effect=AssertionError("Separator should not be instantiated"),
    ):
        result = separate_vocals(audio, make_cfg())

    assert result == vocals


def test_runs_separation_when_output_missing(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    vocals = tmp_path / "test_vocals.wav"

    mock_separator = MagicMock()
    mock_separator.separate.return_value = [str(vocals)]
    # Simulate the library writing the file
    def fake_separate(path):
        vocals.write_bytes(b"separated")
        return [str(vocals)]
    mock_separator.separate.side_effect = fake_separate

    with patch(
        "whisper_subtitle.media.vocal_separation.Separator",
        return_value=mock_separator,
    ):
        result = separate_vocals(audio, make_cfg())

    assert result == vocals
    assert vocals.exists()
    mock_separator.load_model.assert_called_once_with(
        model_filename="UVR_MDXNET_Main.onnx"
    )


def test_separator_failure_raises(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    mock_separator = MagicMock()
    mock_separator.separate.side_effect = RuntimeError("boom")

    with patch(
        "whisper_subtitle.media.vocal_separation.Separator",
        return_value=mock_separator,
    ):
        with pytest.raises(PipelineError, match="separation failed"):
            separate_vocals(audio, make_cfg())
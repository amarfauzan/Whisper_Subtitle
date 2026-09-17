from unittest.mock import MagicMock, patch

import pytest

from whisper_subtitle.config import WhisperConfig,VadSegmentsConfig
from whisper_subtitle.exceptions import WhisperError
from whisper_subtitle.transcription.whisper_backend import (
    CliBackend,
    ServerBackend,
    _parse_server_segments,
    make_whisper_backend,
)
from pathlib import Path




def make_cfg(**overrides) -> WhisperConfig:
    base = dict(
        mode="cli",
        cli=Path("/fake/whisper-cli"),
        server_exe=Path("/fake/whisper-server"),
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
        strategy="whole",
        vad_segments=VadSegmentsConfig(
            padding=0.15,
            merge_gap=0.5,
            min_segment_duration=0.3,
        ),
    )
    base.update(overrides)
    return WhisperConfig(**base)


class TestFactory:
    def test_cli_mode(self):
        assert isinstance(make_whisper_backend(make_cfg(mode="cli")), CliBackend)

    def test_server_mode(self):
        assert isinstance(make_whisper_backend(make_cfg(mode="server")), ServerBackend)

    def test_unknown_mode_raises(self):
        with pytest.raises(WhisperError):
            make_whisper_backend(make_cfg(mode="banana"))


class TestParseServerSegments:
    def test_empty_payload(self):
        assert _parse_server_segments({}) == []
        assert _parse_server_segments({"segments": []}) == []

    def test_single_segment(self):
        payload = {"segments": [{"start": 1.5, "end": 2.5, "text": "hello"}]}
        segs = _parse_server_segments(payload)
        assert len(segs) == 1
        assert segs[0].start == 1.5
        assert segs[0].end == 2.5
        assert segs[0].text == "hello"

    def test_strips_whitespace(self):
        payload = {"segments": [{"start": 0.0, "end": 1.0, "text": "  hi  "}]}
        assert _parse_server_segments(payload)[0].text == "hi"

    def test_skips_empty_text(self):
        payload = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": ""},
                {"start": 1.0, "end": 2.0, "text": "hi"},
                {"start": 2.0, "end": 3.0, "text": "   "},
            ]
        }
        segs = _parse_server_segments(payload)
        assert len(segs) == 1
        assert segs[0].text == "hi"

    def test_missing_start_raises(self):
        with pytest.raises(WhisperError):
            _parse_server_segments({"segments": [{"end": 1.0, "text": "hi"}]})

    def test_non_numeric_start_raises(self):
        with pytest.raises(WhisperError):
            _parse_server_segments(
                {"segments": [{"start": "abc", "end": 1.0, "text": "hi"}]}
            )


class TestCliBackend:
    def test_enter_exit_are_noops(self):
        backend = CliBackend(make_cfg())
        with backend as b:
            assert b is backend

    @patch("whisper_subtitle.transcription.whisper_backend.run_whisper_cli")
    def test_transcribe_calls_cli(self, mock_run):
        mock_run.return_value = MagicMock()
        backend = CliBackend(make_cfg())
        backend.transcribe(Path("/tmp/a.wav"))
        mock_run.assert_called_once()


class TestServerBackend:
    @patch("whisper_subtitle.transcription.whisper_backend._wait_for_port")
    @patch("whisper_subtitle.transcription.whisper_backend._start_server")
    def test_enter_starts_and_waits(self, mock_start, mock_wait):
        fake_proc = MagicMock()
        mock_start.return_value = fake_proc

        backend = ServerBackend(make_cfg(mode="server"))
        with backend as b:
            assert b is backend

        mock_start.assert_called_once()
        mock_wait.assert_called_once()
        fake_proc.terminate.assert_called_once()

    @patch("whisper_subtitle.transcription.whisper_backend._wait_for_port")
    @patch("whisper_subtitle.transcription.whisper_backend._start_server")
    def test_enter_kills_server_on_wait_failure(self, mock_start, mock_wait):
        fake_proc = MagicMock()
        mock_start.return_value = fake_proc
        mock_wait.side_effect = WhisperError("timeout")

        backend = ServerBackend(make_cfg(mode="server"))
        with pytest.raises(WhisperError):
            backend.__enter__()

        fake_proc.kill.assert_called_once()

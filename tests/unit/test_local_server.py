"""Tests for the local llama-server lifecycle manager."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper_subtitle.config import TranslationConfig
from whisper_subtitle.exceptions import TranslationError
from whisper_subtitle.translation import local_server
from whisper_subtitle.translation.local_server import LlamaServer


def make_cfg(tmp_path: Path, **overrides) -> TranslationConfig:
    exe = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    exe.write_text("")
    model.write_text("")
    base = dict(
        api_key="",
        base_url="",
        model="translategemma",
        disable_thinking=False,
        source_language="Korean",
        target_language="English",
        batch_size=100,
        max_retries=3,
        min_split_size=50,
        context="",
        provider="local",
        request_timeout=180.0,
        server_exe=exe,
        server_model=model,
        server_port=8091,
    )
    base.update(overrides)
    return TranslationConfig(**base)


class TestValidation:
    def test_missing_server_exe(self, tmp_path):
        cfg = make_cfg(tmp_path, server_exe=tmp_path / "nope")
        with pytest.raises(TranslationError, match="llama-server not found"):
            LlamaServer(cfg).__enter__()

    def test_missing_model(self, tmp_path):
        cfg = make_cfg(tmp_path, server_model=tmp_path / "nope.gguf")
        with pytest.raises(TranslationError, match="model not found"):
            LlamaServer(cfg).__enter__()

    @patch.object(local_server, "_port_is_open", return_value=True)
    def test_port_in_use_rejected(self, mock_port, tmp_path):
        with pytest.raises(TranslationError, match="already in use"):
            LlamaServer(make_cfg(tmp_path)).__enter__()


class TestLifecycle:
    @patch.object(local_server, "_port_is_open", return_value=False)
    @patch.object(local_server, "_wait_for_port")
    @patch.object(local_server.subprocess, "Popen")
    def test_starts_and_stops(self, mock_popen, mock_wait, mock_port, tmp_path):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc

        cfg = make_cfg(tmp_path)
        with LlamaServer(cfg) as server:
            assert server.process is proc
            args = mock_popen.call_args.args[0]
            assert args[0] == str(cfg.server_exe)
            assert args[1] == "-m"
            assert args[2] == str(cfg.server_model)
            assert "--port" in args
            assert str(cfg.server_port) in args
            assert "--no-jinja" in args

        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5.0)
        proc.kill.assert_not_called()

    @patch.object(local_server, "_port_is_open", return_value=False)
    @patch.object(local_server, "_wait_for_port")
    @patch.object(local_server.subprocess, "Popen")
    def test_kills_when_terminate_times_out(
        self, mock_popen, mock_wait, mock_port, tmp_path
    ):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("llama-server", 5), None]
        mock_popen.return_value = proc

        with LlamaServer(make_cfg(tmp_path)):
            pass

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2

    @patch.object(local_server, "_port_is_open", return_value=False)
    @patch.object(local_server, "_wait_for_port", side_effect=TranslationError("timeout"))
    @patch.object(local_server.subprocess, "Popen")
    def test_startup_timeout_stops_process(
        self, mock_popen, mock_wait, mock_port, tmp_path
    ):
        proc = MagicMock()
        mock_popen.return_value = proc

        with pytest.raises(TranslationError):
            LlamaServer(make_cfg(tmp_path)).__enter__()

        proc.terminate.assert_called_once()

    @patch.object(local_server, "_port_is_open", return_value=False)
    @patch.object(local_server, "_wait_for_port")
    @patch.object(local_server.subprocess, "Popen")
    def test_server_that_exits_during_startup_is_detected(
        self, mock_popen, mock_wait, mock_port, tmp_path
    ):
        proc = MagicMock()
        proc.poll.return_value = 1  # process died
        mock_popen.return_value = proc

        with pytest.raises(TranslationError, match="exited during startup"):
            LlamaServer(make_cfg(tmp_path)).__enter__()

        proc.terminate.assert_called_once()

"""Manage a local llama-server process for the local translation provider.

The server is llama.cpp's OpenAI-compatible endpoint, started around the
translation step and shut down afterwards.
"""

import logging
import socket
import subprocess
import time
from typing import Self

from whisper_subtitle.config import TranslationConfig
from whisper_subtitle.exceptions import TranslationError

log = logging.getLogger(__name__)

_STARTUP_TIMEOUT = 120.0
_POLL_INTERVAL = 0.5
_SHUTDOWN_TIMEOUT = 5.0


class LlamaServer:
    """Context manager that starts and stops a local llama-server."""

    def __init__(self, cfg: TranslationConfig) -> None:
        self.cfg = cfg
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> Self:
        if not self.cfg.server_exe.exists():
            raise TranslationError(
                f"llama-server not found: {self.cfg.server_exe}. "
                f"Set LOCAL_LLM_SERVER_EXE in .env."
            )
        if not self.cfg.server_model.exists():
            raise TranslationError(
                f"Local LLM model not found: {self.cfg.server_model}. "
                f"Set LOCAL_LLM_MODEL in .env."
            )

        # Fail fast if another process already owns the port, otherwise the
        # readiness poll would succeed against the wrong server.
        if _port_is_open(self.cfg.server_port):
            raise TranslationError(
                f"Port {self.cfg.server_port} is already in use; stop that "
                f"process or change LOCAL_LLM_PORT."
            )

        log.info("Starting llama-server on port %d", self.cfg.server_port)
        self.process = subprocess.Popen(
            [
                str(self.cfg.server_exe),
                "-m", str(self.cfg.server_model),
                "--port", str(self.cfg.server_port),
                "--no-jinja",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            _wait_for_port(self.cfg.server_port, _STARTUP_TIMEOUT)
            if self.process.poll() is not None:
                raise TranslationError(
                    f"llama-server exited during startup (code "
                    f"{self.process.returncode}); port {self.cfg.server_port} "
                    f"may be in use."
                )
        except TranslationError:
            self._stop()
            raise

        log.info("llama-server ready")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop()

    def _stop(self) -> None:
        if self.process is None:
            return
        log.info("Stopping llama-server")
        self.process.terminate()
        try:
            self.process.wait(timeout=_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            log.warning(
                "llama-server did not stop in %.0fs, killing", _SHUTDOWN_TIMEOUT
            )
            self.process.kill()
            self.process.wait()
        self.process = None


def _port_is_open(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_is_open(port, timeout=1.0):
            return
        time.sleep(_POLL_INTERVAL)
    raise TranslationError(
        f"llama-server did not start on port {port} within {timeout:.0f}s"
    )

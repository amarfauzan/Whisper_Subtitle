"""Whisper transcription backends: CLI (one-shot) or server (long-running).

The CLI backend spawns `whisper-cli` for every chunk, paying the model-load
cost each time. The server backend starts `whisper-server` once, loads the
model once, and serves every chunk via HTTP. The server mode saves roughly
5-15 seconds per chunk on a large model.

Both backends implement the same interface:
  - `with backend:` (server spawns and kills; CLI is a no-op)
  - `backend.transcribe(audio_path) -> WhisperResult`
"""

import json
import logging
import socket
import subprocess
import time
from pathlib import Path
from typing import Protocol, Self

import requests

from whisper_subtitle.config import WhisperConfig
from whisper_subtitle.exceptions import WhisperError
from whisper_subtitle.models import WhisperSegment
from whisper_subtitle.transcription.whisper_runner import (
    WhisperResult,
    run_whisper_cli,
)

log = logging.getLogger(__name__)


class WhisperBackend(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
    def transcribe(self, audio_path: Path) -> WhisperResult: ...


class CliBackend:
    """One-shot CLI per chunk. No lifecycle, no server."""

    def __init__(self, cfg: WhisperConfig) -> None:
        self.cfg = cfg

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        pass

    def transcribe(self, audio_path: Path) -> WhisperResult:
        return run_whisper_cli(audio_path, self.cfg)


class ServerBackend:
    """Long-running whisper-server. Loads model once, serves all chunks."""

    def __init__(self, cfg: WhisperConfig) -> None:
        self.cfg = cfg
        self.process: subprocess.Popen | None = None
        self.raw_responses: list[dict] = []   # NEW

    def transcribe(self, audio_path: Path) -> WhisperResult:
        return _transcribe_via_server(audio_path, self.cfg, self.raw_responses)

    def __enter__(self) -> Self:
        _ensure_port_free(self.cfg.server_host, self.cfg.server_port)
        self.process = _start_server(self.cfg)
        try:
            _wait_for_port(
                self.cfg.server_host,
                self.cfg.server_port,
                timeout=self.cfg.server_startup_timeout,
            )
            _ensure_server_alive(self.process, self.cfg)
        except Exception:
            self.process.kill()
            self.process.wait()
            raise
        return self

    def __exit__(self, *args) -> None:
        if self.process is None:
            return
        log.info("Stopping whisper server")
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("Server did not stop in 5s, killing")
            self.process.kill()
            self.process.wait()

def make_whisper_backend(cfg: WhisperConfig) -> WhisperBackend:
    """Return the backend selected by cfg.mode."""
    if cfg.mode == "server":
        return ServerBackend(cfg)
    if cfg.mode == "cli":
        return CliBackend(cfg)
    raise WhisperError(
        f"Unknown whisper.mode: {cfg.mode!r} (expected 'cli' or 'server')"
    )


# ----------------------------------------------------------------------
# Server internals
# ----------------------------------------------------------------------


def _start_server(cfg: WhisperConfig) -> subprocess.Popen:
    if not cfg.server_exe.exists():
        raise WhisperError(
            f"whisper-server not found: {cfg.server_exe}. "
            f"Set WHISPER_SERVER in .env or switch whisper.mode to 'cli'."
        )

    command = [
        str(cfg.server_exe),
        "-m", str(cfg.model),
        "-t", str(cfg.server_threads),
        "-bo", str(cfg.best_of),
        "-bs", str(cfg.beam_size),
        "--host", cfg.server_host,
        "--port", str(cfg.server_port),
    ]

    log.info(
        "Starting whisper server on %s:%d",
        cfg.server_host, cfg.server_port,
    )
    log.debug("Server command: %s", " ".join(command))

    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ensure_port_free(host: str, port: int) -> None:
    """Fail fast if something is already listening on the server port.

    Without this check, a stale whisper-server (or any other service)
    on the port makes `_wait_for_port` succeed instantly, and every
    inference request silently hits the wrong listener.
    """
    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError:
        return  # connection refused → port is free
    raise WhisperError(
        f"Port {host}:{port} is already in use by another process. "
        f"Stop that process or change whisper.server_port / "
        f"whisper.ensemble.secondary_port."
    )


def _wait_for_port(host: str, port: int, timeout: float) -> None:
    """Poll until the TCP port accepts connections, or timeout."""
    log.info("Waiting for whisper server at %s:%d", host, port)
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                log.info("Whisper server ready")
                return
        except OSError:
            time.sleep(0.5)
    raise WhisperError(
        f"Whisper server did not start within {timeout}s "
        f"on {host}:{port}"
    )


def _ensure_server_alive(
    process: subprocess.Popen,
    cfg: WhisperConfig,
) -> None:
    """Fail fast if the spawned whisper-server died during startup.

    `_wait_for_port` only proves that *something* accepts connections.
    If another process grabbed the port between `_ensure_port_free` and
    the child's bind, whisper-server exits and every later request
    would hit the foreign listener (typically answered with HTTP 405).
    A child that failed to bind exits almost immediately, so a short
    grace period before declaring readiness catches it.
    """
    time.sleep(0.2)
    if process.poll() is not None:
        raise WhisperError(
            f"whisper-server exited during startup (exit code "
            f"{process.returncode}). Port "
            f"{cfg.server_host}:{cfg.server_port} is probably already "
            f"in use by another process — check it with "
            f"`ss -tlnp | grep {cfg.server_port}` or change "
            f"whisper.server_port / whisper.ensemble.secondary_port."
        )


def _transcribe_via_server(
    audio_path: Path,
    cfg: WhisperConfig,
    raw_sink: list[dict] | None = None,
) -> WhisperResult:
    url = f"http://{cfg.server_host}:{cfg.server_port}/inference"
    log.info("Sending %s to whisper server", audio_path.name)

    with audio_path.open("rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        data = {
            "language": cfg.language,
            "response_format": "verbose_json",
        }
        try:
            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=cfg.server_request_timeout,
            )
        except requests.RequestException as exc:
            raise WhisperError(
                f"Whisper server request failed: {exc}"
            ) from exc

    if response.status_code != 200:
        raise WhisperError(
            f"Whisper server returned {response.status_code}: "
            f"{response.text[:300]}"
        )

    try:
        # Decode explicitly: quantized models can emit a broken multi-byte
        # sequence, and letting requests guess the charset turns those bytes
        # into U+FFFD litter inside the subtitles.
        payload = json.loads(response.content.decode("utf-8", errors="replace"))

        if raw_sink is not None:
            raw_sink.append(payload)

    except ValueError as exc:
        raise WhisperError(
            f"Invalid JSON from whisper server: {exc}"
        ) from exc

    segments = _parse_server_segments(payload)

    # The server does not write a JSON file to disk — we point json_path
    # at what a CLI run would have produced, in case anything downstream
    # wants to know the canonical output location. Nothing reads it
    # currently.
    return WhisperResult(
        segments=segments,
        json_path=audio_path.with_suffix(".json"),
    )


def _parse_server_segments(payload: dict) -> list[WhisperSegment]:
    """Convert whisper-server's verbose_json into WhisperSegment objects.

    The server returns:
        {"segments": [{"start": seconds, "end": seconds, "text": "..."}]}
    """
    raw = payload.get("segments") or []
    segments: list[WhisperSegment] = []
    replacement_chars = 0

    for item in raw:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if "\ufffd" in text:
            # Invalid byte sequence from the model — drop the broken
            # character instead of shipping it into the SRT.
            replacement_chars += text.count("\ufffd")
            text = text.replace("\ufffd", "").strip()
            if not text:
                continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WhisperError(
                f"Malformed segment from whisper server: {item}"
            ) from exc
        segments.append(WhisperSegment(start=start, end=end, text=text))

    if replacement_chars:
        log.warning(
            "Dropped %d undecodable character(s) from the whisper server "
            "response (model emitted invalid UTF-8)",
            replacement_chars,
        )

    return segments

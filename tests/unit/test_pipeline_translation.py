"""Pipeline-level wiring of the translation provider."""

from pathlib import Path
from types import SimpleNamespace

from whisper_subtitle import pipeline
from whisper_subtitle.config import TranslationConfig


def make_translation_cfg(provider: str) -> TranslationConfig:
    return TranslationConfig(
        api_key="test",
        base_url="https://example.invalid",
        model="test-model",
        disable_thinking=True,
        source_language="Korean",
        target_language="English",
        batch_size=100,
        max_retries=3,
        min_split_size=50,
        context="",
        provider=provider,
        request_timeout=180.0,
        server_exe=Path("/fake/llama-server"),
        server_model=Path("/fake/model.gguf"),
        server_port=8080,
    )


class _FakeServer:
    def __init__(self, cfg, events):
        self.cfg = cfg
        self.events = events

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, *exc):
        self.events.append("exit")


def _patch_translator(monkeypatch, tmp_path, output):
    monkeypatch.setattr(
        "whisper_subtitle.translation.translator.translate_subtitles",
        lambda subtitles, cfg, source_srt: output,
    )


def test_local_provider_wraps_translation_in_server(monkeypatch, tmp_path):
    events: list[str] = []
    cfg = make_translation_cfg("local")
    output = tmp_path / "out.srt"
    _patch_translator(monkeypatch, tmp_path, output)
    monkeypatch.setattr(
        "whisper_subtitle.translation.local_server.LlamaServer",
        lambda server_cfg: _FakeServer(server_cfg, events),
    )

    pipeline._run_translation([], tmp_path / "in.srt", SimpleNamespace(translation=cfg))

    assert events == ["enter", "exit"]


def test_deepseek_provider_does_not_start_server(monkeypatch, tmp_path):
    events: list[str] = []
    cfg = make_translation_cfg("deepseek")
    output = tmp_path / "out.srt"
    _patch_translator(monkeypatch, tmp_path, output)
    monkeypatch.setattr(
        "whisper_subtitle.translation.local_server.LlamaServer",
        lambda server_cfg: _FakeServer(server_cfg, events),
    )

    pipeline._run_translation([], tmp_path / "in.srt", SimpleNamespace(translation=cfg))

    assert events == []

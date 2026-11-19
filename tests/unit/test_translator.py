from pathlib import Path
from unittest.mock import MagicMock

import pytest

from whisper_subtitle.config import TranslationConfig
from whisper_subtitle.exceptions import TranslationError
from whisper_subtitle.models import Subtitle
from whisper_subtitle.translation.translator import (
    _build_prompt,
    _make_client,
    _parse_marked_response,
    _translate_batch,
)

# ---------------------------------------------------------------- helpers


def sub(text: str, hint: str | None = None) -> Subtitle:
    return Subtitle(start=0.0, end=1.0, text=text, source_hint=hint)


def make_cfg(**overrides) -> TranslationConfig:
    base = dict(
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
        provider="deepseek",
        request_timeout=180.0,
        server_exe=Path("/fake/llama-server"),
        server_model=Path("/fake/model.gguf"),
        server_port=8080,
    )
    base.update(overrides)
    return TranslationConfig(**base)


def fake_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


# ---------------------------------------------------------------- prompt


class TestBuildPrompt:
    def test_numbers_every_subtitle(self):
        prompt = _build_prompt([sub("one"), sub("two")], make_cfg())
        assert "<<<SRT_001>>>" in prompt
        assert "<<<SRT_002>>>" in prompt
        assert "one" in prompt
        assert "two" in prompt

    def test_single_source_no_hint_markers(self):
        prompt = _build_prompt([sub("hello")], make_cfg())
        assert "[WHISPER]" not in prompt
        assert "[OCR]" not in prompt

    def test_dual_source_includes_both(self):
        prompt = _build_prompt([sub("안녕하세요", hint="안녕하세여")], make_cfg())
        assert "[WHISPER] 안녕하세요" in prompt
        assert "[OCR] 안녕하세여" in prompt

    def test_mixed_batch_only_hinted_entries_dual(self):
        prompt = _build_prompt(
            [sub("first"), sub("second", hint="scond"), sub("third")],
            make_cfg(),
        )
        # The prompt's instruction block contains examples with the same
        # markers. Count only the entries section — everything after the
        # first SRT marker.
        entries_start = prompt.find("<<<SRT_001>>>")
        assert entries_start != -1
        entry_section = prompt[entries_start:]
        assert entry_section.count("[WHISPER]") == 1
        assert entry_section.count("[OCR]") == 1

    def test_includes_source_and_target_language(self):
        prompt = _build_prompt(
            [sub("x")],
            make_cfg(source_language="Korean", target_language="English"),
        )
        assert "Korean" in prompt
        assert "English" in prompt

    def test_includes_context_when_set(self):
        prompt = _build_prompt(
            [sub("x")],
            make_cfg(context="Rescene is a K-pop group"),
        )
        assert "Rescene is a K-pop group" in prompt

    def test_no_context_block_when_empty(self):
        prompt = _build_prompt([sub("x")], make_cfg(context=""))
        assert "Context for the translation" not in prompt


class TestLocalPrompt:
    def test_uses_translategemma_format(self):
        prompt = _build_prompt([sub("안녕하세요")], make_cfg(provider="local"))
        assert "professional Korean (ko) to English (en) translator" in prompt
        assert "Produce only the English translation" in prompt
        assert "Please translate the following Korean text into English:" in prompt
        assert "<<<SRT_001>>>" in prompt
        assert "안녕하세요" in prompt

    def test_local_prompt_omits_deepseek_rules(self):
        prompt = _build_prompt([sub("x")], make_cfg(provider="local"))
        assert "DUAL-SOURCE ENTRIES" not in prompt
        assert "Every <<<SRT_NNN>>> marker represents ONE subtitle" not in prompt

    def test_deepseek_prompt_unchanged(self):
        prompt = _build_prompt([sub("x")], make_cfg())
        assert "Translate the following Korean subtitles" in prompt
        assert "Every <<<SRT_NNN>>> marker represents ONE subtitle" in prompt

    def test_unknown_language_falls_back_to_two_letters(self):
        prompt = _build_prompt(
            [sub("x")], make_cfg(provider="local", source_language="Thai")
        )
        assert "Thai (th)" in prompt


# ---------------------------------------------------------------- parsing


class TestParseMarkedResponse:
    def test_single(self):
        content = "<<<SRT_001>>>\nhello"
        assert _parse_marked_response(content, expected=1) == ["hello"]

    def test_multiple(self):
        content = "<<<SRT_001>>>\nfirst\n\n<<<SRT_002>>>\nsecond"
        assert _parse_marked_response(content, expected=2) == ["first", "second"]

    def test_wrong_count_returns_none(self):
        content = "<<<SRT_001>>>\nfirst"
        assert _parse_marked_response(content, expected=2) is None

    def test_out_of_order_returns_none(self):
        content = "<<<SRT_002>>>\nsecond\n\n<<<SRT_001>>>\nfirst"
        assert _parse_marked_response(content, expected=2) is None

    def test_empty_translation_returns_none(self):
        content = "<<<SRT_001>>>\n\n<<<SRT_002>>>\nsecond"
        assert _parse_marked_response(content, expected=2) is None


# ---------------------------------------------------------------- batch


class TestTranslateBatch:
    def test_happy_path(self):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response(
            "<<<SRT_001>>>\n하나\n\n<<<SRT_002>>>\n둘"
        )
        result = _translate_batch(
            [sub("안녕"), sub("잘가")], make_cfg(), client
        )
        assert result == ["하나", "둘"]

    def test_retries_on_marker_mismatch(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            fake_response("garbage with no markers"),
            fake_response("<<<SRT_001>>>\nok"),
        ]
        result = _translate_batch([sub("x")], make_cfg(), client)
        assert result == ["ok"]
        assert client.chat.completions.create.call_count == 2

    def test_raises_after_exhausting_retries(self):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response("no markers")
        with pytest.raises(TranslationError):
            _translate_batch([sub("x")], make_cfg(max_retries=2), client)
        assert client.chat.completions.create.call_count == 2

    def test_retries_on_api_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("network"),
            fake_response("<<<SRT_001>>>\nok"),
        ]
        result = _translate_batch([sub("x")], make_cfg(), client)
        assert result == ["ok"]


# ---------------------------------------------------------------- hints


class TestTranslateSubtitlesWithHints:
    def test_hints_reach_prompt(self):
        """Regression test: source_hint must appear in the prompt sent to the LLM."""
        client = MagicMock()
        captured = {}

        def fake_create(**kwargs):
            captured["prompt"] = kwargs["messages"][0]["content"]
            return fake_response("<<<SRT_001>>>\ntranslated")

        client.chat.completions.create.side_effect = fake_create

        subs = [sub("원문", hint="원믄")]
        _translate_batch(subs, make_cfg(), client)

        assert "[WHISPER] 원문" in captured["prompt"]
        assert "[OCR] 원믄" in captured["prompt"]


# ---------------------------------------------------------------- providers


class TestTranslateBatchProviders:
    def test_local_uses_low_temperature_and_no_thinking(self):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response(
            "<<<SRT_001>>>\nok"
        )
        _translate_batch([sub("x")], make_cfg(provider="local"), client)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.1
        assert "extra_body" not in kwargs

    def test_deepseek_uses_default_temperature_and_thinking_off(self):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response(
            "<<<SRT_001>>>\nok"
        )
        _translate_batch([sub("x")], make_cfg(), client)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.3
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_deepseek_respects_disable_thinking_false(self):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response(
            "<<<SRT_001>>>\nok"
        )
        _translate_batch([sub("x")], make_cfg(disable_thinking=False), client)
        assert "extra_body" not in client.chat.completions.create.call_args.kwargs

    def test_timeout_comes_from_config(self):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response(
            "<<<SRT_001>>>\nok"
        )
        _translate_batch([sub("x")], make_cfg(request_timeout=42.0), client)
        assert client.chat.completions.create.call_args.kwargs["timeout"] == 42.0


class TestMakeClient:
    def test_local_points_at_server_port(self):
        client = _make_client(make_cfg(provider="local", api_key=""))
        assert client.base_url.host == "127.0.0.1"
        assert client.base_url.port == 8080

    def test_deepseek_requires_api_key(self):
        with pytest.raises(TranslationError, match="DEEPSEEK_API_KEY"):
            _make_client(make_cfg(api_key=""))

    def test_deepseek_uses_configured_base_url(self):
        client = _make_client(make_cfg())
        assert client.base_url.host == "example.invalid"
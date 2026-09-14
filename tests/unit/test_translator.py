from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from whisper_subtitle.config import TranslationConfig
from whisper_subtitle.exceptions import TranslationError
from whisper_subtitle.translation.translator import (
    _build_prompt,
    _parse_marked_response,
    _translate_batch,
)


def make_cfg(**overrides) -> TranslationConfig:
    base = dict(
        api_key="test",
        base_url="https://example.invalid",
        model="test-model",
        source_language="Korean",
        target_language="English",
        batch_size=100,
        max_retries=3,
        min_split_size=50,
        context="",
    )
    base.update(overrides)
    return TranslationConfig(**base)


def fake_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestBuildPrompt:
    def test_numbers_every_subtitle(self):
        prompt = _build_prompt(["one", "two"], make_cfg())
        assert "<<<SRT_001>>>" in prompt
        assert "<<<SRT_002>>>" in prompt
        assert "one" in prompt
        assert "two" in prompt

    def test_includes_source_and_target_language(self):
        prompt = _build_prompt(
            ["x"], make_cfg(source_language="Korean", target_language="English")
        )
        assert "Korean" in prompt
        assert "English" in prompt

    def test_includes_context_when_set(self):
        prompt = _build_prompt(["x"], make_cfg(context="Rescene is a K-pop group"))
        assert "Rescene is a K-pop group" in prompt

    def test_no_context_block_when_empty(self):
        prompt = _build_prompt(["x"], make_cfg(context=""))
        assert "Context for the translation" not in prompt


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


class TestTranslateBatch:
    def test_happy_path(self):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response(
            "<<<SRT_001>>>\n하나\n\n<<<SRT_002>>>\n둘"
        )
        result = _translate_batch(["안녕", "잘가"], make_cfg(), client)
        assert result == ["하나", "둘"]

    def test_retries_on_marker_mismatch(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            fake_response("garbage with no markers"),
            fake_response("<<<SRT_001>>>\nok"),
        ]
        result = _translate_batch(["x"], make_cfg(), client)
        assert result == ["ok"]
        assert client.chat.completions.create.call_count == 2

    def test_raises_after_exhausting_retries(self):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response("no markers")
        with pytest.raises(TranslationError):
            _translate_batch(["x"], make_cfg(max_retries=2), client)
        assert client.chat.completions.create.call_count == 2

    def test_retries_on_api_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("network"),
            fake_response("<<<SRT_001>>>\nok"),
        ]
        result = _translate_batch(["x"], make_cfg(), client)
        assert result == ["ok"]

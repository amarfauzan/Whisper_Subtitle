"""Tests for the LLM-based OCR filter."""

from unittest.mock import MagicMock

import pytest

from whisper_subtitle.config import LlmFilterConfig
from whisper_subtitle.models import OcrEvent
from whisper_subtitle.ocr.llm_filter import (
    _build_prompt,
    _parse_response,
    filter_events_with_llm,
)


def ev(text: str) -> OcrEvent:
    return OcrEvent(
        start=0.0, end=1.0, box=(0, 0, 100, 50),
        text=text, frame_count=3, confidence=0.9,
    )


def cfg(**overrides) -> LlmFilterConfig:
    base = dict(
        enabled=True,
        model="test-model",
        batch_size=100,
        min_events=1,
    )
    base.update(overrides)
    return LlmFilterConfig(**base)


def fake_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


# ---------------------------------------------------------------- prompt


class TestBuildPrompt:
    def test_numbers_entries(self):
        prompt = _build_prompt([ev("안녕"), ev("ZENA")])
        assert "[0] 안녕" in prompt
        assert "[1] ZENA" in prompt

    def test_includes_drop_instruction(self):
        prompt = _build_prompt([ev("x")])
        assert '"drop"' in prompt


# ---------------------------------------------------------------- parsing


class TestParseResponse:
    def test_valid_response(self):
        result = _parse_response('{"drop": [0, 2]}', max_index=5)
        assert result == {0, 2}

    def test_empty_drop_list(self):
        assert _parse_response('{"drop": []}', max_index=5) == set()

    def test_out_of_range_ignored(self):
        result = _parse_response('{"drop": [0, 99, -1]}', max_index=3)
        assert result == {0}

    def test_non_integer_ignored(self):
        result = _parse_response('{"drop": [0, "1", 2.5, 1]}', max_index=5)
        assert result == {0, 1}

    def test_booleans_ignored(self):
        # In Python, True == 1. We must not treat it as index 1.
        result = _parse_response('{"drop": [true, false]}', max_index=5)
        assert result == set()

    def test_malformed_json_returns_empty(self):
        assert _parse_response("not json", max_index=5) == set()

    def test_wrong_shape_returns_empty(self):
        assert _parse_response("[1, 2]", max_index=5) == set()

    def test_missing_drop_key_returns_empty(self):
        assert _parse_response('{"keep": [0]}', max_index=5) == set()

    def test_markdown_fenced_json(self):
        content = '```json\n{"drop": [0]}\n```'
        assert _parse_response(content, max_index=5) == {0}


# ---------------------------------------------------------------- filtering


class TestFilterEventsWithLlm:
    def test_empty_input(self):
        client = MagicMock()
        result = filter_events_with_llm([], cfg(), "key", "url")
        assert result == []
        client.chat.completions.create.assert_not_called()

    def test_below_min_events_skips_api(self, monkeypatch):
        client = MagicMock()
        monkeypatch.setattr(
            "whisper_subtitle.ocr.llm_filter.OpenAI", lambda **kw: client
        )
        events = [ev("a"), ev("b")]
        result = filter_events_with_llm(
            events, cfg(min_events=5), "key", "url"
        )
        assert result == events
        client.chat.completions.create.assert_not_called()

    def test_drops_specified_indices(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response(
            '{"drop": [1, 3]}'
        )
        monkeypatch.setattr(
            "whisper_subtitle.ocr.llm_filter.OpenAI", lambda **kw: client
        )
        events = [ev("keep1"), ev("drop1"), ev("keep2"), ev("drop2")]
        result = filter_events_with_llm(events, cfg(), "key", "url")
        assert [e.text for e in result] == ["keep1", "keep2"]

    def test_invalid_json_keeps_all(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response("not json")
        monkeypatch.setattr(
            "whisper_subtitle.ocr.llm_filter.OpenAI", lambda **kw: client
        )
        events = [ev("a"), ev("b")]
        result = filter_events_with_llm(events, cfg(), "key", "url")
        assert result == events

    def test_api_error_keeps_all(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network")
        monkeypatch.setattr(
            "whisper_subtitle.ocr.llm_filter.OpenAI", lambda **kw: client
        )
        events = [ev("a"), ev("b")]
        result = filter_events_with_llm(events, cfg(), "key", "url")
        assert result == events

    def test_multiple_batches_shift_indices(self, monkeypatch):
        client = MagicMock()
        # Two batches of 3. First batch drops index 1, second drops index 0.
        client.chat.completions.create.side_effect = [
            fake_response('{"drop": [1]}'),
            fake_response('{"drop": [0]}'),
        ]
        monkeypatch.setattr(
            "whisper_subtitle.ocr.llm_filter.OpenAI", lambda **kw: client
        )
        events = [ev(f"e{i}") for i in range(6)]
        result = filter_events_with_llm(
            events, cfg(batch_size=3), "key", "url"
        )
        # Dropped: global index 1 (e1) and global index 3 (e3)
        assert [e.text for e in result] == ["e0", "e2", "e4", "e5"]

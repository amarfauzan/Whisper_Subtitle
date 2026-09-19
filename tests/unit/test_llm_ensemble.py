"""Tests for LLM-based ensemble adjudication."""

from unittest.mock import MagicMock

import pytest

from whisper_subtitle.config import EnsembleConfig
from whisper_subtitle.models import WhisperSegment
from whisper_subtitle.transcription.ensemble import AdjudicationRequest
from whisper_subtitle.transcription.llm_ensemble import (
    _build_prompt,
    _parse_adjudication,
    adjudicate_requests,
)


def cfg(**overrides) -> EnsembleConfig:
    from pathlib import Path
    base = dict(
        enabled=True,
        secondary_model=Path("/fake/v3.bin"),
        secondary_port=8081,
        overlap_threshold=0.4,
        agree_threshold=0.85,
        hint_threshold=0.5,
        min_cue_duration=0.5,
        min_cue_chars=2,
        llm_adjudicate=True,
        llm_batch_size=50,
    )
    base.update(overrides)
    return EnsembleConfig(**base)


def req(index: int, v3: str, bat: str, before="", after="") -> AdjudicationRequest:
    return AdjudicationRequest(
        index=index, v3_text=v3, bat_text=bat,
        context_before=before, context_after=after,
    )


def cue(text: str) -> WhisperSegment:
    return WhisperSegment(start=0.0, end=1.0, text=text)


def fake_response(content: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


# ---------------------------------------------------------------- prompt


class TestBuildPrompt:
    def test_numbers_entries(self):
        prompt = _build_prompt([
            req(0, "안녕", "안녕하세요"),
            req(1, "잘가", "잘가요"),
        ])
        assert "<<<SEG_001>>>" in prompt
        assert "<<<SEG_002>>>" in prompt

    def test_includes_both_readings(self):
        prompt = _build_prompt([req(0, "V3 text", "BAT text")])
        assert "V3:  V3 text" in prompt
        assert "BAT: BAT text" in prompt

    def test_includes_context_when_present(self):
        prompt = _build_prompt([req(0, "a", "b", before="context before")])
        assert "BEFORE: context before" in prompt

    def test_no_context_marker_when_empty(self):
        prompt = _build_prompt([req(0, "a", "b")])
        assert "(no context)" in prompt


# ---------------------------------------------------------------- parsing


class TestParseAdjudication:
    def test_single(self):
        content = "<<<SEG_001>>>\n안녕하세요"
        result = _parse_adjudication(content, [req(5, "a", "b")])
        assert result == {5: "안녕하세요"}

    def test_multiple(self):
        content = "<<<SEG_001>>>\nfirst\n\n<<<SEG_002>>>\nsecond"
        requests = [req(3, "a", "b"), req(7, "c", "d")]
        result = _parse_adjudication(content, requests)
        assert result == {3: "first", 7: "second"}

    def test_wrong_count_returns_empty(self):
        content = "<<<SEG_001>>>\nfirst"
        requests = [req(0, "a", "b"), req(1, "c", "d")]
        assert _parse_adjudication(content, requests) == {}

    def test_out_of_order_returns_empty(self):
        content = "<<<SEG_002>>>\nsecond\n\n<<<SEG_001>>>\nfirst"
        requests = [req(0, "a", "b"), req(1, "c", "d")]
        assert _parse_adjudication(content, requests) == {}

    def test_empty_text_skipped(self):
        content = "<<<SEG_001>>>\n\n\n<<<SEG_002>>>\nsecond"
        requests = [req(0, "a", "b"), req(1, "c", "d")]
        result = _parse_adjudication(content, requests)
        assert result == {1: "second"}

    def test_markdown_fence_stripped(self):
        content = "```\n<<<SEG_001>>>\nhello\n```"
        result = _parse_adjudication(content, [req(0, "a", "b")])
        assert result == {0: "hello"}


# ---------------------------------------------------------------- main


class TestAdjudicateRequests:
    def test_empty_requests_returns_cues_unchanged(self):
        cues = [cue("original")]
        result = adjudicate_requests([], cues, cfg(), "key", "url")
        assert result == cues

    def test_no_api_call_when_empty(self, monkeypatch):
        client = MagicMock()
        monkeypatch.setattr(
            "whisper_subtitle.transcription.llm_ensemble.OpenAI",
            lambda **kw: client,
        )
        adjudicate_requests([], [cue("x")], cfg(), "key", "url")
        client.chat.completions.create.assert_not_called()

    def test_successful_replacement(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response(
            "<<<SEG_001>>>\n새 텍스트"
        )
        monkeypatch.setattr(
            "whisper_subtitle.transcription.llm_ensemble.OpenAI",
            lambda **kw: client,
        )
        cues = [cue("old text")]
        requests = [req(0, "old text", "alternate")]
        result = adjudicate_requests(requests, cues, cfg(), "key", "url")
        assert result[0].text == "새 텍스트"

    def test_api_error_keeps_original(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network")
        monkeypatch.setattr(
            "whisper_subtitle.transcription.llm_ensemble.OpenAI",
            lambda **kw: client,
        )
        cues = [cue("original")]
        requests = [req(0, "original", "alternate")]
        result = adjudicate_requests(requests, cues, cfg(), "key", "url")
        assert result[0].text == "original"

    def test_marker_mismatch_keeps_original(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response("no markers")
        monkeypatch.setattr(
            "whisper_subtitle.transcription.llm_ensemble.OpenAI",
            lambda **kw: client,
        )
        cues = [cue("original")]
        requests = [req(0, "original", "alternate")]
        result = adjudicate_requests(requests, cues, cfg(), "key", "url")
        assert result[0].text == "original"

    def test_only_marked_indices_replaced(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response(
            "<<<SEG_001>>>\nnew"
        )
        monkeypatch.setattr(
            "whisper_subtitle.transcription.llm_ensemble.OpenAI",
            lambda **kw: client,
        )
        cues = [cue("first"), cue("second"), cue("third")]
        requests = [req(1, "second", "alt")]
        result = adjudicate_requests(requests, cues, cfg(), "key", "url")
        assert result[0].text == "first"     # untouched
        assert result[1].text == "new"       # replaced
        assert result[2].text == "third"     # untouched
"""LLM adjudication for ensemble ASR disagreements.

When batisay and large-v3 produce significantly different readings of the
same audio, neither is reliably correct. We send both readings plus
surrounding context to DeepSeek and ask it to pick the right one (or
provide a corrected version).

Runs as a separate step after merge_ensemble(). On any API failure, the
original V3 text is kept — the pipeline never breaks because of an LLM
hiccup.
"""

import json
import logging
import re
from dataclasses import replace

from openai import OpenAI

from whisper_subtitle.config import EnsembleConfig
from whisper_subtitle.models import WhisperSegment
from whisper_subtitle.transcription.ensemble import AdjudicationRequest

log = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """\
Two Korean ASR models transcribed the same audio segment. They disagree.
Pick the reading that is more accurate, or if both are wrong, provide
the correct Korean text. Use the surrounding context to decide.

RULES:
1. Return exactly one response per <<<SEG_NNN>>> marker.
2. Every marker MUST appear exactly once, in order.
3. Each response is the chosen (or corrected) Korean text for that segment.
4. Keep every marker EXACTLY unchanged.
5. Do not add explanations, comments, or markdown.
6. Do not merge or split segments.
7. Do not translate — output Korean only.

When both readings are plausible, prefer the one that reads most naturally
in Korean and matches the surrounding context.

{entries}
"""


def adjudicate_requests(
    requests: list[AdjudicationRequest],
    cues: list[WhisperSegment],
    cfg: EnsembleConfig,
    api_key: str,
    base_url: str,
) -> list[WhisperSegment]:
    """Resolve pending adjudication requests via DeepSeek.

    Args:
        requests: disagreements from merge_ensemble()
        cues: current cue list (V3 text still in place for pending entries)
        cfg: ensemble configuration (batch size, model)
        api_key: DeepSeek API key
        base_url: DeepSeek API base URL

    Returns:
        Updated cue list. Requests that fail to resolve keep their V3 text.
    """
    if not requests:
        return cues

    log.info("Adjudicating %d disagreements via LLM", len(requests))

    client = OpenAI(api_key=api_key, base_url=base_url)

    resolved_texts: dict[int, str] = {}
    total_batches = (len(requests) + cfg.llm_batch_size - 1) // cfg.llm_batch_size

    for batch_idx, start in enumerate(
        range(0, len(requests), cfg.llm_batch_size), start=1
    ):
        batch = requests[start:start + cfg.llm_batch_size]
        log.info(
            "Adjudication batch %d/%d (%d requests)",
            batch_idx, total_batches, len(batch),
        )
        batch_result = _adjudicate_batch(batch, client, cfg)
        resolved_texts.update(batch_result)

    # Apply resolutions; leave unresolved requests untouched
    result: list[WhisperSegment] = []
    for i, cue in enumerate(cues):
        if i in resolved_texts:
            new_text = resolved_texts[i]
            if new_text and new_text != cue.text:
                cue = replace(cue, text=new_text)
        result.append(cue)

    resolved_count = len(resolved_texts)
    log.info(
        "Adjudication complete: %d/%d resolved, %d kept V3 text",
        resolved_count, len(requests), len(requests) - resolved_count,
    )
    return result


def _adjudicate_batch(
    requests: list[AdjudicationRequest],
    client: OpenAI,
    cfg: EnsembleConfig,
) -> dict[int, str]:
    """Send one batch. Returns {cue_index: resolved_text}.

    Empty dict on any failure — the caller keeps V3 text for those entries.
    """
    prompt = _build_prompt(requests)

    try:
        kwargs = dict(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=180,
        )
        # Disable thinking for speed (same as translation)
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — API errors vary
        log.warning("Adjudication API error: %s", exc)
        return {}

    content = (response.choices[0].message.content or "").strip()
    return _parse_adjudication(content, requests)


def _build_prompt(requests: list[AdjudicationRequest]) -> str:
    entries = []
    for i, req in enumerate(requests, start=1):
        context_parts = []
        if req.context_before:
            context_parts.append(f"BEFORE: {req.context_before}")
        if req.context_after:
            context_parts.append(f"AFTER: {req.context_after}")
        context = "\n".join(context_parts) if context_parts else "(no context)"

        entries.append(
            f"<<<SEG_{i:03d}>>>\n"
            f"{context}\n"
            f"V3:  {req.v3_text}\n"
            f"BAT: {req.bat_text}"
        )
    return _PROMPT_TEMPLATE.format(entries="\n\n".join(entries))


_MARKER_RE = re.compile(r"<<<SEG_(\d+)>>>")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _parse_adjudication(
    content: str,
    requests: list[AdjudicationRequest],
) -> dict[int, str]:
    """Parse marked response into {cue_index: text}."""
    content = _FENCE_RE.sub("", content).strip()

    matches = list(_MARKER_RE.finditer(content))
    if len(matches) != len(requests):
        log.warning(
            "Adjudication marker count mismatch: expected %d, got %d",
            len(requests), len(matches),
        )
        return {}

    result: dict[int, str] = {}
    for i, match in enumerate(matches):
        marker_num = int(match.group(1))
        if marker_num != i + 1:
            log.warning("Adjudication markers out of order at %d", marker_num)
            return {}

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        if not text:
            continue

        cue_index = requests[i].index
        result[cue_index] = text

    return result
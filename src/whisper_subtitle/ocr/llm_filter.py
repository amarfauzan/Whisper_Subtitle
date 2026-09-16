"""LLM-based filter for OCR events.

Uses DeepSeek to classify events that pass the deterministic filters
but still look like on-screen clutter: name fragments, watermark
variants, OCR garbage on Korean text. These are exactly what patterns
and repeat filters cannot catch.

Default is KEEP. The LLM only lists indices to drop. On any failure
(JSON parse error, out-of-range indices, API exception, timeout), the
input is returned unchanged.

The prompt is deliberately free of specific names, shows, or brands.
It teaches the categories ("a person's name alone", "a score display")
rather than examples, so it generalizes across videos.
"""

import json
import logging
import re

from openai import OpenAI

from whisper_subtitle.config import LlmFilterConfig
from whisper_subtitle.models import OcrEvent

log = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """\
You are filtering OCR-detected text from a {source_language} {content_type}.
Each entry below is text that appeared on screen. Your job: identify entries
that are NOT real subtitle dialogue so they can be removed.

DROP an entry if it is:
- A person's name alone (a short name with no other words)
- A show, game, or brand name (a recurring title or watermark phrase)
- A score, number, or numeric symbol display
- Short English text or gibberish (all-latin strings that don't read as
  real words)
- Effect text (repeated laughter/crying characters, punctuation only)
- OCR garbage: characters that don't form a real word in {source_language}
- Fragments of a watermark (partial or corrupted logo text)

KEEP an entry if it is:
- Dialogue, narration, or commentary, even short
- Bracketed on-screen descriptions of action or context
- Any sentence with a verb or meaningful phrase that a viewer
  would want to understand

When unsure, KEEP. A false drop is worse than a false keep.

Return ONLY a JSON object in this exact shape:
{{"drop": [list of zero-based indices to remove]}}

If nothing should be dropped, return {{"drop": []}}.
Do not include explanations, markdown, or any other text.

Entries:
{entries}
"""


def filter_events_with_llm(
    events: list[OcrEvent],
    cfg: LlmFilterConfig,
    api_key: str,
    base_url: str,
) -> list[OcrEvent]:
    """Drop events the LLM classifies as on-screen clutter.

    Args:
        events: OCR events (already passed deterministic filters).
        cfg: LLM filter configuration.
        api_key: DeepSeek API key.
        base_url: DeepSeek API base URL.

    Returns:
        The subset of events to keep, in original order.
    """
    if not events:
        return []
    if len(events) < cfg.min_events:
        log.info("LLM filter skipped: only %d events", len(events))
        return list(events)

    client = OpenAI(api_key=api_key, base_url=base_url)

    drop_indices: set[int] = set()
    total_batches = (len(events) + cfg.batch_size - 1) // cfg.batch_size

    for batch_idx, start in enumerate(
        range(0, len(events), cfg.batch_size), start=1
    ):
        batch = events[start:start + cfg.batch_size]
        log.info(
            "LLM filter batch %d/%d (%d events)",
            batch_idx, total_batches, len(batch),
        )
        local_drops = _classify_batch(batch, client, cfg)
        for local_idx in local_drops:
            global_idx = start + local_idx
            drop_indices.add(global_idx)
            log.debug("LLM dropped: %r", events[global_idx].text)

    kept = [ev for i, ev in enumerate(events) if i not in drop_indices]

    dropped = len(events) - len(kept)
    if dropped:
        log.info("LLM filter: dropped %d/%d events", dropped, len(events))
    else:
        log.info("LLM filter: kept all %d events", len(events))

    return kept


def _classify_batch(
    events: list[OcrEvent],
    client: OpenAI,
    cfg: LlmFilterConfig,
) -> set[int]:
    """Ask the LLM which indices in this batch to drop.

    Returns an empty set on any failure — the caller keeps everything.
    """
    prompt = _build_prompt(
        events,
        source_language=cfg.source_language,
        content_type=cfg.content_type,
    )
    try:
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=120,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception as exc:  # noqa: BLE001 — network/API errors vary
        log.warning("LLM filter API error: %s", exc)
        return set()

    content = (response.choices[0].message.content or "").strip()
    return _parse_response(content, max_index=len(events))


def _build_prompt(
    events: list[OcrEvent],
    source_language: str,
    content_type: str,
) -> str:
    lines = [f"[{i}] {ev.text}" for i, ev in enumerate(events)]
    return _PROMPT_TEMPLATE.format(
        entries="\n".join(lines),
        source_language=source_language,
        content_type=content_type,
    )


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _parse_response(content: str, max_index: int) -> set[int]:
    """Extract valid drop indices from the LLM response.

    Returns an empty set if the response is malformed in any way.
    Out-of-range or non-integer indices are silently ignored.
    """
    content = _FENCE_RE.sub("", content).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("LLM filter returned invalid JSON, keeping everything")
        return set()

    if not isinstance(data, dict):
        log.warning("LLM filter response is not an object, keeping everything")
        return set()

    raw_drop = data.get("drop")
    if not isinstance(raw_drop, list):
        log.warning("LLM filter response missing 'drop' list, keeping everything")
        return set()

    valid: set[int] = set()
    for idx in raw_drop:
        if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < max_index:
            valid.add(idx)

    return valid
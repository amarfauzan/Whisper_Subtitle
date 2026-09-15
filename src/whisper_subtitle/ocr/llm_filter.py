"""LLM-based filter for OCR events.

Uses DeepSeek to classify events that pass the deterministic filters
but still look like on-screen clutter: name fragments, watermark
variants, OCR garbage on Korean text. These are exactly what patterns
and repeat filters cannot catch.

Default is KEEP. The LLM only lists indices to drop. On any failure
(JSON parse error, out-of-range indices, API exception, timeout), the
input is returned unchanged.
"""

import json
import logging
import re

from openai import OpenAI

from whisper_subtitle.config import LlmFilterConfig
from whisper_subtitle.exceptions import TranslationError
from whisper_subtitle.models import OcrEvent

log = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """\
You are filtering OCR-detected text from a Korean variety show. Each entry
below is text that appeared on screen. Your job: identify entries that are
NOT real subtitle dialogue so they can be removed.

DROP an entry if it is:
- A person's name alone (e.g. "ZENA", "Minami", "제나", "미나")
- A show, game, or brand name (e.g. "Sports Day", "Playin' Pinball", "Rescene")
- A score or number display (e.g. "9:10", "10:10", "1:2", "14")
- Short English text or gibberish (e.g. "MCNAMS", "TZENA", "ABC", "MINAMIhMAYIOS")
- Effect text (e.g. "ㅋㅋㅋㅋㅋ", "!!!", "(당황)", "두근두근")
- OCR garbage that is not a real Korean word (e.g. "리센", "체나", "춘결승")
- Watermark fragments (e.g. "playin'pinb-all", "W01ZENA", "MAYh15")

KEEP an entry if it is:
- Korean dialogue, narration, or commentary, even short (e.g. "야!", "뭐야?", "진짜?")
- Bracketed on-screen descriptions of action (e.g. "[남은시간8초]", "(도망=3)")
- Any text that a viewer would want to understand

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
        local_drops = _classify_batch(batch, client, cfg.model)
        for local_idx in local_drops:
            global_idx = start + local_idx
            drop_indices.add(global_idx)
            log.debug("LLM dropped: %r", events[global_idx].text)

    kept = [ev for i, ev in enumerate(events) if i not in drop_indices]

    dropped = len(events) - len(kept)
    if dropped:
        log.info(
            "LLM filter: dropped %d/%d events",
            dropped, len(events),
        )
    else:
        log.info("LLM filter: kept all %d events", len(events))

    return kept


def _classify_batch(
    events: list[OcrEvent],
    client: OpenAI,
    model: str,
) -> set[int]:
    """Ask the LLM which indices in this batch to drop.

    Returns an empty set on any failure — the caller keeps everything.
    """
    prompt = _build_prompt(events)
    try:
        response = client.chat.completions.create(
            model=model,
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


def _build_prompt(events: list[OcrEvent]) -> str:
    lines = [f"[{i}] {ev.text}" for i, ev in enumerate(events)]
    return _PROMPT_TEMPLATE.format(entries="\n".join(lines))


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

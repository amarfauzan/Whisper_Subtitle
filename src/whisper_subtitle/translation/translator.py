"""Translate SRT subtitles using DeepSeek's OpenAI-compatible API."""

import logging
import re
from pathlib import Path

from openai import OpenAI

from whisper_subtitle.config import TranslationConfig
from whisper_subtitle.exceptions import TranslationError
from whisper_subtitle.models import Subtitle
from whisper_subtitle.subtitles.srt import parse_srt, render_srt

log = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"<<<SRT_(\d+)>>>")


def translate_srt_file(srt_path: Path, cfg: TranslationConfig) -> Path:
    """Translate a standalone SRT file. No source hints available."""
    if not cfg.api_key:
        raise TranslationError(
            "DEEPSEEK_API_KEY is not set. Add it to .env or skip --translate."
        )

    source_text = srt_path.read_text(encoding="utf-8-sig")
    subtitles = parse_srt(source_text)
    if not subtitles:
        raise TranslationError(f"No subtitles found in {srt_path}")

    return _translate_and_write(subtitles, cfg, srt_path)


def _translate_and_write(
    subtitles: list[Subtitle],
    cfg: TranslationConfig,
    source_srt: Path,
) -> Path:
    """Shared translation logic. Reads Subtitle list, writes SRT."""
    log.info(
        "Translating %d subtitles via %s (hints: %d)",
        len(subtitles), cfg.model,
        sum(1 for s in subtitles if s.source_hint),
    )

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    translated_texts = _translate_all(subtitles, cfg, client)   # ← this

    translated_subtitles = [
        Subtitle(start=s.start, end=s.end, text=t)
        for s, t in zip(subtitles, translated_texts)
    ]

    output_path = source_srt.with_name(f"{source_srt.stem}_translated.srt")
    output_path.write_text(render_srt(translated_subtitles), encoding="utf-8")
    log.info("Wrote %s", output_path.name)
    return output_path


def _translate_all(
    subtitles: list[Subtitle],
    cfg: TranslationConfig,
    client: OpenAI,
) -> list[str]:
    total = len(subtitles)
    batch_size = cfg.batch_size
    total_batches = (total + batch_size - 1) // batch_size

    all_translated: list[str] = []
    for batch_index, start in enumerate(range(0, total, batch_size), start=1):
        batch = subtitles[start:start + batch_size]
        end = min(start + batch_size, total)
        log.info(
            "Batch %d/%d (subtitles %d-%d)",
            batch_index, total_batches, start + 1, end,
        )
        translated = _translate_with_fallback(batch, cfg, client)
        all_translated.extend(translated)

    if len(all_translated) != total:
        raise TranslationError(
            f"Count mismatch: expected {total}, got {len(all_translated)}"
        )
    return all_translated

def _translate_with_fallback(
    subtitles: list[Subtitle],
    cfg: TranslationConfig,
    client: OpenAI,
) -> list[str]:
    try:
        return _translate_batch(subtitles, cfg, client)
    except TranslationError:
        if len(subtitles) <= cfg.min_split_size:
            raise
        midpoint = len(subtitles) // 2
        log.warning(
            "Splitting failed batch of %d into %d + %d",
            len(subtitles), midpoint, len(subtitles) - midpoint,
        )
        left = _translate_with_fallback(subtitles[:midpoint], cfg, client)
        right = _translate_with_fallback(subtitles[midpoint:], cfg, client)
        return left + right


def _translate_batch(
    subtitles: list[Subtitle],
    cfg: TranslationConfig,
    client: OpenAI,
) -> list[str]:
    """Send one batch. Retries on marker mismatch or API error."""
    prompt = _build_prompt(subtitles, cfg)

    for attempt in range(1, cfg.max_retries + 1):
        try:
            log.debug("API call (attempt %d/%d)", attempt, cfg.max_retries)
            kwargs = dict(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=180,
            )
            if cfg.disable_thinking:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = client.chat.completions.create(**kwargs)
            content = (response.choices[0].message.content or "").strip()
            parsed = _parse_marked_response(content, expected=len(subtitles))
            if parsed is not None:
                return parsed
            log.warning(
                "Marker mismatch (attempt %d/%d)", attempt, cfg.max_retries
            )
        except Exception as exc:  # noqa: BLE001 — API errors are varied
            log.warning("API error (attempt %d/%d): %s",
                        attempt, cfg.max_retries, exc)

    raise TranslationError(
        f"Batch of {len(subtitles)} failed after {cfg.max_retries} attempts"
    )

def translate_subtitles(
    subtitles: list[Subtitle],
    cfg: TranslationConfig,
    source_srt: Path,
) -> Path:
    """Translate a list of Subtitle objects in memory.

    Uses source_hint when present to give the LLM two sources of truth.
    Writes <source_srt stem>_translated.srt next to the source file.
    """
    if not cfg.api_key:
        raise TranslationError("DEEPSEEK_API_KEY is not set.")
    if not subtitles:
        raise TranslationError("No subtitles to translate.")

    log.info(
        "Translating %d subtitles via %s (hints: %d)",
        len(subtitles), cfg.model,
        sum(1 for s in subtitles if s.source_hint),
    )

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    translated_texts = _translate_all(subtitles, cfg, client)

    translated_subtitles = [
        Subtitle(start=s.start, end=s.end, text=t)
        for s, t in zip(subtitles, translated_texts)
    ]
    output_path = source_srt.with_name(f"{source_srt.stem}_translated.srt")
    output_path.write_text(render_srt(translated_subtitles), encoding="utf-8")
    log.info("Wrote %s", output_path.name)
    return output_path

def _build_prompt(subtitles: list[Subtitle], cfg: TranslationConfig) -> str:
    """Build the marker-based translation prompt.

    Entries with `source_hint` are shown to the LLM as two readings
    of the same speech: [WHISPER] and [OCR]. Everything else is a
    single-source entry. The DUAL-SOURCE instructions are only added
    when at least one entry has a hint.
    """
    segments = []
    has_hints = False
    for i, sub in enumerate(subtitles, start=1):
        if sub.source_hint:
            body = f"[WHISPER] {sub.text}\n[OCR] {sub.source_hint}"
            has_hints = True
        else:
            body = sub.text
        segments.append(f"<<<SRT_{i:03d}>>>\n{body}")
    marked = "\n\n".join(segments)

    context_block = ""
    if cfg.context:
        context_block = (
            f"Context for the translation:\n{cfg.context}\n\n"
            "Use this context only to improve translation quality. "
            "Do not translate or modify the context.\n"
        )

    dual_source_block = ""
    if has_hints:
        dual_source_block = f"""
DUAL-SOURCE ENTRIES:
Some entries have two lines, labeled WHISPER and OCR. These are two
readings of the same speech from different systems. Pick whichever
reads as more natural and correct in {cfg.source_language}, then
translate that one. Do not mix the two readings into one translation.
"""

    return f"""\
Translate the following {cfg.source_language} subtitles into natural {cfg.target_language}.

{context_block}
RULES:
1. Every <<<SRT_NNN>>> marker represents ONE subtitle.
2. Each marker must have exactly ONE translation.
3. Never merge two subtitles, even if they form one sentence.
4. Never split one subtitle into multiple subtitles.
5. Keep every marker exactly unchanged.
6. Do not add, remove, duplicate, or reorder markers.
7. Use neighboring subtitles as context for incomplete sentences.
8. Return exactly the same number of subtitle segments as the input.
9. Return ONLY the translated subtitles, no explanations or markdown.
{dual_source_block}
{marked}
"""


def _parse_marked_response(content: str, expected: int) -> list[str] | None:
    """Extract marked translations. Returns None if markers don't match."""
    matches = list(_MARKER_RE.finditer(content))
    if len(matches) != expected:
        return None

    translations: list[str] = []
    for i, match in enumerate(matches):
        if int(match.group(1)) != i + 1:
            return None
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        if not text:
            return None
        translations.append(text)
    return translations

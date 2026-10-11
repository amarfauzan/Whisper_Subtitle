import os
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

from whisper_subtitle.exceptions import PipelineError


@dataclass(frozen=True)
class VadSegmentsConfig:
    padding: float
    merge_gap: float
    min_segment_duration: float
    max_batch_duration: float    # 0 disables batching
    silence_ms: int
    min_coverage: float

@dataclass(frozen=True)
class EnsembleConfig:
    enabled: bool
    secondary_model: Path
    secondary_port: int
    overlap_threshold: float
    agree_threshold: float
    hint_threshold: float
    min_cue_duration: float
    min_cue_chars: int
    llm_adjudicate: bool
    llm_batch_size: int
    model: str = "deepseek-chat"

@dataclass(frozen=True)
class UvrConfig:
    enabled: bool
    model: str
    output_single_stem: str
    model_dir: str

@dataclass(frozen=True)
class DiagnosticsConfig:
    enabled: bool
    abort_on_loop: bool
    save_intermediates: bool
    save_raw_whisper: bool = False

@dataclass(frozen=True)
class WhisperConfig:
    mode: str
    cli: Path
    server_exe: Path
    model: Path
    vad_model: Path
    language: str
    beam_size: int
    best_of: int
    max_line_length: int
    server_host: str
    server_port: int
    server_threads: int
    server_startup_timeout: float
    server_request_timeout: float
    strategy: str                     # ← new: "whole" or "vad_segments"
    vad_segments: VadSegmentsConfig   # ← new
    ensemble: EnsembleConfig  
    diagnostics: DiagnosticsConfig

@dataclass(frozen=True)
class VadConfig:
    exe: Path
    model: Path
    max_merge_duration: float
    max_speech_duration: float

@dataclass(frozen=True)
class TranslationConfig:
    api_key: str
    base_url: str
    model: str
    disable_thinking: bool
    source_language: str
    target_language: str
    batch_size: int
    max_retries: int
    min_split_size: int
    context: str

@dataclass(frozen=True)
class TranscriptionConfig:
    enabled: bool


@dataclass(frozen=True)
class OcrFilterConfig:
    min_duration: float
    max_duration: float
    min_box_height_ratio: float
    max_box_height_ratio: float
    min_text_length: int
    min_confidence: float
    max_text_repeats: int
    timestamp_padding: float
    blocklist_texts: tuple[str, ...]
    blocklist_patterns: tuple[str, ...]
    repeat_span_seconds: float = 20.0

@dataclass(frozen=True)
class LlmFilterConfig:
    enabled: bool
    model: str
    batch_size: int
    min_events: int
    source_language: str
    content_type: str

@dataclass(frozen=True)
class OcrConfig:
    enabled: bool
    det_model: Path
    rec_model: Path
    dict_file: Path
    providers: tuple[str, ...]
    sample_fps: int
    rec_confidence: float
    llm_filter: LlmFilterConfig
    crop_top_ratio: float
    crop_bottom_ratio: float
    crop_left_ratio: float
    crop_right_ratio: float
    filter: OcrFilterConfig

@dataclass(frozen=True)
class MergeConfig:
    overlap_threshold: float
    agree_threshold: float
    fill_gaps: bool
    gap_fill_min_duration: float
    gap_fill_prefix: str
    keep_unrelated_ocr: bool = True
    unrelated_threshold: float = 0.3
    info_min_chars: int = 20


@dataclass(frozen=True)
class SubtitleConfig:
    """Final-SRT shaping: split over-long cues into readable ones."""

    max_chars_per_cue: int = 40
    max_cue_duration: float = 6.0
    min_cue_duration: float = 0.5


@dataclass(frozen=True)
class Config:
    chunk_length_seconds: int
    whisper: WhisperConfig
    vad: VadConfig
    translation: TranslationConfig
    transcription: TranscriptionConfig
    ocr: OcrConfig
    merge: MergeConfig
    uvr: UvrConfig
    subtitles: SubtitleConfig = SubtitleConfig()




def _env_path(name: str) -> Path:
    """Read a required .env variable and return it as a Path.

    Raises PipelineError (instead of a bare KeyError) when the variable
    is missing or empty, so the CLI can report a clean message.
    """
    value = os.environ.get(name, "")
    if not value.strip():
        raise PipelineError(
            f"Missing environment variable {name}. "
            f"Check that .env exists and defines it."
        )
    return Path(value)


def load_config(yaml_path: Path = Path("config/default.yaml")) -> Config:
    load_dotenv()
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"Config file not found: {yaml_path}") from exc
    except yaml.YAMLError as exc:
        raise PipelineError(f"Invalid YAML in {yaml_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PipelineError(f"Config file {yaml_path} must contain a YAML mapping")
    return Config(
        chunk_length_seconds=raw["chunk_length_seconds"],
        whisper=WhisperConfig(
            mode=raw["whisper"].get("mode", "cli"),
            cli=_env_path("WHISPER_CLI"),
            server_exe=_env_path("WHISPER_SERVER"),
            model=_env_path("WHISPER_MODEL"),
            vad_model=_env_path("VAD_MODEL"),
            language=raw["whisper"]["language"],
            beam_size=raw["whisper"]["beam_size"],
            best_of=raw["whisper"]["best_of"],
            max_line_length=raw["whisper"]["max_line_length"],
            server_host=raw["whisper"].get("server_host", "127.0.0.1"),
            server_port=raw["whisper"].get("server_port", 8080),
            server_threads=raw["whisper"].get("server_threads", 8),
            server_startup_timeout=raw["whisper"].get("server_startup_timeout", 60.0),
            server_request_timeout=raw["whisper"].get("server_request_timeout", 300.0),
            strategy=raw["whisper"].get("strategy", "whole"),
            vad_segments=VadSegmentsConfig(
                padding=raw["whisper"].get("vad_segments", {}).get("padding", 0.15),
                merge_gap=raw["whisper"].get("vad_segments", {}).get("merge_gap", 0.0),
                min_segment_duration=raw["whisper"].get("vad_segments", {}).get("min_segment_duration", 0.3),
                max_batch_duration=raw["whisper"].get("vad_segments", {}).get("max_batch_duration", 0.0),
                silence_ms=raw["whisper"].get("vad_segments", {}).get("silence_ms", 200),
                min_coverage=raw["whisper"].get("vad_segments", {}).get("min_coverage", 0.9),
            ),
            ensemble=_load_ensemble_config(raw["whisper"]),
            diagnostics=DiagnosticsConfig(
                enabled=raw["whisper"].get("diagnostics", {}).get("enabled", True),
                abort_on_loop=raw["whisper"].get("diagnostics", {}).get("abort_on_loop", True),
                save_intermediates=raw["whisper"].get("diagnostics", {}).get("save_intermediates", True),
                save_raw_whisper=raw["whisper"].get("diagnostics", {}).get("save_raw_whisper", False),
            ),
        ),
        vad=VadConfig(
            exe=_env_path("VAD_EXE"),
            model=_env_path("VAD_MODEL"),
            max_merge_duration=raw["vad"]["max_merge_duration"],
            max_speech_duration=raw["vad"].get("max_speech_duration", 6.0),  # ← new
        ),
        uvr=UvrConfig(
            enabled=raw.get("uvr", {}).get("enabled", False),
            model=raw.get("uvr", {}).get("model", "UVR_MDXNET_Main.onnx"),
            output_single_stem=raw.get("uvr", {}).get("output_single_stem", "Vocals"),
            model_dir=raw.get("uvr", {}).get("model_dir", ""),
        ),
        translation=TranslationConfig(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=raw["translation"]["base_url"],
            model=raw["translation"]["model"],
            disable_thinking=raw["translation"].get("disable_thinking", True),
            source_language=raw["translation"]["source_language"],
            target_language=raw["translation"]["target_language"],
            batch_size=raw["translation"]["batch_size"],
            max_retries=raw["translation"]["max_retries"],
            min_split_size=raw["translation"]["min_split_size"],
            context=raw["translation"]["context"],
        ),
        transcription=TranscriptionConfig(
            enabled=raw.get("transcription", {}).get("enabled", True),
        ),
        ocr=_load_ocr_config(raw["ocr"]),
        merge=MergeConfig(
            overlap_threshold=raw["merge"]["overlap_threshold"],
            agree_threshold=raw["merge"]["agree_threshold"],
            fill_gaps=raw["merge"]["fill_gaps"],
            gap_fill_min_duration=raw["merge"]["gap_fill_min_duration"],
            gap_fill_prefix=raw["merge"].get("gap_fill_prefix", ""),
            keep_unrelated_ocr=raw["merge"].get("keep_unrelated_ocr", True),
            unrelated_threshold=raw["merge"].get("unrelated_threshold", 0.3),
            info_min_chars=raw["merge"].get("info_min_chars", 20),
        ),
        subtitles=SubtitleConfig(
            max_chars_per_cue=raw.get("subtitles", {}).get("max_chars_per_cue", 40),
            max_cue_duration=raw.get("subtitles", {}).get("max_cue_duration", 6.0),
            min_cue_duration=raw.get("subtitles", {}).get("min_cue_duration", 0.5),
        ),
    )



def with_mode(cfg: Config, mode: str) -> Config:
    """Return a copy of cfg with transcription/ocr enabled flags set
    according to mode.

    Modes:
      - "whisper": transcription on, OCR off
      - "ocr":     transcription off, OCR on
      - "merged":  both on
    """
    if mode == "whisper":
        return replace(
            cfg,
            transcription=TranscriptionConfig(enabled=True),
            ocr=replace(cfg.ocr, enabled=False),
        )
    if mode == "ocr":
        return replace(
            cfg,
            transcription=TranscriptionConfig(enabled=False),
            ocr=replace(cfg.ocr, enabled=True),
        )
    if mode == "merged":
        return replace(
            cfg,
            transcription=TranscriptionConfig(enabled=True),
            ocr=replace(cfg.ocr, enabled=True),
        )
    raise ValueError(f"Unknown mode: {mode}")

def _load_ocr_config(raw: dict) -> OcrConfig:
    return OcrConfig(
        enabled=raw.get("enabled", False),
        det_model=_env_path("OCR_DET_MODEL"),
        rec_model=_env_path("OCR_REC_MODEL"),
        dict_file=_env_path("OCR_DICT_FILE"),
        providers=tuple(raw.get("providers", ["CPUExecutionProvider"])),
        sample_fps=raw.get("sample_fps", 3),
        rec_confidence=raw.get("rec_confidence", 0.5),
        crop_top_ratio=raw.get("crop_top_ratio", 0.0),
        crop_bottom_ratio=raw.get("crop_bottom_ratio", 0.0),
        crop_left_ratio=raw.get("crop_left_ratio", 0.0),
        crop_right_ratio=raw.get("crop_right_ratio", 0.0),
        llm_filter=LlmFilterConfig(
            enabled=raw.get("llm_filter", {}).get("enabled", False),
            model=raw.get("llm_filter", {}).get("model", "deepseek-chat"),
            batch_size=raw.get("llm_filter", {}).get("batch_size", 100),
            min_events=raw.get("llm_filter", {}).get("min_events", 5),
            source_language=raw.get("llm_filter", {}).get("source_language", "Korean"),
            content_type=raw.get("llm_filter", {}).get("content_type", "variety show"),
        ),
        filter=OcrFilterConfig(
            min_duration=raw["filter"]["min_duration"],
            max_duration=raw["filter"]["max_duration"],
            min_box_height_ratio=raw["filter"]["min_box_height_ratio"],
            max_box_height_ratio=raw["filter"]["max_box_height_ratio"],
            min_text_length=raw["filter"]["min_text_length"],
            timestamp_padding=raw["filter"].get("timestamp_padding", 0.25), # default is 0.25
            min_confidence=raw["filter"]["min_confidence"],
            max_text_repeats=raw["filter"]["max_text_repeats"],   
            blocklist_texts=tuple(raw["filter"].get("blocklist_texts", [])),
            blocklist_patterns=tuple(raw["filter"].get("blocklist_patterns", [])),
            repeat_span_seconds=raw["filter"].get("repeat_span_seconds", 20.0),
        ),
    )


def _load_ensemble_config(whisper_raw: dict) -> EnsembleConfig:
    raw = whisper_raw.get("ensemble", {})
    enabled = raw.get("enabled", False)

    if enabled:
        if "WHISPER_MODEL_SECONDARY" not in os.environ:
            raise PipelineError(
                "whisper.ensemble.enabled is true but "
                "WHISPER_MODEL_SECONDARY is not set in .env"
            )
        secondary_model = Path(os.environ["WHISPER_MODEL_SECONDARY"])
    else:
        # Dummy value; never used when disabled
        secondary_model = Path()

    return EnsembleConfig(
        enabled=enabled,
        secondary_model=secondary_model,
        secondary_port=raw.get("secondary_port", 8081),
        overlap_threshold=raw.get("overlap_threshold", 0.4),
        agree_threshold=raw.get("agree_threshold", 0.85),
        hint_threshold=raw.get("hint_threshold", 0.5),
        min_cue_duration=raw.get("min_cue_duration", 0.5),
        min_cue_chars=raw.get("min_cue_chars", 2),
        llm_adjudicate=raw.get("llm_adjudicate", True),
        llm_batch_size=raw.get("llm_batch_size", 50),
        model=raw.get("model", "deepseek-chat"),
    )
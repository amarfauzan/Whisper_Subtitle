import os
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class WhisperConfig:
    cli: Path
    model: Path
    vad_model: Path
    language: str
    beam_size: int
    best_of: int
    max_line_length: int

@dataclass(frozen=True)
class VadConfig:
    exe: Path
    model: Path
    max_merge_duration: float

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
    blocklist_texts: tuple[str, ...]
    blocklist_patterns: tuple[str, ...]


@dataclass(frozen=True)
class OcrConfig:
    enabled: bool
    det_model: Path
    rec_model: Path
    dict_file: Path
    providers: tuple[str, ...]
    sample_fps: int
    rec_confidence: float
    crop_top_ratio: float
    crop_bottom_ratio: float
    crop_left_ratio: float
    crop_right_ratio: float
    filter: OcrFilterConfig

@dataclass(frozen=True)
class Config:
    chunk_length_seconds: int
    whisper: WhisperConfig
    vad: VadConfig
    translation: TranslationConfig
    transcription: TranscriptionConfig
    ocr: OcrConfig



def load_config(yaml_path: Path = Path("config/default.yaml")) -> Config:
    load_dotenv()
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return Config(
        chunk_length_seconds=raw["chunk_length_seconds"],
        whisper=WhisperConfig(
            cli=Path(os.environ["WHISPER_CLI"]),
            model=Path(os.environ["WHISPER_MODEL"]),
            vad_model=Path(os.environ["VAD_MODEL"]),
            language=raw["whisper"]["language"],
            beam_size=raw["whisper"]["beam_size"],
            best_of=raw["whisper"]["best_of"],
            max_line_length=raw["whisper"]["max_line_length"],
        ),
        vad=VadConfig(
            exe=Path(os.environ["VAD_EXE"]),
            model=Path(os.environ["VAD_MODEL"]),
            max_merge_duration=raw["vad"]["max_merge_duration"],
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
        det_model=Path(os.environ["OCR_DET_MODEL"]),
        rec_model=Path(os.environ["OCR_REC_MODEL"]),
        dict_file=Path(os.environ["OCR_DICT_FILE"]),
        providers=tuple(raw.get("providers", ["CPUExecutionProvider"])),
        sample_fps=raw.get("sample_fps", 3),
        rec_confidence=raw.get("rec_confidence", 0.5),
        crop_top_ratio=raw.get("crop_top_ratio", 0.0),
        crop_bottom_ratio=raw.get("crop_bottom_ratio", 0.0),
        crop_left_ratio=raw.get("crop_left_ratio", 0.0),
        crop_right_ratio=raw.get("crop_right_ratio", 0.0),
        filter=OcrFilterConfig(
            min_duration=raw["filter"]["min_duration"],
            max_duration=raw["filter"]["max_duration"],
            min_box_height_ratio=raw["filter"]["min_box_height_ratio"],
            max_box_height_ratio=raw["filter"]["max_box_height_ratio"],
            min_text_length=raw["filter"]["min_text_length"],
            min_confidence=raw["filter"]["min_confidence"],
            blocklist_texts=tuple(raw["filter"].get("blocklist_texts", [])),
            blocklist_patterns=tuple(raw["filter"].get("blocklist_patterns", [])),
        ),
    )



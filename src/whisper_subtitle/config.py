import os
from dataclasses import dataclass
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
    source_language: str
    target_language: str
    batch_size: int
    max_retries: int
    min_split_size: int
    context: str

@dataclass(frozen=True)
class Config:
    chunk_length_seconds: int
    whisper: WhisperConfig
    vad: VadConfig
    translation: TranslationConfig


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
            # API key may be empty — we only require it when --translate is used.
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=raw["translation"]["base_url"],
            model=raw["translation"]["model"],
            source_language=raw["translation"]["source_language"],
            target_language=raw["translation"]["target_language"],
            batch_size=raw["translation"]["batch_size"],
            max_retries=raw["translation"]["max_retries"],
            min_split_size=raw["translation"]["min_split_size"],
            context=raw["translation"]["context"],
        ),
    )

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class WhisperConfig:
    cli: Path
    model: Path
    language: str
    beam_size: int
    best_of: int


@dataclass(frozen=True)
class VadConfig:
    exe: Path
    model: Path
    min_subtitle_duration: float
    max_subtitle_duration: float


@dataclass(frozen=True)
class Config:
    chunk_length_seconds: int
    whisper: WhisperConfig
    vad: VadConfig


def load_config(yaml_path: Path = Path("config/default.yaml")) -> Config:
    load_dotenv()
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return Config(
        chunk_length_seconds=raw["chunk_length_seconds"],
        whisper=WhisperConfig(
            cli=Path(os.environ["WHISPER_CLI"]),
            model=Path(os.environ["WHISPER_MODEL"]),
            language=raw["whisper"]["language"],
            beam_size=raw["whisper"]["beam_size"],
            best_of=raw["whisper"]["best_of"],
        ),
        vad=VadConfig(
            exe=Path(os.environ["VAD_EXE"]),
            model=Path(os.environ["VAD_MODEL"]),
            min_subtitle_duration=raw["vad"]["min_subtitle_duration"],
            max_subtitle_duration=raw["vad"]["max_subtitle_duration"],
        ),
    )

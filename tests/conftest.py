from pathlib import Path

from whisper_subtitle.config import (
    EnsembleConfig,
    VadSegmentsConfig,
    WhisperConfig,
)


def default_whisper_config(**overrides) -> WhisperConfig:
    """Build a WhisperConfig with all fields populated.

    Tests use this instead of constructing WhisperConfig directly.
    When a new field is added to WhisperConfig, update only this
    function — every test helper keeps working.
    """
    base = dict(
        mode="cli",
        cli=Path("/fake/cli"),
        server_exe=Path("/fake/server"),
        model=Path("/fake/model.bin"),
        vad_model=Path("/fake/vad.bin"),
        language="ko",
        beam_size=3,
        best_of=3,
        max_line_length=30,
        server_host="127.0.0.1",
        server_port=8080,
        server_threads=8,
        server_startup_timeout=60.0,
        server_request_timeout=300.0,
        strategy="whole",
        vad_segments=VadSegmentsConfig(
            padding=0.15,
            merge_gap=0.0,
            min_segment_duration=0.3,
            max_batch_duration=0.0,
            silence_ms=200,
            min_coverage=0.9,
        ),
        ensemble=EnsembleConfig(
            enabled=False,
            secondary_model=Path("/fake/v3.bin"),
            secondary_port=8081,
            overlap_threshold=0.4,
            agree_threshold=0.85,
            hint_threshold=0.5,
            min_cue_duration=0.5,
            min_cue_chars=2,
            llm_adjudicate=True,
            llm_batch_size=50,
        ),
    )
    base.update(overrides)
    return WhisperConfig(**base)
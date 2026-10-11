"""Optional vocal separation using MDX-Net via audio-separator.

Runs as a preprocessing step between extract_audio and VAD. The MDX-Net
model separates the mixed audio into vocals + instrumental; we keep the
vocals only.

Resume behavior: if <audio>_vocals.wav exists and is newer than the
input, the existing file is returned without re-running the model.

Speed note: MDX-Net on CPU runs at roughly 0.8x real-time. A 10-minute
chunk takes ~8 minutes to separate. This is why the stage is opt-in.
"""

import logging
from pathlib import Path

from audio_separator.separator import Separator

from whisper_subtitle.config import UvrConfig
from whisper_subtitle.exceptions import PipelineError

log = logging.getLogger(__name__)


def separate_vocals(audio_path: Path, cfg: UvrConfig) -> Path:
    """Run MDX-Net on an audio file. Returns path to vocals-only output."""
    if not audio_path.exists():
        raise PipelineError(f"Audio not found: {audio_path}")

    output_path = audio_path.with_name(f"{audio_path.stem}_vocals.wav")

    if (
        output_path.exists()
        and output_path.stat().st_mtime >= audio_path.stat().st_mtime
    ):
        log.info("Vocal separation already exists, skipping: %s", output_path.name)
        return output_path

    log.info("Loading MDX-Net model: %s", cfg.model)

    kwargs = {
        "output_dir": str(audio_path.parent),
        "output_format": "wav",
        "output_single_stem": cfg.output_single_stem,
    }
    if cfg.model_dir:
        kwargs["model_file_dir"] = str(cfg.model_dir)

    separator = Separator(**kwargs)

    try:
        separator.load_model(model_filename=cfg.model)
    except Exception as exc:
        raise PipelineError(f"Failed to load MDX-Net model: {exc}") from exc

    log.info("Separating vocals: %s", audio_path.name)
    try:
        outputs = separator.separate(str(audio_path))
    except Exception as exc:
        raise PipelineError(f"MDX-Net separation failed: {exc}") from exc

    if not outputs:
        raise PipelineError(
            f"MDX-Net produced no output for {audio_path.name}"
        )

    produced = Path(outputs[0])
    if not produced.is_absolute():
        produced = audio_path.parent / produced

    if not produced.exists():
        raise PipelineError(
            f"MDX-Net reported output but file missing: {produced}"
        )

    if produced != output_path:
        produced.rename(output_path)

    log.info("Vocal separation complete: %s", output_path.name)
    return output_path
from whisper_subtitle.config import load_config

MINIMAL_YAML = """\
chunk_length_seconds: 600
whisper:
  language: ko
  beam_size: 3
  best_of: 3
  max_line_length: 30
vad:
  max_merge_duration: 2.0
translation:
  base_url: https://api.deepseek.com/v1
  model: deepseek-chat
  source_language: Korean
  target_language: English
  batch_size: 100
  max_retries: 3
  min_split_size: 50
  context: ""
transcription:
  enabled: true
ocr:
  enabled: false
  filter:
    min_duration: 0.5
    max_duration: 8.0
    min_box_height_ratio: 0.02
    max_box_height_ratio: 1.0
    min_text_length: 2
    min_confidence: 0.84
    max_text_repeats: 8
merge:
  overlap_threshold: 0.3
  agree_threshold: 0.9
  fill_gaps: true
  gap_fill_min_duration: 0.5
"""


def test_uvr_disabled_by_default(tmp_path):
    # Defaults come from a config without a uvr block — deliberately
    # independent of the user's live config/default.yaml, which they
    # may toggle on/off while tuning.
    yaml_path = tmp_path / "no_uvr.yaml"
    yaml_path.write_text(MINIMAL_YAML, encoding="utf-8")
    cfg = load_config(yaml_path)
    assert cfg.uvr.enabled is False
    assert cfg.uvr.model == "UVR_MDXNET_Main.onnx"
    assert cfg.uvr.output_single_stem == "Vocals"


def test_uvr_loads_from_yaml(tmp_path):
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(
        MINIMAL_YAML
        + """
uvr:
  enabled: true
  model: CustomModel.onnx
  output_single_stem: Vocals
  model_dir: /custom
""",
        encoding="utf-8",
    )
    cfg = load_config(yaml_path)
    assert cfg.uvr.enabled is True
    assert cfg.uvr.model == "CustomModel.onnx"
    assert cfg.uvr.model_dir == "/custom"

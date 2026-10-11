from whisper_subtitle.config import load_config


def test_uvr_disabled_by_default():
    cfg = load_config()
    assert cfg.uvr.enabled is False


def test_uvr_loads_from_yaml(tmp_path):
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text("""
chunk_length_seconds: 600
whisper: {...}
vad: {...}
translation: {...}
transcription:
  enabled: true
ocr:
  enabled: false
merge: {...}
uvr:
  enabled: true
  model: CustomModel.onnx
  output_single_stem: Vocals
  model_dir: /custom
""", encoding="utf-8")
    cfg = load_config(yaml_path)
    assert cfg.uvr.enabled is True
    assert cfg.uvr.model == "CustomModel.onnx"
    assert cfg.uvr.model_dir == "/custom"
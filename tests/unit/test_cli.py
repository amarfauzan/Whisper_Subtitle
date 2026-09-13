from whisper_subtitle.cli import main


def test_missing_file_returns_1(tmp_path):
    rc = main([str(tmp_path / "nope.mp4")])
    assert rc == 1

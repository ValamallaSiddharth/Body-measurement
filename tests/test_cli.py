import json
from unittest.mock import patch
from PIL import Image

from bodymeasure.cli import main


def test_cli_calls_same_pipeline(tmp_path, capsys):
    image = tmp_path / "photo.png"
    Image.new("RGB", (128, 128)).save(image)
    with patch("bodymeasure.pipeline.measure", return_value={"status": "ok"}) as measure:
        assert main(["--front", str(image), "--side", str(image), "--gender", "female", "--height-cm", "170"]) == 0
    assert measure.call_args.args[2:] == ("female", {"height_cm": 170.})
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_cli_file_error_is_json(capsys):
    assert main(["--front", "missing-file.jpg", "--side", "missing-file.jpg", "--gender", "male"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"

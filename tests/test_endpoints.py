import io
from unittest.mock import patch

import pytest
from PIL import Image

from bodymeasure.web import create_app, PROCESSING_SLOT


def upload(size=(128, 128)):
    data = io.BytesIO()
    Image.new("RGB", size).save(data, "PNG")
    data.seek(0)
    return data, "test.png"


@pytest.mark.parametrize("height", [None, "175.5"])
def test_endpoint_calls_shared_pipeline(height):
    data = {"front_image": upload(), "side_image": upload(), "gender": "MALE"}
    if height is not None:
        data["height_cm"] = height
    with patch("bodymeasure.pipeline.measure", return_value={"status": "ok"}) as measure:
        response = create_app().test_client().post("/measure", data=data)
    assert response.status_code == 200
    assert measure.call_args.args[2:] == ("male", {"height_cm": float(height) if height else None})


@pytest.mark.parametrize("change", ["missing", "gender", "tiny", "height"])
def test_invalid_endpoint_input(change):
    data = {"front_image": upload(), "side_image": upload(), "gender": "male"}
    if change == "missing":
        data.pop("front_image")[0].close()
    elif change == "gender":
        data["gender"] = "unknown"
    elif change == "tiny":
        data["front_image"][0].close()
        data["front_image"] = upload((127, 128))
    else:
        data["height_cm"] = "NaN"
    with patch("bodymeasure.pipeline.measure") as measure:
        response = create_app().test_client().post("/measure", data=data)
    assert response.status_code == 400
    assert response.json["status"] == "error"
    measure.assert_not_called()


def test_busy_and_health():
    client = create_app().test_client()
    assert client.get("/health").json["status"] == "ok"
    PROCESSING_SLOT.acquire()
    try:
        assert client.post("/measure", data={"gender": "male"}).status_code == 503
    finally:
        PROCESSING_SLOT.release()

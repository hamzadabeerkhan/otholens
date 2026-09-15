from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def image_bytes() -> bytes:
    stream = BytesIO()
    Image.new("L", (128, 96), color=120).save(stream, format="PNG")
    return stream.getvalue()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["research_only"] is True


def test_analysis_returns_versioned_placeholder() -> None:
    response = client.post("/v1/studies/analyse", files={"file": ("synthetic.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["schema_version"] == "1.0"
    assert result["research_only"] is True
    assert result["prediction"]["model_version"] == "placeholder-not-trained"
    assert sum(result["prediction"]["probabilities"].values()) == 1.0
    assert result["input"]["processed_width"] == 512


def test_invalid_upload_fails_without_result() -> None:
    response = client.post("/v1/studies/analyse", files={"file": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 422


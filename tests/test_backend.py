"""Unit and integration tests for FastAPI backend routes, validation, caps, and inference.

Tests:
- GET /api/health (with and without checkpoint)
- GET /api/models
- GET /api/metrics
- CORS restriction to localhost:3000
- POST /api/analyze with valid 15 s MP3
- POST /api/analyze with valid 30 s MP3
- POST /api/analyze with renamed .txt file
- POST /api/analyze with corrupt audio file
- POST /api/analyze with oversize file (> 30 MB)
- POST /api/analyze with audio exceeding 5-minute cap (> 300 s)
- POST /api/analyze with missing checkpoint
- Deletion of temporary files after request completion
"""

import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from backend.app.main import app
from backend.app.services.genre import GenreModelService, get_genre_service
from src.config import load_config

client = TestClient(app)


def create_in_memory_mp3(duration_sec: float, sr: int = 22050, freq: float = 440.0) -> bytes:
    """Generate a valid MP3 file in memory with specified duration."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    signal = 0.4 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="MP3")
    return buf.getvalue()


def test_health_endpoint_normal():
    """Verify GET /api/health returns 200 with model loaded."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "unavailable")
    assert "model_loaded" in data
    assert "version" in data
    assert "device" in data


def test_health_endpoint_missing_checkpoint(monkeypatch):
    """Verify GET /api/health reports status 'unavailable' when checkpoint is missing."""
    dummy_service = GenreModelService(checkpoint_path="non_existent/path/genre_cnn.pt")
    with patch("backend.app.main.get_genre_service", return_value=dummy_service):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["model_loaded"] is False
        assert "not found" in data["detail"].lower() or "missing" in data["detail"].lower()


def test_models_endpoint():
    """Verify GET /api/models returns model schema, genres, and preprocessing params."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "status" in data
    assert "class_names" in data
    assert "preprocessing_params" in data
    assert "normalization_stats" in data
    if data["status"] == "available":
        assert len(data["class_names"]) == 8


def test_metrics_endpoint_missing_cnn_val():
    """Verify GET /api/metrics reports 'validation metrics not available yet' when cnn_val.json is missing.

    Strict contract: Must never read pilot_* files even if present on disk.
    """
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["cnn_validation"] is None
    assert data["cnn_status"] == "validation metrics not available yet"
    # Verify no pilot metrics leak into the response
    assert "pilot_cnn_2epoch" not in response.text
    assert "0.19642857" not in response.text


def test_metrics_endpoint_with_cnn_val(tmp_path):
    """Verify GET /api/metrics parses cnn_val.json with status and epoch fields when present."""
    fake_val = {
        "status": "completed",
        "epochs_run": 25,
        "epochs_configured": 25,
        "best_epoch": 18,
        "best_validation_macro_f1": 0.584,
        "classification_report": {"accuracy": 0.60},
    }
    metrics_dir = tmp_path / "reports" / "metrics"
    metrics_dir.mkdir(parents=True)
    fake_val_file = metrics_dir / "cnn_val.json"
    fake_val_file.write_text(json.dumps(fake_val), encoding="utf-8")

    with patch("backend.app.main.get_project_root", return_value=tmp_path):
        response = client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["cnn_status"] == "available"
        assert data["run_status"] == "completed"
        assert data["epochs_run"] == 25
        assert data["epochs_configured"] == 25



def test_cors_restrictions():
    """Verify CORS permits localhost:3000 and rejects external origins."""
    # Permitted origin: http://localhost:3000
    res_allowed = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res_allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"

    # Permitted origin: http://127.0.0.1:3000
    res_allowed_ip = client.options(
        "/api/health",
        headers={
            "Origin": "http://127.0.0.1:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res_allowed_ip.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"

    # Prohibited origin: https://malicious-site.com
    res_disallowed = client.options(
        "/api/health",
        headers={
            "Origin": "https://malicious-site.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in res_disallowed.headers


def test_analyze_valid_30s_mp3():
    """Verify POST /api/analyze on 30s MP3 produces valid downsampled visualizations and predictions."""
    real_mp3_path = Path("data/raw/fma_small/000/000182.mp3")
    if real_mp3_path.exists():
        with open(real_mp3_path, "rb") as f:
            mp3_bytes = f.read()
    else:
        mp3_bytes = create_in_memory_mp3(30.0)

    # Track files before request to verify cleanup
    temp_dir = Path(tempfile.gettempdir())
    existing_temp = set(temp_dir.glob("sonicscope_*.tmp"))

    response = client.post(
        "/api/analyze",
        files={"file": ("000182.mp3", mp3_bytes, "audio/mpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    # Visualizations downsampling constraints
    assert "visualizations" in data
    assert data["visualizations"]["waveform_points"] <= 1000
    assert len(data["visualizations"]["waveform"]) <= 1000

    spec = data["visualizations"]["spectrogram"]
    assert len(spec) <= 128
    if len(spec) > 0:
        assert len(spec[0]) <= 300

    # Acoustic features
    features = data["acoustic_features"]
    assert features["bpm_label"] == "estimate"
    assert "tempo_bpm_estimate" in features
    assert "rms_mean" in features
    assert len(features["chroma"]) == 12

    # Genre classification
    assert "genre" in data
    assert data["genre"]["top_label"] in [s["label"] for s in data["genre"]["scores"]]
    assert 0.0 <= data["genre"]["confidence"] <= 1.0

    # Verify temp file was deleted
    current_temp = set(temp_dir.glob("sonicscope_*.tmp"))
    assert current_temp == existing_temp, "Temporary file was not cleaned up!"


def test_analyze_valid_15s_mp3():
    """Verify POST /api/analyze on 15s MP3 handles zero-padding and returns valid response."""
    mp3_bytes = create_in_memory_mp3(15.0)

    response = client.post(
        "/api/analyze",
        files={"file": ("short_sample.mp3", mp3_bytes, "audio/mpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    assert abs(data["duration_seconds"] - 15.0) < 1.0
    assert data["visualizations"]["waveform_points"] <= 1000
    assert any("zero-padded" in w for w in data["warnings"])


def test_analyze_renamed_txt_file():
    """Verify uploading a text file renamed to .mp3 fails real decode validation with HTTP 400."""
    fake_audio_bytes = b"This is plain text content pretending to be an audio file.\n"

    response = client.post(
        "/api/analyze",
        files={"file": ("fake_song.mp3", fake_audio_bytes, "audio/mpeg")},
    )
    assert response.status_code == 400
    assert "audio could not be decoded" in response.json()["detail"].lower()


def test_analyze_corrupt_file():
    """Verify uploading corrupt random bytes fails real decode validation with HTTP 400."""
    corrupt_bytes = b"\xff\xfb\x90\x44" + b"\x00\x11\x22\x33\xfe\xed\xfa\xce" * 50

    response = client.post(
        "/api/analyze",
        files={"file": ("corrupt.mp3", corrupt_bytes, "audio/mpeg")},
    )
    assert response.status_code == 400
    assert "audio could not be decoded" in response.json()["detail"].lower()


def test_analyze_oversize_file():
    """Verify uploading a file exceeding max_upload_mb is rejected with HTTP 413."""
    cfg = load_config()
    max_mb = cfg.audio.max_upload_mb
    oversize_bytes = b"0" * ((max_mb + 1) * 1024 * 1024)

    response = client.post(
        "/api/analyze",
        files={"file": ("giant_file.mp3", oversize_bytes, "audio/mpeg")},
    )
    assert response.status_code == 413
    assert f"{max_mb} mb limit" in response.json()["detail"].lower()


def test_analyze_duration_exceeded():
    """Verify audio exceeding max_duration_seconds is rejected with HTTP 400."""
    cfg = load_config()
    max_dur = float(cfg.audio.max_duration_seconds)
    max_min = int(max_dur // 60)
    long_mp3_bytes = create_in_memory_mp3(max_dur + 60.0)

    response = client.post(
        "/api/analyze",
        files={"file": ("too_long.mp3", long_mp3_bytes, "audio/mpeg")},
    )
    assert response.status_code == 400
    assert f"{max_min} minutes" in response.json()["detail"].lower()


def test_analyze_missing_checkpoint():
    """Verify POST /api/analyze returns HTTP 503 when the model checkpoint is missing."""
    dummy_service = GenreModelService(checkpoint_path="non_existent/path/genre_cnn.pt")
    mp3_bytes = create_in_memory_mp3(5.0)

    with patch("backend.app.routes.analyze.get_genre_service", return_value=dummy_service):
        response = client.post(
            "/api/analyze",
            files={"file": ("test.mp3", mp3_bytes, "audio/mpeg")},
        )
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()

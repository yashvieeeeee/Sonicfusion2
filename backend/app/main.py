"""FastAPI application entry point for SonicScope AI.

Follows PLAN §13 FastAPI endpoint contract:
- GET /api/health: Model readiness and service status.
- POST /api/analyze: MP3 upload, validation, acoustic features, and genre inference.
- GET /api/models: Model architecture, class names, preprocessing parameters.
- GET /api/metrics: Read-only academic validation metrics.
- CORS restricted strictly to localhost:3000.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.analyze import router as analyze_router
from backend.app.schemas import (
    CatalogTrackResponse,
    HealthResponse,
    MetricsResponse,
    ModelsResponse,
)
from backend.app.services.genre import get_genre_service
from backend.app.services.similarity import get_similarity_service
from src.config import get_project_root, load_config, resolve_path


app = FastAPI(
    title="SonicScope AI API",
    version="1.0.0",
    description="Music intelligence from MP3 audio: genre recognition, acoustic analysis, and interactive spectrograms.",
)

# CORS: Restricted strictly to localhost:3000 per user specification and PLAN §13
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include routes
app.include_router(analyze_router)


@app.get("/api/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Check backend health and ML model availability.

    Reports 'unavailable' if the GenreCNN checkpoint is missing or degraded.
    """
    genre_svc = get_genre_service()
    is_loaded = genre_svc.is_loaded()

    if not is_loaded:
        return HealthResponse(
            status="unavailable",
            model_loaded=False,
            version="1.0",
            device=str(genre_svc.device),
            detail=genre_svc.get_load_error() or "Genre model checkpoint is missing or unloaded.",
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        version="1.0",
        device=str(genre_svc.device),
        detail="GenreCNN checkpoint loaded and ready for inference.",
    )


@app.get("/api/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    """Return trained genre list, preprocessing profile, and model specifications."""
    genre_svc = get_genre_service()
    info = genre_svc.get_model_info()

    cfg = load_config()
    audio_cfg = cfg.get("audio", {})
    max_duration_seconds = float(audio_cfg.get("max_duration_seconds", 900.0))
    max_upload_mb = int(audio_cfg.get("max_upload_mb", 30))
    allowed_formats = list(audio_cfg.get("allowed_formats", ["mp3", "wav", "m4a"]))

    return ModelsResponse(
        model_name=info["model_name"],
        status=info["status"],
        architecture=info["architecture"],
        num_classes=info["num_classes"],
        class_names=info["class_names"],
        checkpoint_version=info["checkpoint_version"],
        sample_rate=info["sample_rate"],
        preprocessing_params=info["preprocessing_params"],
        normalization_stats=info["normalization_stats"],
        max_duration_seconds=max_duration_seconds,
        max_upload_mb=max_upload_mb,
        allowed_formats=allowed_formats,
    )


@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """Return read-only academic validation metrics for the dashboard."""
    root = get_project_root()
    cnn_metrics_path = root / "reports" / "metrics" / "cnn_val.json"
    baseline_metrics_path = root / "reports" / "metrics" / "baseline_val.json"
    audit_path = root / "reports" / "metrics" / "data_exclusions.csv"

    cnn_val = None
    cnn_status = "validation metrics not available yet"
    run_status = None
    epochs_run = None
    epochs_configured = None

    # Strict contract: Only read cnn_val.json; NEVER read pilot_* or fallback to pilot files
    if cnn_metrics_path.exists() and cnn_metrics_path.name == "cnn_val.json":
        try:
            with open(cnn_metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure it is a valid dict and not an empty or pilot artifact
            if isinstance(data, dict):
                cnn_val = data
                cnn_status = "available"
                run_status = data.get("status")
                epochs_run = data.get("epochs_run")
                epochs_configured = data.get("epochs_configured")
        except Exception:
            cnn_val = None
            cnn_status = "validation metrics not available yet"

    baseline_val = None
    if baseline_metrics_path.exists():
        try:
            with open(baseline_metrics_path, "r", encoding="utf-8") as f:
                baseline_val = json.load(f)
        except Exception:
            baseline_val = None

    manifest_summary: Dict[str, Any] = {"manifest_available": False}
    manifest_path = root / "data" / "processed" / "manifest.csv"
    if manifest_path.exists():
        manifest_summary["manifest_available"] = True
        manifest_summary["manifest_path"] = str(manifest_path)

    return MetricsResponse(
        status="available" if (cnn_val or baseline_val) else "no_metrics_found",
        cnn_status=cnn_status,
        cnn_validation=cnn_val,
        baseline_validation=baseline_val,
        manifest_summary=manifest_summary,
        run_status=run_status,
        epochs_run=epochs_run,
        epochs_configured=epochs_configured,
    )


@app.get("/api/catalog/{track_id}", response_model=CatalogTrackResponse)
def get_catalog_track(track_id: int) -> CatalogTrackResponse:
    """Return approved display metadata for a catalog track by ID (PLAN §12, §13)."""
    sim_svc = get_similarity_service()
    if not sim_svc.is_available():
        raise HTTPException(
            status_code=503,
            detail="Similarity catalog index is unavailable.",
        )

    track = sim_svc.get_catalog_track(track_id)
    if track is None:
        raise HTTPException(
            status_code=404,
            detail=f"Track ID {track_id} not found in similarity catalog.",
        )

    return CatalogTrackResponse(
        track_id=int(track["track_id"]),
        title=str(track.get("title", f"FMA Track {track_id}")),
        artist=str(track.get("artist", "Unknown Artist")),
        genre=str(track.get("genre", "Unknown")),
        duration=float(track.get("duration", 30.0)),
    )

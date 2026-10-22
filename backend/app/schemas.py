"""Pydantic schemas and typed API response contracts.

Follows PLAN §13 FastAPI endpoint contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="System status: 'ok' or 'unavailable'")
    model_loaded: bool = Field(..., description="Whether the GenreCNN checkpoint is loaded")
    version: str = Field("1.0", description="API version")
    device: str = Field("cpu", description="Compute device used for inference")
    detail: Optional[str] = Field(None, description="Diagnostic message or warning if degraded")


class GenreScore(BaseModel):
    """Individual genre prediction label and probability score."""

    label: str
    score: float


class GenreResult(BaseModel):
    """Aggregated genre classification result."""

    model: str = Field("GenreCNN (3-block 2D CNN)", description="Model architecture identifier")
    top_label: str = Field(..., description="Highest-scoring genre name")
    confidence: float = Field(..., description="Softmax confidence score for top genre (0.0 - 1.0)")
    is_uncertain: bool = Field(..., description="True if top confidence < 0.40 or margin < 0.15")
    status: str = Field(..., description="'confident' or 'uncertain / multiple influences'")
    scores: List[GenreScore] = Field(..., description="Ranked list of all genre predictions")


class AcousticFeatures(BaseModel):
    """Acoustic features extracted from decoded audio."""

    tempo_bpm_estimate: float = Field(..., description="Algorithmic tempo estimate (BPM)")
    bpm_label: str = Field("estimate", description="Explicit estimate label per PLAN §11")
    rms_mean: float = Field(..., description="Mean Root Mean Square energy")
    rms_std: float = Field(..., description="Standard deviation of RMS energy")
    chroma: List[float] = Field(..., description="12-bin pitch-class activation profile (mean energy)")
    chroma_labels: List[str] = Field(
        default=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        description="Pitch class names for chroma bins",
    )


class Visualizations(BaseModel):
    """Downsampled visualization matrices for web rendering."""

    waveform: List[float] = Field(..., description="Downsampled waveform peak envelope (<= 1000 points)")
    spectrogram: List[List[float]] = Field(
        ..., description="Downsampled log-mel spectrogram matrix (<= 128 freq x 300 time bins)"
    )
    waveform_points: int = Field(..., description="Number of points in downsampled waveform")
    spectrogram_shape: List[int] = Field(..., description="[frequency_bins, time_bins]")


class SimilarTrack(BaseModel):
    """Nearest-neighbor similar catalog track."""

    track_id: int
    title: str
    artist: str
    genre: str
    feature_similarity_score: float = Field(..., description="Cosine feature similarity score (0.0 - 1.0)")
    explanation: str = Field(..., description="Human-readable rationale for similarity")


class CatalogTrackResponse(BaseModel):
    """Catalog track metadata schema."""

    track_id: int
    title: str
    artist: str
    genre: str
    duration: float = 30.0


class AnalysisResponse(BaseModel):
    """Complete audio analysis response."""

    analysis_id: str = Field(..., description="Unique UUID for this analysis session")
    filename: str = Field(..., description="Original client upload filename (sanitized)")
    duration_seconds: float = Field(..., description="Decoded duration of audio file in seconds")
    sample_rate: int = Field(..., description="Processing sample rate in Hz")
    genre: GenreResult = Field(..., description="Genre classification results")
    acoustic_features: AcousticFeatures = Field(..., description="Acoustic measurements")
    visualizations: Visualizations = Field(..., description="Downsampled waveform and spectrogram")
    similar_tracks: List[SimilarTrack] = Field(default_factory=list, description="Top-5 similar catalog songs")
    similarity_status: str = Field("available", description="Similarity status: 'available' or 'unavailable'")
    identical_catalog_track_excluded: bool = Field(False, description="True if an identical catalog track was excluded")
    warnings: List[str] = Field(default_factory=list, description="Academic caveats and warnings")


class ModelsResponse(BaseModel):
    """Status and configuration of active and optional ML models."""

    model_name: str
    status: str
    architecture: str
    num_classes: int
    class_names: List[str]
    checkpoint_version: str
    sample_rate: int
    preprocessing_params: Dict[str, Any]
    normalization_stats: Dict[str, float]
    max_duration_seconds: float = 900.0
    max_upload_mb: int = 30
    allowed_formats: List[str] = ["mp3", "wav", "m4a"]


class MetricsResponse(BaseModel):
    """Read-only academic summary of training, validation, and test metrics."""

    status: str
    cnn_status: str = Field(
        "validation metrics not available yet",
        description="Status of CNN validation metrics: 'available' or 'validation metrics not available yet'",
    )
    cnn_validation: Optional[Dict[str, Any]] = None
    baseline_validation: Optional[Dict[str, Any]] = None
    manifest_summary: Optional[Dict[str, Any]] = None
    run_status: Optional[str] = Field(
        None,
        description="Status of CNN training run: 'completed', 'early_stopped', 'interrupted', 'failed'",
    )
    epochs_run: Optional[int] = Field(None, description="Number of epochs actually executed")
    epochs_configured: Optional[int] = Field(None, description="Number of epochs configured")
    test_status: str = Field(
        "not available yet",
        description="Status of official test split evaluation: 'available' or 'not available yet'",
    )
    test_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Full evaluation results on official test split from test_metrics.json",
    )
    figures_available: List[str] = Field(
        default_factory=list,
        description="List of available figure filenames in reports/figures",
    )
    eda_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Exploratory data analysis summary from eda_summary.json",
    )


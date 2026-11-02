"""Acoustic feature extraction and baseline feature generation."""

from src.features.extraction import (
    FEATURE_NAMES,
    extract_track_features,
    extract_features_from_file,
)
from src.features.embeddings import (
    AudioEmbedder,
    segment_audio_windows,
    MERT_MODEL_ID,
    MODEL_ALIAS,
)

__all__ = [
    "FEATURE_NAMES",
    "extract_track_features",
    "extract_features_from_file",
    "AudioEmbedder",
    "segment_audio_windows",
    "MERT_MODEL_ID",
    "MODEL_ALIAS",
]

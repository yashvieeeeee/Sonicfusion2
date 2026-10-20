"""Acoustic feature extraction for the classical baseline and similarity engine.

Follows PLAN §08 specification:
- Timbral: MFCC 1-13 mean and std (26 features)
- Tonal: 12-bin chroma mean (12 features)
- Spectral: centroid, rolloff, bandwidth, zero-crossing rate: mean and std (8 features)
- Dynamics: RMS energy mean and std (2 features)
- Rhythm: tempo estimate, onset-strength summary (3 features)
Total: 51 acoustic features per track.
Includes explicit NaN, Inf, and silent-clip handling.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Union
import librosa
import numpy as np

from src.audio.loading import load_audio

# Canonical feature column names
FEATURE_NAMES: List[str] = (
    [f"mfcc_{i}_mean" for i in range(1, 14)]
    + [f"mfcc_{i}_std" for i in range(1, 14)]
    + [f"chroma_{i}_mean" for i in range(1, 13)]
    + [
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
        "rms_mean",
        "rms_std",
        "tempo",
        "onset_strength_mean",
        "onset_strength_std",
    ]
)


def _sanitize_val(val: float) -> float:
    """Ensure a float value is finite, replacing NaN/Inf with 0.0."""
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return float(val)


def extract_track_features(y: np.ndarray, sr: int = 22050) -> Dict[str, float]:
    """Extract standard 51-dimensional acoustic feature vector from 1D audio.

    Args:
        y: 1D audio waveform.
        sr: Sample rate in Hz.

    Returns:
        Dict mapping feature names to float values.
    """
    # Explicit silent clip handling (PLAN §08):
    # If the clip is completely silent or extremely quiet, return zeroed feature vector
    max_amp = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    if max_amp < 1e-6:
        return {name: 0.0 for name in FEATURE_NAMES}

    features: Dict[str, float] = {}

    try:
        # 1. Timbral: MFCC 1-13 mean & std (26 columns)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f"mfcc_{i+1}_mean"] = _sanitize_val(float(np.mean(mfcc[i])))
            features[f"mfcc_{i+1}_std"] = _sanitize_val(float(np.std(mfcc[i])))

        # 2. Tonal: 12-bin chroma mean (12 columns)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        for i in range(12):
            features[f"chroma_{i+1}_mean"] = _sanitize_val(float(np.mean(chroma[i])))

        # 3. Spectral features (8 columns)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid_mean"] = _sanitize_val(float(np.mean(centroid)))
        features["spectral_centroid_std"] = _sanitize_val(float(np.std(centroid)))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff_mean"] = _sanitize_val(float(np.mean(rolloff)))
        features["spectral_rolloff_std"] = _sanitize_val(float(np.std(rolloff)))

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features["spectral_bandwidth_mean"] = _sanitize_val(float(np.mean(bandwidth)))
        features["spectral_bandwidth_std"] = _sanitize_val(float(np.std(bandwidth)))

        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        features["zero_crossing_rate_mean"] = _sanitize_val(float(np.mean(zcr)))
        features["zero_crossing_rate_std"] = _sanitize_val(float(np.std(zcr)))

        # 4. Dynamics: RMS energy (2 columns)
        rms = librosa.feature.rms(y=y)[0]
        features["rms_mean"] = _sanitize_val(float(np.mean(rms)))
        features["rms_std"] = _sanitize_val(float(np.std(rms)))

        # 5. Rhythm: tempo estimate and onset strength (3 columns)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(np.atleast_1d(tempo)[0])
        features["tempo"] = _sanitize_val(tempo_val)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features["onset_strength_mean"] = _sanitize_val(float(np.mean(onset_env)))
        features["onset_strength_std"] = _sanitize_val(float(np.std(onset_env)))

    except Exception:
        # Fallback if signal processing encounters unexpected numeric singularity
        for name in FEATURE_NAMES:
            if name not in features:
                features[name] = 0.0

    return features


def extract_features_from_file(
    file_path: Union[str, Path],
    sr: int = 22050,
    duration: float = 30.0,
) -> Dict[str, float]:
    """Load audio file and extract all 51 acoustic features.

    Args:
        file_path: Path to the audio file.
        sr: Sample rate.
        duration: Audio duration in seconds (default 30.0 s).

    Returns:
        Dict mapping feature names to values.
    """
    y = load_audio(file_path, sr=sr, duration=duration, mono=True)
    return extract_track_features(y, sr=sr)

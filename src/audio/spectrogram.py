"""Spectrogram computation, normalization, and tensor conversion.

Follows PLAN §07 audio contract:
- 128 mel bands
- n_fft = 2048
- hop_length = 512
- power = 2.0
- power_to_db with ref=np.max
- Normalization with train-only mean and std
- Output shape [1, 128, ~431]
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Union
import librosa
import numpy as np
import torch


class RunningStats:
    """Online / streaming calculation of mean and variance using Chan/Welford algorithm."""

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def update(self, arr: np.ndarray) -> None:
        """Update running statistics with an array of values."""
        flat = arr.ravel().astype(np.float64)
        n = len(flat)
        if n == 0:
            return
        chunk_mean = float(np.mean(flat))
        chunk_var = float(np.var(flat))
        chunk_M2 = chunk_var * n

        if self.count == 0:
            self.count = n
            self.mean = chunk_mean
            self.M2 = chunk_M2
        else:
            delta = chunk_mean - self.mean
            new_count = self.count + n
            self.mean += delta * (n / new_count)
            self.M2 += chunk_M2 + (delta ** 2) * self.count * n / new_count
            self.count = new_count

    @property
    def variance(self) -> float:
        """Return sample variance."""
        return float(self.M2 / self.count) if self.count > 0 else 0.0

    @property
    def std(self) -> float:
        """Return standard deviation."""
        return math.sqrt(max(0.0, self.variance))


def compute_logmel_spectrogram(
    segment: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    power: float = 2.0,
    top_db: float = 80.0,
) -> np.ndarray:
    """Compute power-to-dB log-mel spectrogram for an audio segment.

    Args:
        segment: 1D audio waveform (expected duration ~10.0 s).
        sr: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length between successive frames.
        n_mels: Number of mel frequency bands.
        power: Exponent for mel power spectrogram (2.0 = power).
        top_db: Threshold decibels below peak for dynamic range clipping.

    Returns:
        2D float32 array of shape (n_mels, frames), e.g., (128, 431).
    """
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
    return logmel.astype(np.float32)


def normalize_spectrogram(
    logmel: np.ndarray,
    mean: float,
    std: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize a log-mel spectrogram using training dataset statistics.

    Args:
        logmel: Unnormalized log-mel spectrogram.
        mean: Training set mean.
        std: Training set standard deviation.
        eps: Small constant for numerical stability.

    Returns:
        Normalized float32 spectrogram: (logmel - mean) / (std + eps).
    """
    return ((logmel - mean) / (std + eps)).astype(np.float32)


def to_cnn_tensor(spectrogram: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert a spectrogram array to a 3D CNN tensor with channel dimension.

    Input shape: (128, 431) or (1, 128, 431).
    Output shape: (1, 128, 431) torch.float32 tensor.
    """
    if isinstance(spectrogram, np.ndarray):
        tensor = torch.from_numpy(spectrogram).float()
    else:
        tensor = spectrogram.float()

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # [1, H, W]
    elif tensor.dim() == 3 and tensor.shape[0] != 1:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}, expected 1 channel")

    return tensor


def save_norm_stats(
    mean: float,
    std: float,
    output_path: Union[str, Path],
    extra_metadata: Dict[str, Any] | None = None,
) -> None:
    """Save training normalization statistics to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "mean": float(mean),
        "std": float(std),
    }
    if extra_metadata:
        payload.update(extra_metadata)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_norm_stats(stats_path: Union[str, Path]) -> Dict[str, float]:
    """Load training normalization statistics from a JSON file."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalization statistics file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "mean": float(data["mean"]),
        "std": float(data["std"]),
    }

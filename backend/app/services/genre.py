"""Genre classification service using trained GenreCNN checkpoint.

Follows PLAN §09 and §13:
- Loads self-contained checkpoint (weights, class names, normalization stats).
- Segments audio into deterministic 10-second windows.
- Normalizes spectrograms with saved training statistics.
- Averages per-window softmax probabilities for track-level inference.
- Flags prediction uncertainty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from src.audio.loading import segment_audio
from src.audio.spectrogram import (
    compute_logmel_spectrogram,
    normalize_spectrogram,
    to_cnn_tensor,
)
from src.config import get_project_root, load_config, resolve_device, resolve_path
from src.models.genre_cnn import GenreCNN, aggregate_window_probabilities


class GenreModelService:
    """Singleton service managing GenreCNN checkpoint loading and inference."""

    def __init__(self, checkpoint_path: Optional[str | Path] = None) -> None:
        self.cfg = load_config()
        self.checkpoint_path = (
            resolve_path(checkpoint_path)
            if checkpoint_path
            else resolve_path(self.cfg.paths.cnn_checkpoint_path)
        )
        self.device = torch.device(resolve_device(self.cfg))

        self.model: Optional[GenreCNN] = None
        self.class_names: List[str] = []
        self.sample_rate: int = 22050
        self.preprocessing_params: Dict[str, Any] = {}
        self.normalization_stats: Dict[str, float] = {}
        self.version: str = "1.0"
        self._load_error: Optional[str] = None

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Attempt to load checkpoint if it exists."""
        if not self.checkpoint_path.exists():
            self._load_error = f"Checkpoint not found at: {self.checkpoint_path}"
            self.model = None
            return

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            self.class_names = checkpoint.get("class_names", [])
            self.sample_rate = int(checkpoint.get("sample_rate", 22050))
            self.preprocessing_params = (
                checkpoint.get("preprocessing_params")
                or checkpoint.get("preprocessing params")
                or {}
            )
            self.normalization_stats = (
                checkpoint.get("normalization_stats")
                or checkpoint.get("normalization stats")
                or {"mean": -43.0, "std": 15.0}
            )
            self.version = str(checkpoint.get("version", "1.0"))

            model = GenreCNN(
                num_classes=len(self.class_names),
                in_channels=1,
                dropout=0.3,
            )
            state_dict = checkpoint.get("state_dict") if "state_dict" in checkpoint else checkpoint["model_state_dict"]
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.model = model
            self._load_error = None
        except Exception as exc:
            self._load_error = f"Failed to load checkpoint: {exc}"
            self.model = None

    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        return self.model is not None

    def get_load_error(self) -> Optional[str]:
        """Return error message if checkpoint failed to load or is missing."""
        return self._load_error

    def reload(self) -> bool:
        """Try reloading the checkpoint from disk."""
        self._load_checkpoint()
        return self.is_loaded()

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for GET /api/models."""
        if not self.is_loaded():
            return {
                "model_name": "GenreCNN (3-block 2D CNN)",
                "status": "unavailable",
                "architecture": "3-block 2D CNN",
                "num_classes": 0,
                "class_names": [],
                "checkpoint_version": "none",
                "sample_rate": 22050,
                "preprocessing_params": {},
                "normalization_stats": {},
            }

        return {
            "model_name": "GenreCNN (3-block 2D CNN)",
            "status": "available",
            "architecture": (
                "Conv2d(1,16,3,p=1)+BN+ReLU+MaxPool(2) -> "
                "Conv2d(16,32,3,p=1)+BN+ReLU+MaxPool(2) -> "
                "Conv2d(32,64,3,p=1)+BN+ReLU+MaxPool(2) -> "
                "AdaptiveAvgPool2d(1)+Flatten+Dropout(0.3)+Linear(64,8)"
            ),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "checkpoint_version": self.version,
            "sample_rate": self.sample_rate,
            "preprocessing_params": self.preprocessing_params,
            "normalization_stats": self.normalization_stats,
        }

    def predict(self, y: np.ndarray, sr: int = 22050) -> Dict[str, Any]:
        """Classify genre on a 1D audio waveform by aggregating 10s window probabilities.

        Args:
            y: Standardized 1D audio array (padded/truncated to 30.0s).
            sr: Audio sample rate in Hz.

        Returns:
            Dictionary matching GenreResult schema.
        """
        if not self.is_loaded() or self.model is None:
            raise RuntimeError(
                f"Genre model is unavailable: {self._load_error or 'Checkpoint not loaded'}"
            )

        win_sec = float(self.preprocessing_params.get("window_duration", 10.0))
        num_windows = int(self.preprocessing_params.get("windows_per_track", 3))
        n_mels = int(self.preprocessing_params.get("n_mels", 128))
        n_fft = int(self.preprocessing_params.get("n_fft", 2048))
        hop_length = int(self.preprocessing_params.get("hop_length", 512))

        mean = float(self.normalization_stats.get("mean", 0.0))
        std = float(self.normalization_stats.get("std", 1.0))

        # Segment into 10s windows
        segments = segment_audio(y, sr=sr, window_sec=win_sec, num_windows=num_windows)

        # Compute log-mels and normalize
        window_tensors: List[torch.Tensor] = []
        for seg in segments:
            raw_logmel = compute_logmel_spectrogram(
                segment=seg,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            normed = normalize_spectrogram(raw_logmel, mean=mean, std=std)
            window_tensors.append(to_cnn_tensor(normed))

        batch = torch.stack(window_tensors, dim=0).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=-1)
            avg_probs = aggregate_window_probabilities(probs)

        # Rank classes
        ranked_indices = np.argsort(avg_probs)[::-1]
        scores = [
            {"label": self.class_names[idx], "score": round(float(avg_probs[idx]), 4)}
            for idx in ranked_indices
        ]

        top_label = scores[0]["label"]
        top_confidence = scores[0]["score"]
        second_confidence = scores[1]["score"] if len(scores) > 1 else 0.0

        # Uncertainty detection per PLAN §09
        confidence_margin = top_confidence - second_confidence
        is_uncertain = bool((top_confidence < 0.40) or (confidence_margin < 0.15))

        return {
            "model": "GenreCNN (3-block 2D CNN)",
            "top_label": top_label,
            "confidence": top_confidence,
            "is_uncertain": is_uncertain,
            "status": "uncertain / multiple influences" if is_uncertain else "confident",
            "scores": scores,
        }


# Global singleton instance
_model_service: Optional[GenreModelService] = None


def get_genre_service(checkpoint_path: Optional[str | Path] = None) -> GenreModelService:
    """Retrieve or initialize the global GenreModelService singleton."""
    global _model_service
    if _model_service is None or checkpoint_path is not None:
        _model_service = GenreModelService(checkpoint_path=checkpoint_path)
    return _model_service

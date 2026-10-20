"""Unit tests for GenreCNN model, augmentation, checkpoint structure, and prediction."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch

from src.models.genre_cnn import GenreCNN, apply_spectrogram_augmentation
from scripts.predict import predict_genre


def test_genre_cnn_forward_shape():
    """Verify GenreCNN accepts [B, 1, 128, 431] and outputs [B, num_classes]."""
    num_classes = 8
    model = GenreCNN(num_classes=num_classes, in_channels=1, dropout=0.3)
    model.eval()

    # Batch of 2 spectrogram windows
    dummy_input = torch.randn(2, 1, 128, 431)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, num_classes), f"Expected (2, {num_classes}), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_genre_cnn_parameter_count():
    """Verify GenreCNN is lightweight (< 100k parameters per §09 design)."""
    model = GenreCNN(num_classes=8, in_channels=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 10_000 < total_params < 100_000, f"Unexpected parameter count: {total_params}"


def test_spectrogram_augmentation():
    """Verify time shift and gain augmentation retain tensor shape and values."""
    spec = torch.ones(1, 128, 431)
    augmented = apply_spectrogram_augmentation(spec, max_time_shift=20, gain_db_range=2.0)

    assert augmented.shape == spec.shape
    assert not torch.isnan(augmented).any()


def test_checkpoint_structure():
    """Verify trained checkpoint contains all required metadata and weights."""
    ckpt_path = Path("models/checkpoints/genre_cnn.pt")
    if not ckpt_path.exists():
        pytest.skip("models/checkpoints/genre_cnn.pt not found (run training first)")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert "state_dict" in checkpoint or "model_state_dict" in checkpoint, (
        "Checkpoint must contain either 'state_dict' or 'model_state_dict'"
    )
    for key in ["class_names", "sample_rate", "preprocessing_params", "normalization_stats", "version"]:
        assert key in checkpoint, f"Missing required checkpoint key: {key}"

    assert len(checkpoint["class_names"]) == 8
    assert checkpoint["sample_rate"] == 22050
    assert "mean" in checkpoint["normalization_stats"]
    assert "std" in checkpoint["normalization_stats"]


def test_predict_genre_pipeline():
    """Verify predict_genre end-to-end on a generated audio file."""
    ckpt_path = Path("models/checkpoints/genre_cnn.pt")
    if not ckpt_path.exists():
        pytest.skip("models/checkpoints/genre_cnn.pt not found")

    # Create a 1-second synthetic sine wave WAV file in temp dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = Path(tmp_dir) / "test_tone.wav"
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(tmp_wav, audio, sr)

        res = predict_genre(
            audio_path=tmp_wav,
            checkpoint_path=ckpt_path,
            top_k=3,
            device_str="cpu",
        )

        assert "predicted_genre" in res
        assert "confidence" in res
        assert "is_uncertain" in res
        assert len(res["top_predictions"]) == 3
        assert len(res["all_class_scores"]) == 8
        assert 0.0 <= res["confidence"] <= 1.0

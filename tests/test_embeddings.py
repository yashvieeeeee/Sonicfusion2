"""Unit tests for pretrained audio embedding segmentation and extraction (5.0s non-overlapping)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.embeddings import (
    DEFAULT_DURATION_SEC,
    DEFAULT_HOP_SEC,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_SEC,
    MODEL_ALIAS,
    segment_audio_windows,
)


def test_segment_audio_windows_5s_no_overlap():
    """Verify that a 30s audio segment produces exactly 6 non-overlapping 5s windows."""
    sr = 24000
    y = np.arange(int(30.0 * sr), dtype=np.float32)

    windows = segment_audio_windows(
        y,
        sr=sr,
        window_sec=5.0,
        hop_sec=5.0,
        target_duration=30.0,
    )

    # Exactly 6 windows of 5 * 24000 = 120000 samples each
    assert windows.shape == (6, 120000)
    assert windows.dtype == np.float32

    # Verify windows are non-overlapping and contiguous
    # Window 0: [0, 120000), Window 1 starts at 120000
    assert windows[0, 0] == 0
    assert windows[0, -1] == 119999
    assert windows[1, 0] == 120000
    assert windows[5, -1] == int(30.0 * sr) - 1


def test_segment_audio_windows_padding_on_short_audio_5s():
    """Verify that audio shorter than 30s is zero-padded to produce 6 windows."""
    sr = 24000
    y_short = np.ones(int(12.0 * sr), dtype=np.float32)

    windows = segment_audio_windows(
        y_short,
        sr=sr,
        window_sec=5.0,
        hop_sec=5.0,
        target_duration=30.0,
    )

    assert windows.shape == (6, 120000)
    # Beyond 12s, samples should be zero (windows 3, 4, 5 are all 0)
    np.testing.assert_allclose(windows[3], 0.0)
    np.testing.assert_allclose(windows[4], 0.0)
    np.testing.assert_allclose(windows[5], 0.0)


def test_saved_5s_embeddings_artifacts_structure():
    """Verify saved 5s embedding artifacts in data/embeddings/MERT-v1-95M-5s."""
    emb_dir = Path("data/embeddings") / MODEL_ALIAS
    if not (emb_dir / "index.csv").exists():
        pytest.skip(f"Embeddings directory {emb_dir} not yet extracted")

    index_df = pd.read_csv(emb_dir / "index.csv")
    assert len(index_df) >= 10
    assert "track_id" in index_df.columns
    assert "num_windows" in index_df.columns
    assert "embedding_dim" in index_df.columns

    # Verify expected window count is 6
    assert (index_df["num_windows"] == 6).all()
    assert (index_df["embedding_dim"] == 768).all()

    sample_id = int(index_df.iloc[0]["track_id"])
    win_file = emb_dir / "windows" / f"{sample_id:06d}.npy"
    mean_file = emb_dir / "mean" / f"{sample_id:06d}.npy"

    assert win_file.exists()
    assert mean_file.exists()

    win_arr = np.load(win_file)
    mean_arr = np.load(mean_file)

    assert win_arr.shape == (6, 768)
    assert mean_arr.shape == (768,)
    # Track mean should match mean of windows
    np.testing.assert_allclose(mean_arr, win_arr.mean(axis=0), atol=1e-5)

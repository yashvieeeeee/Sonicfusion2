"""Unit tests for audio processing, spectrogram caching, and baseline feature extraction.

Verifies requirements from docs/PLAN.md (§05, §07, §08):
- Tensor shape is strictly [1, 128, ~431].
- Validation and test sets use saved training normalization statistics.
- No track appears in more than one split across manifest and features.
- Baseline features in features.parquet match schema and contain no NaN/Inf.
- Silent audio clips and edge cases are handled safely.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.audio.loading import load_audio, segment_audio
from src.audio.spectrogram import (
    compute_logmel_spectrogram,
    load_norm_stats,
    normalize_spectrogram,
    to_cnn_tensor,
)
from src.config import load_config, resolve_path
from src.features.extraction import FEATURE_NAMES, extract_track_features


@pytest.fixture
def cfg():
    """Project configuration fixture."""
    return load_config()


def test_tensor_shape_1_128_431(cfg):
    """Verify spectrogram tensor output shape is strictly [1, 128, 431]."""
    sr = int(cfg.audio.sample_rate)
    win_sec = float(cfg.audio.window_duration)
    n_mels = int(cfg.audio.n_mels)
    n_fft = int(cfg.audio.n_fft)
    hop_length = int(cfg.audio.hop_length)

    # 1. Synthetic 10s audio segment
    synthetic_audio = np.random.randn(int(sr * win_sec)).astype(np.float32)
    spec = compute_logmel_spectrogram(
        segment=synthetic_audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    assert spec.shape == (128, 431), f"Expected shape (128, 431), got {spec.shape}"

    # 2. Converted 3D CNN tensor
    tensor = to_cnn_tensor(spec)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == torch.Size([1, 128, 431])
    assert tensor.dtype == torch.float32

    # 3. Check any cached .npy file on disk if available
    cache_dir = resolve_path(cfg.paths.spectrogram_cache_dir)
    npy_files = list(cache_dir.glob("*.npy"))
    if npy_files:
        cached_arr = np.load(npy_files[0])
        assert cached_arr.shape == (128, 431)
        assert cached_arr.dtype == np.float16
        cached_tensor = to_cnn_tensor(cached_arr)
        assert cached_tensor.shape == torch.Size([1, 128, 431])


def test_val_test_use_saved_train_stats(cfg):
    """Verify validation/test audio is normalized using saved training statistics (PLAN §07)."""
    norm_stats_path = resolve_path(cfg.paths.norm_stats_path)
    assert norm_stats_path.exists(), "norm_stats.json does not exist"

    stats = load_norm_stats(norm_stats_path)
    train_mean = stats["mean"]
    train_std = stats["std"]
    assert train_std > 0.0

    manifest_path = resolve_path(cfg.paths.manifest_path)
    manifest_df = pd.read_csv(manifest_path)

    # Find a validation or test track
    eval_tracks = manifest_df[manifest_df["split"].isin(["validation", "test"])]
    if eval_tracks.empty:
        pytest.skip("No validation/test tracks in current limited manifest")

    eval_row = eval_tracks.iloc[0]
    t_id = int(eval_row["track_id"])
    full_path = resolve_path(str(eval_row["path"]))

    # Compute raw log-mel for window 0
    y = load_audio(full_path, sr=cfg.audio.sample_rate, duration=cfg.audio.duration)
    segments = segment_audio(y, sr=cfg.audio.sample_rate, window_sec=cfg.audio.window_duration)
    raw_spec = compute_logmel_spectrogram(
        segment=segments[0],
        sr=cfg.audio.sample_rate,
        n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length,
        n_mels=cfg.audio.n_mels,
    )

    # Normalize with saved train stats
    expected_normed = normalize_spectrogram(raw_spec, mean=train_mean, std=train_std)

    # Check cached window file
    cached_path = resolve_path(cfg.paths.spectrogram_cache_dir) / f"{t_id:06d}_0.npy"
    if cached_path.exists():
        cached_arr = np.load(cached_path).astype(np.float32)
        # float16 serialization has ~1e-2 tolerance
        assert np.allclose(cached_arr, expected_normed, atol=1e-2)


def test_no_track_appears_in_two_splits(cfg):
    """Verify split integrity: no track appears in more than one split."""
    manifest_path = resolve_path(cfg.paths.manifest_path)
    if manifest_path.exists():
        m_df = pd.read_csv(manifest_path)
        assert m_df.groupby("track_id")["split"].nunique().max() == 1

    features_path = resolve_path(cfg.paths.features_path)
    if features_path.exists():
        f_df = pd.read_parquet(features_path)
        assert f_df.groupby("track_id")["split"].nunique().max() == 1


def test_features_parquet_schema_and_no_nan(cfg):
    """Verify features.parquet contains all 51 acoustic features without NaN or Inf."""
    features_path = resolve_path(cfg.paths.features_path)
    assert features_path.exists(), "features.parquet does not exist"

    df = pd.read_parquet(features_path)
    assert not df.empty

    # Metadata columns
    for col in ["track_id", "genre", "split"]:
        assert col in df.columns

    # All 51 acoustic features present
    for feat in FEATURE_NAMES:
        assert feat in df.columns, f"Missing feature: {feat}"

    # No NaN or Inf in numeric columns
    numeric_data = df[FEATURE_NAMES]
    assert not numeric_data.isna().any().any(), "Found NaN in features.parquet"
    assert not np.isinf(numeric_data.values).any(), "Found Inf in features.parquet"


def test_silent_clip_handling():
    """Verify that feature extraction on pure silence returns clean, finite values without failure."""
    sr = 22050
    silent_audio = np.zeros(sr * 30, dtype=np.float32)
    feats = extract_track_features(silent_audio, sr=sr)

    assert len(feats) == len(FEATURE_NAMES)
    for k, v in feats.items():
        assert isinstance(v, (int, float))
        assert not np.isnan(v)
        assert not np.isinf(v)
        assert v == 0.0


def test_audio_loading_and_segmentation(cfg):
    """Verify load_audio does not pad short audio and segment_audio splits into 3 padded windows."""
    sr = int(cfg.audio.sample_rate)

    # 1. Short audio (1 second): load_audio returns unpadded audio (sr samples)
    short_audio = np.ones(sr, dtype=np.float32)
    tmp_path = resolve_path("data/cache/temp_short.wav")
    import soundfile as sf
    sf.write(str(tmp_path), short_audio, sr)

    try:
        loaded = load_audio(tmp_path, sr=sr, duration=30.0)
        # load_audio must NOT pad; returns at most duration seconds (1 second in this case)
        assert loaded.shape == (sr,)
        assert loaded.dtype == np.float32
        assert np.isclose(loaded[0], 1.0, atol=1e-4)

        # Padding happens in the segmenting step per PLAN §07
        segments = segment_audio(loaded, sr=sr, window_sec=10.0, num_windows=3)
        assert len(segments) == 3
        for seg in segments:
            assert seg.shape == (int(sr * 10),)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

"""Unit tests for CNN training script: atomic logging, interruptions, resumption, and metadata.

All tests strictly isolate to pytest tmp_path fixtures and never touch production reports/ or models/.
"""

import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
import importlib

train_module = importlib.import_module("scripts.05_train_cnn")
train_cnn = train_module.train_cnn
compute_file_sha256 = train_module.compute_file_sha256
get_git_state = train_module.get_git_state



@pytest.fixture
def fake_dataset_env(tmp_path):
    """Create a minimal synthetic dataset environment in tmp_path."""
    env_dir = tmp_path / "test_env"
    env_dir.mkdir()

    processed_dir = env_dir / "data" / "processed"
    processed_dir.mkdir(parents=True)
    cache_dir = env_dir / "data" / "cache" / "logmels"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = env_dir / "models" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    metrics_dir = env_dir / "reports" / "metrics"
    metrics_dir.mkdir(parents=True)
    figures_dir = env_dir / "reports" / "figures"
    figures_dir.mkdir(parents=True)

    # 1. Label map (2 genres for fast test)
    label_map = {"Electronic": 0, "Rock": 1}
    label_map_path = processed_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f)

    # 2. Norm stats
    norm_stats = {"mean": 0.0, "std": 1.0}
    norm_stats_path = processed_dir / "norm_stats.json"
    with open(norm_stats_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f)

    # 3. Manifest (2 training tracks, 2 validation tracks)
    manifest_rows = [
        {"track_id": 1, "path": "data/raw/fma_small/000/000001.mp3", "genre": "Electronic", "split": "training", "artist_id": 10},
        {"track_id": 2, "path": "data/raw/fma_small/000/000002.mp3", "genre": "Rock", "split": "training", "artist_id": 11},
        {"track_id": 3, "path": "data/raw/fma_small/000/000003.mp3", "genre": "Electronic", "split": "validation", "artist_id": 12},
        {"track_id": 4, "path": "data/raw/fma_small/000/000004.mp3", "genre": "Rock", "split": "validation", "artist_id": 13},
    ]
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = processed_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # 4. Dummy cached spectrograms [128, 431]
    for tid in [1, 2, 3, 4]:
        for w in range(3):
            arr = np.random.randn(128, 431).astype(np.float32)
            np.save(cache_dir / f"{tid:06d}_{w}.npy", arr)

    # 5. Config YAML
    config_path = env_dir / "config.yaml"
    config_content = """seed: 42
device: cpu
paths:
  manifest_path: data/processed/manifest.csv
  label_map_path: data/processed/label_map.json
  norm_stats_path: data/processed/norm_stats.json
  spectrogram_cache_dir: data/cache/logmels
  cnn_checkpoint_path: models/checkpoints/genre_cnn.pt
  metrics_dir: reports/metrics
  figures_dir: reports/figures
audio:
  duration: 30.0
  window_duration: 10.0
  windows_per_track: 3
  n_mels: 128
  n_fft: 2048
  hop_length: 512
  power: 2.0
  mono: true
  sample_rate: 22050
cnn:
  architecture:
    in_channels: 1
    dropout: 0.1
  training:
    learning_rate: 0.001
    weight_decay: 0.01
    epochs: 2
    early_stopping_patience: 3
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    return {
        "env_dir": env_dir,
        "config_path": config_path,
        "manifest_path": manifest_path,
        "label_map_path": label_map_path,
        "norm_stats_path": norm_stats_path,
        "cache_dir": cache_dir,
        "checkpoints_dir": checkpoints_dir,
        "metrics_dir": metrics_dir,
        "figures_dir": figures_dir,
    }


def test_interrupted_run_persists_history_and_status(fake_dataset_env):
    """Verify that when training is interrupted by KeyboardInterrupt, history and status are saved."""
    env = fake_dataset_env
    epochs_seen = []

    # Simulate interrupt at start of epoch 2 (after epoch 1 completes 3 training batches)
    orig_step = torch.optim.AdamW.step
    step_count = 0

    def mock_step(self, *args, **kwargs):
        nonlocal step_count
        step_count += 1
        if step_count > 3:
            raise KeyboardInterrupt("Simulated user Ctrl+C")
        return orig_step(self, *args, **kwargs)

    with patch.object(torch.optim.AdamW, "step", mock_step):
        res = train_cnn(
            epochs_override=2,
            batch_size=2,
            config_path=str(env["config_path"]),
            checkpoint_dir_override=env["checkpoints_dir"],
            metrics_dir_override=env["metrics_dir"],
            figures_dir_override=env["figures_dir"],
            manifest_path_override=env["manifest_path"],
            cache_dir_override=env["cache_dir"],
            norm_stats_path_override=env["norm_stats_path"],
            label_map_path_override=env["label_map_path"],
        )

    assert res["status"] == "interrupted"

    # Verify history was saved atomically
    history_file = env["metrics_dir"] / "cnn_history.json"
    assert history_file.exists()
    with open(history_file, "r") as f:
        history = json.load(f)
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert "train_loss" in history[0]
    assert "val_macro_f1" in history[0]
    assert "learning_rate" in history[0]
    assert "seconds" in history[0]

    # Verify cnn_val.json recorded interrupted status
    val_file = env["metrics_dir"] / "cnn_val.json"
    assert val_file.exists()
    with open(val_file, "r") as f:
        val_data = json.load(f)
    assert val_data["status"] == "interrupted"

    # Verify run metadata recorded interrupted status
    meta_file = env["metrics_dir"] / "cnn_run_metadata.json"
    assert meta_file.exists()
    with open(meta_file, "r") as f:
        meta_data = json.load(f)
    assert meta_data["status"] == "interrupted"
    assert meta_data["epochs_run"] == 1

    # Verify last checkpoint was saved
    last_ckpt = env["checkpoints_dir"] / "genre_cnn_last.pt"
    assert last_ckpt.exists()


def test_resume_restores_epoch_counter_and_refuses_sha_mismatch(fake_dataset_env):
    """Verify --resume continues from the next epoch and validates artifact SHA-256 hashes."""
    env = fake_dataset_env

    # 1. Run 1 epoch cleanly
    res1 = train_cnn(
        epochs_override=1,
        batch_size=2,
        config_path=str(env["config_path"]),
        checkpoint_dir_override=env["checkpoints_dir"],
        metrics_dir_override=env["metrics_dir"],
        figures_dir_override=env["figures_dir"],
        manifest_path_override=env["manifest_path"],
        cache_dir_override=env["cache_dir"],
        norm_stats_path_override=env["norm_stats_path"],
        label_map_path_override=env["label_map_path"],
    )
    assert res1["status"] == "completed"
    assert res1["epochs_run"] == 1

    last_ckpt = env["checkpoints_dir"] / "genre_cnn_last.pt"
    assert last_ckpt.exists()

    # 2. Test SHA-256 mismatch refusal: mutate manifest
    manifest_orig_bytes = env["manifest_path"].read_bytes()
    with open(env["manifest_path"], "a") as f:
        f.write("\n999,data/raw/fake.mp3,Rock,training,99\n")

    with pytest.raises(ValueError, match="SHA-256 hash mismatch"):
        train_cnn(
            epochs_override=2,
            batch_size=2,
            resume=True,
            config_path=str(env["config_path"]),
            checkpoint_dir_override=env["checkpoints_dir"],
            metrics_dir_override=env["metrics_dir"],
            figures_dir_override=env["figures_dir"],
            manifest_path_override=env["manifest_path"],
            cache_dir_override=env["cache_dir"],
            norm_stats_path_override=env["norm_stats_path"],
            label_map_path_override=env["label_map_path"],
        )

    # Restore manifest
    env["manifest_path"].write_bytes(manifest_orig_bytes)

    # 3. Resume training up to epoch 2
    res2 = train_cnn(
        epochs_override=2,
        batch_size=2,
        resume=True,
        config_path=str(env["config_path"]),
        checkpoint_dir_override=env["checkpoints_dir"],
        metrics_dir_override=env["metrics_dir"],
        figures_dir_override=env["figures_dir"],
        manifest_path_override=env["manifest_path"],
        cache_dir_override=env["cache_dir"],
        norm_stats_path_override=env["norm_stats_path"],
        label_map_path_override=env["label_map_path"],
    )
    assert res2["status"] == "completed"
    assert res2["epochs_run"] == 2

    # Check history has both epochs
    with open(env["metrics_dir"] / "cnn_history.json", "r") as f:
        history = json.load(f)
    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2


def test_fresh_run_refuses_if_last_checkpoint_exists_unless_fresh_passed(fake_dataset_env):
    """Verify fresh run fails if genre_cnn_last.pt exists unless --fresh archives it."""
    env = fake_dataset_env

    # 1. Run 1 epoch
    train_cnn(
        epochs_override=1,
        batch_size=2,
        config_path=str(env["config_path"]),
        checkpoint_dir_override=env["checkpoints_dir"],
        metrics_dir_override=env["metrics_dir"],
        figures_dir_override=env["figures_dir"],
        manifest_path_override=env["manifest_path"],
        cache_dir_override=env["cache_dir"],
        norm_stats_path_override=env["norm_stats_path"],
        label_map_path_override=env["label_map_path"],
    )
    assert (env["checkpoints_dir"] / "genre_cnn_last.pt").exists()

    # 2. Starting fresh without --fresh or --resume must raise RuntimeError
    with pytest.raises(RuntimeError, match="Pass --resume to continue"):
        train_cnn(
            epochs_override=1,
            batch_size=2,
            resume=False,
            fresh=False,
            config_path=str(env["config_path"]),
            checkpoint_dir_override=env["checkpoints_dir"],
            metrics_dir_override=env["metrics_dir"],
            figures_dir_override=env["figures_dir"],
            manifest_path_override=env["manifest_path"],
            cache_dir_override=env["cache_dir"],
            norm_stats_path_override=env["norm_stats_path"],
            label_map_path_override=env["label_map_path"],
        )

    # 3. Starting with fresh=True archives previous artifacts and succeeds
    res_fresh = train_cnn(
        epochs_override=1,
        batch_size=2,
        resume=False,
        fresh=True,
        config_path=str(env["config_path"]),
        checkpoint_dir_override=env["checkpoints_dir"],
        metrics_dir_override=env["metrics_dir"],
        figures_dir_override=env["figures_dir"],
        manifest_path_override=env["manifest_path"],
        cache_dir_override=env["cache_dir"],
        norm_stats_path_override=env["norm_stats_path"],
        label_map_path_override=env["label_map_path"],
    )
    assert res_fresh["status"] == "completed"

    # Confirm archived previous checkpoint exists
    archived_checkpoints = list(env["checkpoints_dir"].glob("*prev*.pt")) + list(env["checkpoints_dir"].glob("pilot*.pt"))
    assert len(archived_checkpoints) > 0

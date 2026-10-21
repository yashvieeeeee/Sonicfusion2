"""Unit and integration tests for scripts/06_evaluate.py.

Verifies docs/PLAN.md Gate G8 requirements:
- Refuses to run if test_metrics.json already exists unless --force is passed.
- Refuses to run if git status --porcelain --untracked-files=no is not empty (dirty tracked files).
- Overwrite succeeds when --force is passed.
- Produces valid test_metrics.json, model_comparison.csv, model_comparison.md,
  misclassified_examples.csv, and all 5 visualization PNGs.
- All tests are isolated using pytest tmp_path fixtures and never touch production reports/ or models/.
"""

import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.dummy import DummyClassifier

import importlib
eval_module = importlib.import_module("scripts.06_evaluate")
check_git_status = eval_module.check_git_status
compute_bootstrap_and_statistical_tests = eval_module.compute_bootstrap_and_statistical_tests
compute_file_sha256 = eval_module.compute_file_sha256
evaluate_models = eval_module.evaluate_models
find_most_confused_pairs = eval_module.find_most_confused_pairs
from src.features.extraction import FEATURE_NAMES


@pytest.fixture
def eval_synthetic_env(tmp_path):
    """Create a self-contained synthetic test environment with minimal models and data."""
    root = tmp_path / "project_root"
    root.mkdir()

    # Directories
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True)
    cache_dir = root / "data" / "cache" / "logmels"
    cache_dir.mkdir(parents=True)
    checkpoints_dir = root / "models" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    baseline_dir = root / "models" / "baseline"
    baseline_dir.mkdir(parents=True)
    metrics_dir = root / "reports" / "metrics"
    metrics_dir.mkdir(parents=True)
    figures_dir = root / "reports" / "figures"
    figures_dir.mkdir(parents=True)
    configs_dir = root / "configs"
    configs_dir.mkdir(parents=True)

    # 1. Label map (4 classes)
    classes = ["Electronic", "Folk", "Pop", "Rock"]
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    label_map_path = processed_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f)

    # 2. Norm stats
    norm_stats = {"mean": -40.0, "std": 15.0}
    norm_stats_path = processed_dir / "norm_stats.json"
    with open(norm_stats_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f)

    # 3. Manifest with 4 test tracks (and dummy train/val)
    manifest_rows = [
        {"track_id": 1, "path": "data/raw/fma_small/000/000001.mp3", "genre": "Electronic", "split": "training", "artist_id": 1},
        {"track_id": 2, "path": "data/raw/fma_small/000/000002.mp3", "genre": "Rock", "split": "validation", "artist_id": 2},
        {"track_id": 101, "path": "data/raw/fma_small/000/000101.mp3", "genre": "Electronic", "split": "test", "artist_id": 10},
        {"track_id": 102, "path": "data/raw/fma_small/000/000102.mp3", "genre": "Folk", "split": "test", "artist_id": 11},
        {"track_id": 103, "path": "data/raw/fma_small/000/000103.mp3", "genre": "Pop", "split": "test", "artist_id": 12},
        {"track_id": 104, "path": "data/raw/fma_small/000/000104.mp3", "genre": "Rock", "split": "test", "artist_id": 13},
    ]
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = processed_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # 4. Features parquet (for all tracks)
    rng = np.random.default_rng(42)
    feat_data = {"track_id": [r["track_id"] for r in manifest_rows], "genre": [r["genre"] for r in manifest_rows], "split": [r["split"] for r in manifest_rows]}
    for feat in FEATURE_NAMES:
        feat_data[feat] = rng.normal(size=len(manifest_rows)).astype(np.float32)
    features_df = pd.DataFrame(feat_data)
    features_path = processed_dir / "features.parquet"
    features_df.to_parquet(features_path, index=False)

    # 5. Precached log-mel spectrograms for test tracks (3 windows each)
    for tid in [101, 102, 103, 104]:
        for w in range(3):
            dummy_spec = rng.normal(size=(128, 431)).astype(np.float16)
            np.save(cache_dir / f"{tid:06d}_{w}.npy", dummy_spec)

    # 6. Dummy Random Forest pipeline (joblib)
    import joblib
    dummy_clf = DummyClassifier(strategy="constant", constant="Electronic")
    dummy_clf.fit(features_df[FEATURE_NAMES], features_df["genre"])
    dummy_clf.classes_ = np.array(classes)
    rf_pipe = Pipeline([("classifier", dummy_clf)])
    rf_model_path = baseline_dir / "random_forest.joblib"
    joblib.dump(rf_pipe, rf_model_path)

    # 7. Dummy GenreCNN checkpoint
    from src.models.genre_cnn import GenreCNN
    mock_net = GenreCNN(num_classes=len(classes), in_channels=1, dropout=0.3)
    cnn_ckpt_path = checkpoints_dir / "genre_cnn.pt"
    torch.save(
        {
            "model_state_dict": mock_net.state_dict(),
            "class_names": classes,
            "sample_rate": 22050,
            "preprocessing_params": {
                "window_duration": 10.0,
                "windows_per_track": 3,
                "n_mels": 128,
                "n_fft": 2048,
                "hop_length": 512,
            },
            "normalization_stats": norm_stats,
            "version": "1.0",
        },
        cnn_ckpt_path,
    )

    # 8. Dummy config.yaml
    config_yaml = f"""seed: 42
device: cpu
dataset:
  min_track_seconds: 25.0
paths:
  data_root: {root}/data
  raw_dir: {root}/data/raw
  fma_small_dir: {root}/data/raw/fma_small
  fma_metadata_dir: {root}/data/raw/fma_metadata
  processed_dir: {root}/data/processed
  cache_dir: {root}/data/cache
  manifest_path: {root}/data/processed/manifest.csv
  features_path: {root}/data/processed/features.parquet
  label_map_path: {root}/data/processed/label_map.json
  norm_stats_path: {root}/data/processed/norm_stats.json
  spectrogram_cache_dir: {root}/data/cache/logmels
  models_dir: {root}/models
  baseline_model_path: {root}/models/baseline/random_forest.joblib
  cnn_checkpoint_path: {root}/models/checkpoints/genre_cnn.pt
  reports_dir: {root}/reports
  metrics_dir: {root}/reports/metrics
  figures_dir: {root}/reports/figures
audio:
  sample_rate: 22050
  duration: 30
  window_duration: 10
  windows_per_track: 3
  n_mels: 128
  n_fft: 2048
  hop_length: 512
  power: 2.0
  mono: true
"""
    config_path = configs_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_yaml)

    # 9. Exclusions file
    excl_path = metrics_dir / "data_exclusions.csv"
    excl_path.write_text("track_id,genre,split,file_size_bytes,decodable_seconds,reason\n999,Rock,training,100,0.5,corrupt\n", encoding="utf-8")

    # 10. CNN run metadata
    cnn_meta_path = metrics_dir / "cnn_run_metadata.json"
    cnn_meta_path.write_text(json.dumps({"status": "completed", "best_epoch": 12}), encoding="utf-8")

    return {
        "root": root,
        "config_path": str(config_path),
        "metrics_dir": metrics_dir,
        "figures_dir": figures_dir,
        "classes": classes,
    }


def test_refuse_when_metrics_file_already_exists(eval_synthetic_env):
    """Ensure evaluate_models refuses to run if test_metrics.json exists without --force."""
    metrics_dir = eval_synthetic_env["metrics_dir"]
    test_metrics_file = metrics_dir / "test_metrics.json"
    test_metrics_file.write_text("{}", encoding="utf-8")

    with patch.object(eval_module, "check_git_status", return_value=("mock_hash", False, [])):
        with pytest.raises(RuntimeError, match="already exists"):
            evaluate_models(
                config_path=eval_synthetic_env["config_path"],
                force=False,
                latency_tracks=0,
            )


def test_force_overwrites_existing_metrics(eval_synthetic_env):
    """Ensure evaluate_models succeeds when --force is passed even if test_metrics.json exists."""
    metrics_dir = eval_synthetic_env["metrics_dir"]
    test_metrics_file = metrics_dir / "test_metrics.json"
    test_metrics_file.write_text("{}", encoding="utf-8")

    with patch.object(eval_module, "check_git_status", return_value=("mock_hash", False, [])):
        res = evaluate_models(
            config_path=eval_synthetic_env["config_path"],
            force=True,
            latency_tracks=0,
        )
        assert "random_forest" in res
        assert "cnn" in res
        assert test_metrics_file.exists()


def test_refuse_when_git_working_tree_is_dirty(eval_synthetic_env):
    """Ensure evaluate_models refuses to run if git status reports dirty tracked files."""
    dirty_files = ["src/models/genre_cnn.py", "configs/config.yaml"]
    with patch.object(eval_module, "check_git_status", return_value=("mock_hash", True, dirty_files)):
        with pytest.raises(RuntimeError, match="Git working tree is dirty"):
            evaluate_models(
                config_path=eval_synthetic_env["config_path"],
                force=False,
                latency_tracks=0,
            )


def test_statistical_tests_and_confidence_intervals():
    """Verify bootstrap CI and McNemar statistical calculation."""
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1] * 10)
    y_pred_cnn = y_true.copy()
    y_pred_rf = y_true.copy()
    # Introduce some errors
    y_pred_cnn[:10] = (y_pred_cnn[:10] + 1) % 4
    y_pred_rf[:20] = (y_pred_rf[:20] + 1) % 4

    stats = compute_bootstrap_and_statistical_tests(
        y_true=y_true,
        y_pred_cnn=y_pred_cnn,
        y_pred_rf=y_pred_rf,
        n_resamples=100,
        seed=42,
    )

    assert "cnn" in stats
    assert "rf" in stats
    assert "accuracy_ci_95" in stats["cnn"]
    assert "macro_f1_ci_95" in stats["cnn"]
    assert len(stats["cnn"]["accuracy_ci_95"]) == 2
    assert stats["cnn"]["accuracy_ci_95"][0] <= stats["cnn"]["accuracy_ci_95"][1]

    assert "mcnemar_test" in stats
    assert "statistic" in stats["mcnemar_test"]
    assert "p_value" in stats["mcnemar_test"]
    assert 0.0 <= stats["mcnemar_test"]["p_value"] <= 1.0


def test_most_confused_pairs_detection():
    """Verify top confused pairs identification."""
    class_names = ["Electronic", "Folk", "Pop", "Rock"]
    y_true = np.array([0, 0, 0, 1, 1, 2, 3])
    y_pred = np.array([1, 1, 1, 0, 0, 2, 3])  # Electronic -> Folk (3 times), Folk -> Electronic (2 times)

    confused = find_most_confused_pairs(y_true, y_pred, class_names, top_k=2)
    assert len(confused) == 2
    assert confused[0]["true_class"] == "Electronic"
    assert confused[0]["predicted_class"] == "Folk"
    assert confused[0]["count"] == 3
    assert confused[1]["true_class"] == "Folk"
    assert confused[1]["predicted_class"] == "Electronic"
    assert confused[1]["count"] == 2


def test_full_evaluation_pipeline_in_temp_out_dir(eval_synthetic_env, tmp_path):
    """Test full evaluation pipeline writing into a dedicated temp output directory."""
    temp_out = tmp_path / "custom_out"
    temp_out.mkdir()

    with patch.object(eval_module, "check_git_status", return_value=("mock_commit_123", False, [])):
        res = evaluate_models(
            config_path=eval_synthetic_env["config_path"],
            force=False,
            limit=4,
            out_dir_override=temp_out,
            latency_tracks=0,
        )

        # Check returned payload
        assert res["git"]["commit_hash"] == "mock_commit_123"
        assert res["split_counts"]["training"] == 1
        assert res["split_counts"]["validation"] == 1
        assert res["split_counts"]["test"] == 4
        assert res["evaluated_test_tracks_count"] == 4
        assert res["exclusions_count"] == 1
        assert res["cnn_training_metadata"]["status"] == "completed"
        assert res["cnn_training_metadata"]["best_epoch"] == 12

        # Check generated files in temp_out
        metrics_dir = temp_out / "metrics"
        figures_dir = temp_out / "figures"

        assert (metrics_dir / "test_metrics.json").exists()
        assert (metrics_dir / "model_comparison.csv").exists()
        assert (metrics_dir / "model_comparison.md").exists()
        assert (metrics_dir / "misclassified_examples.csv").exists()

        assert (figures_dir / "rf_confusion_matrix_raw.png").exists()
        assert (figures_dir / "rf_confusion_matrix_norm.png").exists()
        assert (figures_dir / "cnn_confusion_matrix_raw.png").exists()
        assert (figures_dir / "cnn_confusion_matrix_norm.png").exists()
        assert (figures_dir / "f1_per_class_comparison.png").exists()

        # Check misclassified_examples.csv content
        misc_df = pd.read_csv(metrics_dir / "misclassified_examples.csv")
        assert "section" in misc_df.columns
        assert "true_label" in misc_df.columns
        assert "predicted_label" in misc_df.columns

        # Check model_comparison.csv content
        comp_df = pd.read_csv(metrics_dir / "model_comparison.csv")
        assert len(comp_df) == 2
        assert "Random Forest" in comp_df["model"].values
        assert "2D CNN" in comp_df["model"].values

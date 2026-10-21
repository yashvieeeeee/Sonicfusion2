"""Evaluate frozen Random Forest and CNN models on the official FMA test split.

Follows docs/PLAN.md (§08, §09, §16, Gate G8):
- Evaluates frozen Random Forest and CNN on the official TEST split exactly once.
- Refuses to run if reports/metrics/test_metrics.json already exists unless --force is passed.
- Refuses to run if git status --porcelain --untracked-files=no is not empty (prints dirty files).
- Records git commit hash, UTC timestamp, package versions, and hardware device.
- Uses identical 800 test track IDs from manifest.csv, same label map, and track-level predictions.
- CNN uses shared window aggregation from src/models/genre_cnn (three windows per track, averaged).
- Random Forest uses features from features.parquet and pipeline in models/baseline.
- Computes 95% bootstrap confidence intervals (1000 resamples, fixed seed) for accuracy & macro-F1.
- Computes paired bootstrap and McNemar test on CNN-vs-RF accuracy difference (no winner declared in code).
- Generates raw and row-normalized confusion matrices for each model (4 PNGs) + per-class F1 bar chart.
- Generates model_comparison.csv and model_comparison.md tables.
- Benchmarks latency over N=50 test tracks (fixed seed) for decode, feature extraction, CNN forward, RF inference.
- Generates misclassified_examples.csv with 5 CNN-misclassified tracks and top 3 confused class pairs for each model.
- Supports --limit N and --out-dir for safe testing without overwriting production metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import torch
import torch.nn.functional as F

from src.audio.loading import load_audio, segment_audio
from src.audio.spectrogram import (
    compute_logmel_spectrogram,
    normalize_spectrogram,
    to_cnn_tensor,
)
from src.config import get_project_root, load_config, resolve_device, resolve_path
from src.features.extraction import FEATURE_NAMES, extract_track_features
from src.models.baseline import load_baseline_model
from src.models.genre_cnn import GenreCNN, aggregate_window_probabilities


def compute_file_sha256(filepath: Path) -> str:
    """Compute hex SHA-256 digest of a file; returns empty string if not found."""
    if not filepath.exists() or not filepath.is_file():
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_git_status(project_root: Path) -> Tuple[str, bool, List[str]]:
    """Retrieve git commit hash and verify tracked working tree cleanliness.

    Refuses if `git status --porcelain --untracked-files=no` is not empty.

    Returns:
        commit_hash: Current git HEAD commit hash (or 'unknown').
        is_dirty: True if any tracked files have uncommitted changes.
        dirty_files: List of dirty tracked files reported by git.
    """
    commit_hash = "unknown"
    dirty_files: List[str] = []
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            commit_hash = res.stdout.strip()

        res_dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res_dirty.returncode == 0 and res_dirty.stdout.strip():
            dirty_files = [
                line.strip()
                for line in res_dirty.stdout.strip().splitlines()
                if line.strip()
            ]
    except Exception:
        pass

    is_dirty = len(dirty_files) > 0
    return commit_hash, is_dirty, dirty_files


def get_hardware_info(device_str: str) -> Dict[str, str]:
    """Gather hardware and environment descriptor strings."""
    hw_info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "device": device_str,
    }
    if device_str.startswith("cuda") and torch.cuda.is_available():
        hw_info["cuda_device_name"] = torch.cuda.get_device_name(0)
    else:
        hw_info["device_name"] = platform.processor() or platform.machine()
    return hw_info


def get_package_versions() -> Dict[str, str]:
    """Capture library versions for reproducibility."""
    import sklearn
    import torchaudio
    import librosa
    import scipy

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "librosa": librosa.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }


def find_most_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Identify the top-K most frequent off-diagonal misclassifications."""
    pair_counts: Dict[Tuple[int, int], int] = {}
    for yt, yp in zip(y_true, y_pred):
        if yt != yp:
            pair = (int(yt), int(yp))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "true_class": class_names[true_idx],
            "predicted_class": class_names[pred_idx],
            "count": count,
            "description": f"{class_names[true_idx]} -> {class_names[pred_idx]}",
        }
        for (true_idx, pred_idx), count in sorted_pairs
    ]


def compute_bootstrap_and_statistical_tests(
    y_true: np.ndarray,
    y_pred_cnn: np.ndarray,
    y_pred_rf: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute 95% bootstrap CIs and paired significance tests between CNN and RF.

    Reports numerical test statistics and confidence intervals without declaring a winner.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    cnn_acc_boot: List[float] = []
    cnn_f1_boot: List[float] = []
    rf_acc_boot: List[float] = []
    rf_f1_boot: List[float] = []
    diff_acc_boot: List[float] = []  # CNN accuracy - RF accuracy

    for _ in range(n_resamples):
        idx = rng.integers(0, n_samples, size=n_samples)
        b_true = y_true[idx]
        b_cnn = y_pred_cnn[idx]
        b_rf = y_pred_rf[idx]

        acc_c = float(accuracy_score(b_true, b_cnn))
        f1_c = float(f1_score(b_true, b_cnn, average="macro", zero_division=0))
        acc_r = float(accuracy_score(b_true, b_rf))
        f1_r = float(f1_score(b_true, b_rf, average="macro", zero_division=0))

        cnn_acc_boot.append(acc_c)
        cnn_f1_boot.append(f1_c)
        rf_acc_boot.append(acc_r)
        rf_f1_boot.append(f1_r)
        diff_acc_boot.append(acc_c - acc_r)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    def get_ci(arr: List[float]) -> List[float]:
        return [
            float(round(np.percentile(arr, lower_pct), 4)),
            float(round(np.percentile(arr, upper_pct), 4)),
        ]

    # Paired McNemar Test
    # Discordance table between paired 0/1 indicator predictions
    cnn_correct = (y_pred_cnn == y_true)
    rf_correct = (y_pred_rf == y_true)

    n00 = int(np.sum(~cnn_correct & ~rf_correct))
    n01 = int(np.sum(~cnn_correct & rf_correct))  # Discordant: RF correct only
    n10 = int(np.sum(cnn_correct & ~rf_correct))  # Discordant: CNN correct only
    n11 = int(np.sum(cnn_correct & rf_correct))   # Both correct

    b = n10
    c = n01
    discordant = b + c
    if discordant > 0:
        # Edwards continuity correction
        mcnemar_stat = float(((abs(b - c) - 1.0) ** 2) / discordant)
        mcnemar_p_value = float(1.0 - chi2.cdf(mcnemar_stat, df=1))
        exact_p_value = float(binomtest(min(b, c), discordant, 0.5).pvalue)
    else:
        mcnemar_stat = 0.0
        mcnemar_p_value = 1.0
        exact_p_value = 1.0

    diff_arr = np.array(diff_acc_boot)
    paired_boot_p_value = float(
        2.0 * min(np.mean(diff_arr <= 0.0), np.mean(diff_arr >= 0.0))
    )

    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "confidence_level": 0.95,
        "cnn": {
            "accuracy_ci_95": get_ci(cnn_acc_boot),
            "macro_f1_ci_95": get_ci(cnn_f1_boot),
        },
        "rf": {
            "accuracy_ci_95": get_ci(rf_acc_boot),
            "macro_f1_ci_95": get_ci(rf_f1_boot),
        },
        "difference_cnn_minus_rf": {
            "accuracy_difference_ci_95": get_ci(diff_acc_boot),
            "mean_accuracy_difference": float(round(float(np.mean(diff_arr)), 4)),
            "paired_bootstrap_p_value": float(round(paired_boot_p_value, 5)),
        },
        "mcnemar_test": {
            "contingency_table": {
                "both_correct": n11,
                "cnn_correct_rf_incorrect": n10,
                "rf_correct_cnn_incorrect": n01,
                "both_incorrect": n00,
            },
            "statistic": float(round(mcnemar_stat, 4)),
            "p_value": float(round(mcnemar_p_value, 5)),
            "exact_binomial_p_value": float(round(exact_p_value, 5)),
        },
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str,
    normalize: bool = False,
    cmap: str = "Blues",
) -> None:
    """Plot and save confusion matrix PNG."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    plot_cm = cm.astype(float)
    if normalize:
        row_sums = plot_cm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        plot_cm = plot_cm / row_sums

    im = ax.imshow(plot_cm, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if normalize:
        cbar.set_label("Row Proportion", rotation=270, labelpad=15)
    else:
        cbar.set_label("Sample Count", rotation=270, labelpad=15)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Genre",
        xlabel="Predicted Genre",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate numbers in cells
    thresh = plot_cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = plot_cm[i, j]
            text_str = f"{val:.2f}" if normalize else f"{int(val)}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text_str, ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1_comparison(
    rf_report: Dict[str, Any],
    cnn_report: Dict[str, Any],
    class_names: List[str],
    output_path: Path,
) -> None:
    """Plot grouped bar chart comparing per-class F1 between Random Forest and 2D CNN."""
    rf_f1s = [float(rf_report.get(c, {}).get("f1-score", 0.0)) for c in class_names]
    cnn_f1s = [float(cnn_report.get(c, {}).get("f1-score", 0.0)) for c in class_names]

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    rects1 = ax.bar(x - width / 2, rf_f1s, width, label="Random Forest Baseline", color="#3b82f6", alpha=0.9)
    rects2 = ax.bar(x + width / 2, cnn_f1s, width, label="2D CNN (Log-Mel)", color="#8b5cf6", alpha=0.9)

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Test Set Genre Classification: Per-Class F1 Score Comparison", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    # Add data labels above bars
    def autolabel(rects: Any) -> None:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def measure_latencies(
    test_df: pd.DataFrame,
    rf_pipeline: Any,
    cnn_model: GenreCNN,
    device: torch.device,
    mean: float,
    std: float,
    sr: int = 22050,
    duration: float = 30.0,
    win_sec: float = 10.0,
    num_windows: int = 3,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_benchmark: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """Benchmark median inference latencies over N test tracks with a fixed seed."""
    rng = random.Random(seed)
    track_pool = test_df.to_dict(orient="records")
    if len(track_pool) > n_benchmark:
        sample_tracks = rng.sample(track_pool, n_benchmark)
    else:
        sample_tracks = track_pool

    decode_times_ms: List[float] = []
    feature_times_ms: List[float] = []
    rf_times_ms: List[float] = []
    cnn_forward_times_ms: List[float] = []

    for item in sample_tracks:
        full_path = resolve_path(str(item["path"]))
        if not full_path.exists():
            continue

        # 1. Decode latency
        t0 = time.perf_counter()
        try:
            y = load_audio(full_path, sr=sr, duration=duration, mono=True)
        except Exception:
            continue
        decode_times_ms.append((time.perf_counter() - t0) * 1000.0)

        # 2. Classical feature extraction latency
        t0 = time.perf_counter()
        try:
            feats = extract_track_features(y, sr=sr)
            feat_df = pd.DataFrame([feats])[FEATURE_NAMES]
            feature_times_ms.append((time.perf_counter() - t0) * 1000.0)

            # 3. RF inference latency
            t0 = time.perf_counter()
            _ = rf_pipeline.predict(feat_df)
            rf_times_ms.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            pass

        # 4. CNN forward pass latency
        try:
            segments = segment_audio(y, sr=sr, window_sec=win_sec, num_windows=num_windows)
            window_tensors = []
            for seg in segments:
                raw_logmel = compute_logmel_spectrogram(
                    seg, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
                )
                normed = normalize_spectrogram(raw_logmel, mean=mean, std=std)
                window_tensors.append(to_cnn_tensor(normed))
            batch = torch.stack(window_tensors, dim=0).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                _ = cnn_model(batch)
            cnn_forward_times_ms.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            pass

    return {
        "n_samples_measured": len(decode_times_ms),
        "median_decode_ms": float(round(float(np.median(decode_times_ms)), 2)) if decode_times_ms else 0.0,
        "median_feature_extraction_ms": float(round(float(np.median(feature_times_ms)), 2)) if feature_times_ms else 0.0,
        "median_rf_inference_ms": float(round(float(np.median(rf_times_ms)), 2)) if rf_times_ms else 0.0,
        "median_cnn_forward_ms": float(round(float(np.median(cnn_forward_times_ms)), 2)) if cnn_forward_times_ms else 0.0,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI options for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate frozen baseline and CNN models on official FMA test split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force evaluation even if test_metrics.json exists or working tree is dirty.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test tracks for verification/smoke runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional base output directory override (for testing without modifying production reports/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda', 'auto'). Defaults to config.",
    )
    parser.add_argument(
        "--latency-tracks",
        type=int,
        default=50,
        help="Number of test tracks to use for latency benchmark (default: 50).",
    )
    return parser.parse_args()


def evaluate_models(
    config_path: Optional[str] = None,
    force: bool = False,
    limit: Optional[int] = None,
    out_dir_override: Optional[str | Path] = None,
    device_override: Optional[str] = None,
    latency_tracks: int = 50,
) -> Dict[str, Any]:
    """Execute official held-out test split evaluation for Random Forest and CNN."""
    cfg = load_config(config_path)
    root = get_project_root()

    # Determine output directories
    if out_dir_override:
        base_out = Path(out_dir_override).resolve()
        metrics_dir = base_out / "metrics"
        figures_dir = base_out / "figures"
    else:
        metrics_dir = resolve_path(cfg.paths.metrics_dir)
        figures_dir = resolve_path(cfg.paths.figures_dir)

    metrics_file = metrics_dir / "test_metrics.json"

    # Pre-flight Check 1: Refuse if test_metrics.json already exists
    if metrics_file.exists() and not force:
        raise RuntimeError(
            f"Refusing to run: {metrics_file} already exists. "
            "Evaluation on the official test split runs exactly once. "
            "Use --force to override."
        )

    # Pre-flight Check 2: Refuse if tracked files in git working tree are dirty
    commit_hash, is_dirty, dirty_files = check_git_status(root)
    if is_dirty and not force:
        print("\n[ERROR] Refusing to run evaluation: tracked files in git working tree are dirty:")
        for f in dirty_files:
            print(f"  {f}")
        raise RuntimeError(
            f"Git working tree is dirty ({len(dirty_files)} modified tracked files). "
            "Commit or stash changes before running evaluation, or pass --force."
        )

    # Resolve required artifact paths
    manifest_path = resolve_path(cfg.paths.manifest_path)
    features_path = resolve_path(cfg.paths.features_path)
    label_map_path = resolve_path(cfg.paths.label_map_path)
    norm_stats_path = resolve_path(cfg.paths.norm_stats_path)
    rf_model_path = resolve_path(cfg.paths.baseline_model_path)
    cnn_ckpt_path = resolve_path(cfg.paths.cnn_checkpoint_path)
    exclusions_path = resolve_path(cfg.paths.metrics_dir) / "data_exclusions.csv"
    cnn_meta_path = resolve_path(cfg.paths.metrics_dir) / "cnn_run_metadata.json"
    cfg_file_path = resolve_path("configs/config.yaml")

    for req_file, name in [
        (manifest_path, "Manifest CSV"),
        (features_path, "Features Parquet"),
        (label_map_path, "Label Map JSON"),
        (norm_stats_path, "Norm Stats JSON"),
        (rf_model_path, "Random Forest baseline model"),
        (cnn_ckpt_path, "CNN checkpoint"),
    ]:
        if not req_file.exists():
            raise FileNotFoundError(f"Required artifact {name} not found at: {req_file}")

    # Compute SHA-256 digests
    sha256_checksums = {
        "cnn_checkpoint": compute_file_sha256(cnn_ckpt_path),
        "rf_joblib": compute_file_sha256(rf_model_path),
        "manifest_csv": compute_file_sha256(manifest_path),
        "config_yaml": compute_file_sha256(cfg_file_path),
        "label_map_json": compute_file_sha256(label_map_path),
        "norm_stats_json": compute_file_sha256(norm_stats_path),
    }

    # Load label map
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_to_idx: Dict[str, int] = json.load(f)
    class_names: List[str] = sorted(label_to_idx.keys(), key=lambda k: label_to_idx[k])
    num_classes = len(class_names)

    # Load manifest and split counts
    manifest_df = pd.read_csv(manifest_path)
    split_counts = {
        "training": int((manifest_df["split"] == "training").sum()),
        "validation": int((manifest_df["split"] == "validation").sum()),
        "test": int((manifest_df["split"] == "test").sum()),
    }

    # Exclusions count
    exclusions_count = 0
    if exclusions_path.exists():
        try:
            excl_df = pd.read_csv(exclusions_path)
            exclusions_count = len(excl_df)
        except Exception:
            pass

    # Read CNN run status & best epoch from cnn_run_metadata.json (with fallback to archived metadata)
    cnn_run_status = "unknown"
    cnn_best_epoch = 1
    if cnn_meta_path.exists():
        try:
            with open(cnn_meta_path, "r", encoding="utf-8") as f:
                c_meta = json.load(f)
            cnn_run_status = str(c_meta.get("status", "completed"))
            cnn_best_epoch = int(c_meta.get("best_epoch", 1))
        except Exception:
            pass
    else:
        # Check for previous metadata archives in metrics dir
        prev_metas = sorted(metrics_dir.glob("cnn_run_metadata_prev_*.json"), reverse=True)
        if not prev_metas:
            # Check default production metrics dir
            prod_metrics_dir = resolve_path(cfg.paths.metrics_dir)
            prev_metas = sorted(prod_metrics_dir.glob("cnn_run_metadata_prev_*.json"), reverse=True)
        if prev_metas:
            try:
                with open(prev_metas[0], "r", encoding="utf-8") as f:
                    c_meta = json.load(f)
                cnn_run_status = str(c_meta.get("status", "unknown"))
                cnn_best_epoch = int(c_meta.get("best_epoch", 1))
            except Exception:
                pass

    # Filter to TEST split
    test_df = manifest_df[manifest_df["split"] == "test"].copy()
    if limit is not None and limit > 0:
        test_df = test_df.iloc[:limit].copy()
        print(f"[TEST LIMIT] Evaluating on first {len(test_df)} test tracks (--limit {limit}).")
    else:
        print(f"Evaluating on all {len(test_df)} held-out test tracks.")

    test_track_ids = [int(tid) for tid in test_df["track_id"]]
    y_true_labels = [str(g) for g in test_df["genre"]]
    y_true_indices = np.array([label_to_idx[g] for g in y_true_labels], dtype=int)

    # 1. Random Forest Evaluation
    print("Running Random Forest baseline inference on test tracks...")
    rf_pipeline = load_baseline_model(rf_model_path)
    features_df = pd.read_parquet(features_path)

    # Ensure track_id alignment
    if "track_id" in features_df.columns:
        features_df = features_df.set_index("track_id")
    test_features = features_df.loc[test_track_ids]
    X_test_rf = test_features[FEATURE_NAMES]

    rf_pred_labels = rf_pipeline.predict(X_test_rf)
    rf_pred_indices = np.array([label_to_idx[lbl] for lbl in rf_pred_labels], dtype=int)
    rf_pred_probs = rf_pipeline.predict_proba(X_test_rf)

    # Align RF prob columns to class_names order if needed
    if hasattr(rf_pipeline, "classes_"):
        rf_class_order = list(rf_pipeline.classes_)
        rf_aligned_probs = np.zeros((len(test_track_ids), num_classes), dtype=float)
        for c_idx, c_name in enumerate(class_names):
            if c_name in rf_class_order:
                src_col = rf_class_order.index(c_name)
                rf_aligned_probs[:, c_idx] = rf_pred_probs[:, src_col]
        rf_pred_probs = rf_aligned_probs

    rf_accuracy = float(accuracy_score(y_true_indices, rf_pred_indices))
    rf_macro_f1 = float(f1_score(y_true_indices, rf_pred_indices, average="macro", zero_division=0))
    rf_clf_report = classification_report(
        y_true_indices,
        rf_pred_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # 2. CNN Evaluation
    device_str = device_override or resolve_device(cfg)
    device = torch.device(device_str)
    print(f"Running 2D CNN inference on test tracks (device={device_str})...")

    checkpoint = torch.load(cnn_ckpt_path, map_location=device, weights_only=False)
    cnn_model = GenreCNN(num_classes=num_classes, in_channels=1, dropout=0.3)
    state_dict = checkpoint.get("state_dict") if "state_dict" in checkpoint else checkpoint["model_state_dict"]
    cnn_model.load_state_dict(state_dict)
    cnn_model.to(device)
    cnn_model.eval()

    # Preprocessing & normalization configuration
    prep_params = checkpoint.get("preprocessing_params") or checkpoint.get("preprocessing params") or {}
    norm_stats = checkpoint.get("normalization_stats") or checkpoint.get("normalization stats") or {}
    norm_mean = float(norm_stats.get("mean", 0.0))
    norm_std = float(norm_stats.get("std", 1.0))
    if norm_mean == 0.0 and norm_stats_path.exists():
        try:
            with open(norm_stats_path, "r", encoding="utf-8") as f:
                ns = json.load(f)
            norm_mean = float(ns.get("mean", 0.0))
            norm_std = float(ns.get("std", 1.0))
        except Exception:
            pass

    sr = int(prep_params.get("sample_rate", cfg.audio.sample_rate))
    duration = float(prep_params.get("duration", cfg.audio.duration))
    win_sec = float(prep_params.get("window_duration", cfg.audio.window_duration))
    num_windows = int(prep_params.get("windows_per_track", cfg.audio.windows_per_track))
    n_mels = int(prep_params.get("n_mels", cfg.audio.n_mels))
    n_fft = int(prep_params.get("n_fft", cfg.audio.n_fft))
    hop_length = int(prep_params.get("hop_length", cfg.audio.hop_length))
    cache_dir = resolve_path(cfg.paths.spectrogram_cache_dir)

    cnn_pred_indices: List[int] = []
    cnn_track_probs: List[np.ndarray] = []

    for _, row in test_df.iterrows():
        t_id = int(row["track_id"])
        rel_path = str(row["path"])
        win_tensors: List[torch.Tensor] = []

        for w_idx in range(num_windows):
            cache_file = cache_dir / f"{t_id:06d}_{w_idx}.npy"
            if cache_file.exists():
                spec_arr = np.load(cache_file).astype(np.float32)
            else:
                full_audio = resolve_path(rel_path)
                y = load_audio(full_audio, sr=sr, duration=duration, mono=True)
                segments = segment_audio(y, sr=sr, window_sec=win_sec, num_windows=num_windows)
                raw_logmel = compute_logmel_spectrogram(
                    segments[w_idx], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
                )
                spec_arr = normalize_spectrogram(raw_logmel, mean=norm_mean, std=norm_std)

            win_tensors.append(to_cnn_tensor(spec_arr))

        batch = torch.stack(win_tensors, dim=0).to(device)
        with torch.no_grad():
            logits = cnn_model(batch)
            window_probs = F.softmax(logits, dim=-1)
            # Shared window aggregation code from src/models/genre_cnn
            avg_prob = aggregate_window_probabilities(window_probs)

        pred_class_idx = int(np.argmax(avg_prob))
        cnn_pred_indices.append(pred_class_idx)
        cnn_track_probs.append(avg_prob)

    cnn_pred_arr = np.array(cnn_pred_indices, dtype=int)
    cnn_track_probs_arr = np.array(cnn_track_probs, dtype=float)

    cnn_accuracy = float(accuracy_score(y_true_indices, cnn_pred_arr))
    cnn_macro_f1 = float(f1_score(y_true_indices, cnn_pred_arr, average="macro", zero_division=0))
    cnn_clf_report = classification_report(
        y_true_indices,
        cnn_pred_arr,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # 3. Statistical Analysis & Confidence Intervals
    print("Computing 1,000 bootstrap resamples and paired significance tests...")
    stat_results = compute_bootstrap_and_statistical_tests(
        y_true=y_true_indices,
        y_pred_cnn=cnn_pred_arr,
        y_pred_rf=rf_pred_indices,
        n_resamples=1000,
        seed=int(cfg.seed),
    )

    # 4. Latency Measurements
    print(f"Measuring latency over N={min(latency_tracks, len(test_df))} tracks...")
    latency_results = measure_latencies(
        test_df=test_df,
        rf_pipeline=rf_pipeline,
        cnn_model=cnn_model,
        device=device,
        mean=norm_mean,
        std=norm_std,
        sr=sr,
        duration=duration,
        win_sec=win_sec,
        num_windows=num_windows,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        n_benchmark=latency_tracks,
        seed=int(cfg.seed),
    )

    # 5. Most Confused Class Pairs
    rf_confused_pairs = find_most_confused_pairs(y_true_indices, rf_pred_indices, class_names, top_k=3)
    cnn_confused_pairs = find_most_confused_pairs(y_true_indices, cnn_pred_arr, class_names, top_k=3)

    # 6. Misclassified Examples for Error Analysis
    misclassified_examples: List[Dict[str, Any]] = []
    for idx, (yt, yp, probs, tid) in enumerate(
        zip(y_true_indices, cnn_pred_arr, cnn_track_probs_arr, test_track_ids)
    ):
        if yt != yp:
            top3_idx = np.argsort(probs)[::-1][:3]
            top3_scores = [
                {"label": class_names[i], "score": float(round(probs[i], 4))}
                for i in top3_idx
            ]
            misclassified_examples.append(
                {
                    "track_id": int(tid),
                    "true_label": class_names[yt],
                    "predicted_label": class_names[yp],
                    "top_3_scores": top3_scores,
                    "rank1_label": top3_scores[0]["label"],
                    "rank1_score": top3_scores[0]["score"],
                    "rank2_label": top3_scores[1]["label"] if len(top3_scores) > 1 else "",
                    "rank2_score": top3_scores[1]["score"] if len(top3_scores) > 1 else 0.0,
                    "rank3_label": top3_scores[2]["label"] if len(top3_scores) > 2 else "",
                    "rank3_score": top3_scores[2]["score"] if len(top3_scores) > 2 else 0.0,
                }
            )
        if len(misclassified_examples) >= 5:
            break

    # 7. Generate Figures
    figures_dir.mkdir(parents=True, exist_ok=True)
    rf_cm = confusion_matrix(y_true_indices, rf_pred_indices, labels=np.arange(num_classes))
    cnn_cm = confusion_matrix(y_true_indices, cnn_pred_arr, labels=np.arange(num_classes))

    # Confusion matrix plots
    plot_confusion_matrix(
        rf_cm,
        class_names,
        figures_dir / "rf_confusion_matrix_raw.png",
        title="Random Forest: Test Confusion Matrix (Raw Counts)",
        normalize=False,
    )
    plot_confusion_matrix(
        rf_cm,
        class_names,
        figures_dir / "rf_confusion_matrix_norm.png",
        title="Random Forest: Test Confusion Matrix (Row-Normalized)",
        normalize=True,
    )
    plot_confusion_matrix(
        cnn_cm,
        class_names,
        figures_dir / "cnn_confusion_matrix_raw.png",
        title="2D CNN: Test Confusion Matrix (Raw Counts)",
        normalize=False,
    )
    plot_confusion_matrix(
        cnn_cm,
        class_names,
        figures_dir / "cnn_confusion_matrix_norm.png",
        title="2D CNN: Test Confusion Matrix (Row-Normalized)",
        normalize=True,
    )

    # Per-class F1 comparison chart
    plot_per_class_f1_comparison(
        rf_clf_report,
        cnn_clf_report,
        class_names,
        figures_dir / "f1_per_class_comparison.png",
    )

    # 8. Export Model Comparison Table (.csv and .md)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv_path = metrics_dir / "model_comparison.csv"
    comparison_md_path = metrics_dir / "model_comparison.md"

    rf_acc_ci = stat_results["rf"]["accuracy_ci_95"]
    rf_f1_ci = stat_results["rf"]["macro_f1_ci_95"]
    cnn_acc_ci = stat_results["cnn"]["accuracy_ci_95"]
    cnn_f1_ci = stat_results["cnn"]["macro_f1_ci_95"]

    rf_latency_ms = latency_results["median_rf_inference_ms"]
    cnn_latency_ms = latency_results["median_cnn_forward_ms"]

    comp_rows = [
        {
            "model": "Random Forest",
            "test_accuracy": round(rf_accuracy, 4),
            "accuracy_ci_95": f"[{rf_acc_ci[0]:.4f}, {rf_acc_ci[1]:.4f}]",
            "macro_f1": round(rf_macro_f1, 4),
            "macro_f1_ci_95": f"[{rf_f1_ci[0]:.4f}, {rf_f1_ci[1]:.4f}]",
            "median_latency_ms": rf_latency_ms,
        },
        {
            "model": "2D CNN",
            "test_accuracy": round(cnn_accuracy, 4),
            "accuracy_ci_95": f"[{cnn_acc_ci[0]:.4f}, {cnn_acc_ci[1]:.4f}]",
            "macro_f1": round(cnn_macro_f1, 4),
            "macro_f1_ci_95": f"[{cnn_f1_ci[0]:.4f}, {cnn_f1_ci[1]:.4f}]",
            "median_latency_ms": cnn_latency_ms,
        },
    ]
    pd.DataFrame(comp_rows).to_csv(comparison_csv_path, index=False)

    md_content = f"""# SonicFusion 2: Test Split Model Comparison

Evaluated on official held-out test split ({len(test_df)} tracks).

| Model | Test Accuracy (95% CI) | Macro-F1 (95% CI) | Median Inference Latency |
|---|---|---|---|
| Random Forest | {rf_accuracy:.4f} [{rf_acc_ci[0]:.4f}, {rf_acc_ci[1]:.4f}] | {rf_macro_f1:.4f} [{rf_f1_ci[0]:.4f}, {rf_f1_ci[1]:.4f}] | {rf_latency_ms:.2f} ms |
| 2D CNN | {cnn_accuracy:.4f} [{cnn_acc_ci[0]:.4f}, {cnn_acc_ci[1]:.4f}] | {cnn_macro_f1:.4f} [{cnn_f1_ci[0]:.4f}, {cnn_f1_ci[1]:.4f}] | {cnn_latency_ms:.2f} ms |

### Paired Statistical Comparison (Accuracy Difference: CNN - RF)
- 95% Bootstrap CI on Accuracy Difference: [{stat_results['difference_cnn_minus_rf']['accuracy_difference_ci_95'][0]:.4f}, {stat_results['difference_cnn_minus_rf']['accuracy_difference_ci_95'][1]:.4f}]
- Mean Accuracy Difference: {stat_results['difference_cnn_minus_rf']['mean_accuracy_difference']:+.4f}
- Paired Bootstrap p-value: {stat_results['difference_cnn_minus_rf']['paired_bootstrap_p_value']:.4f}
- McNemar Chi-Square Statistic: {stat_results['mcnemar_test']['statistic']:.4f} (p-value: {stat_results['mcnemar_test']['p_value']:.4f}, exact binomial p-value: {stat_results['mcnemar_test']['exact_binomial_p_value']:.4f})
"""
    with open(comparison_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 9. Export Error Analysis CSV
    misclassified_csv_path = metrics_dir / "misclassified_examples.csv"
    error_analysis_rows: List[Dict[str, Any]] = []

    for ex in misclassified_examples:
        error_analysis_rows.append(
            {
                "section": "cnn_misclassified_example",
                "track_id_or_model": ex["track_id"],
                "true_label": ex["true_label"],
                "predicted_label": ex["predicted_label"],
                "rank1_label": ex["rank1_label"],
                "rank1_score": ex["rank1_score"],
                "rank2_label": ex["rank2_label"],
                "rank2_score": ex["rank2_score"],
                "rank3_label": ex["rank3_label"],
                "rank3_score": ex["rank3_score"],
                "confusion_count": "",
            }
        )

    for p in cnn_confused_pairs:
        error_analysis_rows.append(
            {
                "section": "cnn_most_confused_pair",
                "track_id_or_model": "2D CNN",
                "true_label": p["true_class"],
                "predicted_label": p["predicted_class"],
                "rank1_label": "",
                "rank1_score": "",
                "rank2_label": "",
                "rank2_score": "",
                "rank3_label": "",
                "rank3_score": "",
                "confusion_count": p["count"],
            }
        )

    for p in rf_confused_pairs:
        error_analysis_rows.append(
            {
                "section": "rf_most_confused_pair",
                "track_id_or_model": "Random Forest",
                "true_label": p["true_class"],
                "predicted_label": p["predicted_class"],
                "rank1_label": "",
                "rank1_score": "",
                "rank2_label": "",
                "rank2_score": "",
                "rank3_label": "",
                "rank3_score": "",
                "confusion_count": p["count"],
            }
        )

    pd.DataFrame(error_analysis_rows).to_csv(misclassified_csv_path, index=False)

    # 10. Build Full Test Metrics Payload
    now_utc = datetime.now(timezone.utc).isoformat()
    hw_info = get_hardware_info(device_str)
    pkg_versions = get_package_versions()

    test_metrics_payload: Dict[str, Any] = {
        "timestamp": now_utc,
        "note": "Model selection used validation only. Evaluated on official FMA test split exactly once.",
        "git": {
            "commit_hash": commit_hash,
            "is_dirty": is_dirty,
            "dirty_files": dirty_files,
        },
        "hardware": hw_info,
        "package_versions": pkg_versions,
        "sha256_checksums": sha256_checksums,
        "split_counts": split_counts,
        "evaluated_test_tracks_count": len(test_df),
        "exclusions_count": exclusions_count,
        "cnn_training_metadata": {
            "status": cnn_run_status,
            "best_epoch": cnn_best_epoch,
        },
        "random_forest": {
            "accuracy": round(rf_accuracy, 4),
            "macro_f1": round(rf_macro_f1, 4),
            "bootstrap_ci_95": {
                "accuracy": rf_acc_ci,
                "macro_f1": rf_f1_ci,
            },
            "classification_report": rf_clf_report,
            "most_confused_pairs": rf_confused_pairs,
        },
        "cnn": {
            "accuracy": round(cnn_accuracy, 4),
            "macro_f1": round(cnn_macro_f1, 4),
            "bootstrap_ci_95": {
                "accuracy": cnn_acc_ci,
                "macro_f1": cnn_f1_ci,
            },
            "classification_report": cnn_clf_report,
            "most_confused_pairs": cnn_confused_pairs,
            "misclassified_examples_sample": [
                {
                    "track_id": ex["track_id"],
                    "true_label": ex["true_label"],
                    "predicted_label": ex["predicted_label"],
                    "top_3_scores": ex["top_3_scores"],
                }
                for ex in misclassified_examples
            ],
        },
        "statistical_comparison": stat_results,
        "latency_benchmark": {
            "device": device_str,
            "hardware_name": hw_info.get("device_name", hw_info.get("processor", "unknown")),
            **latency_results,
        },
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(test_metrics_payload, f, indent=2)

    print(f"\nSuccessfully wrote test metrics to {metrics_file.relative_to(root) if metrics_file.is_relative_to(root) else metrics_file}")
    print(f"Summary:")
    print(f"  Random Forest Test Accuracy: {rf_accuracy:.4f} (Macro-F1: {rf_macro_f1:.4f})")
    print(f"  2D CNN Test Accuracy        : {cnn_accuracy:.4f} (Macro-F1: {cnn_macro_f1:.4f})")
    print(f"  Paired Bootstrap Diff 95% CI: [{stat_results['difference_cnn_minus_rf']['accuracy_difference_ci_95'][0]:.4f}, {stat_results['difference_cnn_minus_rf']['accuracy_difference_ci_95'][1]:.4f}]")

    return test_metrics_payload


if __name__ == "__main__":
    args = parse_args()
    evaluate_models(
        config_path=args.config,
        force=args.force,
        limit=args.limit,
        out_dir_override=args.out_dir,
        device_override=args.device,
        latency_tracks=args.latency_tracks,
    )

"""Exploratory Data Analysis (EDA) for SonicFusion 2.

Follows docs/PLAN.md (§07, §16, §19):
- Reads features.parquet, manifest.csv, data/cache/logmels, and label_map.json.
- Generates publication-ready figures in reports/figures/:
  1. Class counts by official split (training, validation, test).
  2. 2 example log-mel spectrograms per genre (captioned with track_id).
  3. Track duration distribution.
  4. Spectral centroid, RMS energy, and tempo distributions by genre.
- Exports structured reports/metrics/eda_summary.json:
  - Per-split track totals
  - Exclusions count
  - Mean decodable audio duration
  - Feature distributions and class counts
- Supports --limit N for rapid testing and --out-dir for safe dry-runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.audio.loading import load_audio, segment_audio
from src.audio.spectrogram import compute_logmel_spectrogram
from src.config import get_project_root, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for EDA."""
    parser = argparse.ArgumentParser(description="Generate exploratory data analysis figures and metrics summary.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of manifest tracks for rapid verification.")
    parser.add_argument("--out-dir", type=str, default=None, help="Base output directory override.")
    return parser.parse_args()


def plot_class_counts_by_split(
    manifest_df: pd.DataFrame,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Generate a grouped bar chart of genre class counts across official splits."""
    splits = ["training", "validation", "test"]
    counts = {s: [] for s in splits}

    for s in splits:
        split_df = manifest_df[manifest_df["split"] == s]
        for c in class_names:
            counts[s].append(int((split_df["genre"] == c).sum()))

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    colors = {"training": "#3b82f6", "validation": "#10b981", "test": "#8b5cf6"}

    for i, s in enumerate(splits):
        rects = ax.bar(x + (i - 1) * width, counts[s], width, label=s.capitalize(), color=colors[s], alpha=0.9)
        # Add labels
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{int(height)}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_ylabel("Track Count", fontsize=11)
    ax.set_title("FMA Small: Genre Class Distribution Across Official Splits (PLAN §06, §07)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_example_spectrograms(
    manifest_df: pd.DataFrame,
    class_names: List[str],
    cache_dir: Path,
    output_path: Path,
    sr: int = 22050,
) -> None:
    """Plot 2 example log-mel spectrograms per genre (16 panels total), captioned with track_id."""
    n_genres = len(class_names)
    fig, axes = plt.subplots(n_genres, 2, figsize=(12, 2.2 * n_genres), dpi=150, sharex=True, sharey=True)

    if n_genres == 1:
        axes = np.array([axes])

    for row_idx, genre in enumerate(class_names):
        genre_tracks = manifest_df[manifest_df["genre"] == genre]
        sample_tracks = genre_tracks.head(2).to_dict(orient="records")

        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            if col_idx < len(sample_tracks):
                track = sample_tracks[col_idx]
                t_id = int(track["track_id"])
                cache_file = cache_dir / f"{t_id:06d}_0.npy"

                spec_arr: Optional[np.ndarray] = None
                if cache_file.exists():
                    try:
                        spec_arr = np.load(cache_file).astype(np.float32)
                    except Exception:
                        spec_arr = None

                if spec_arr is None:
                    # Fallback compute from audio if cache missing
                    audio_path = resolve_path(str(track["path"]))
                    if audio_path.exists():
                        try:
                            y = load_audio(audio_path, sr=sr, duration=30.0, mono=True)
                            segs = segment_audio(y, sr=sr, window_sec=10.0, num_windows=3)
                            spec_arr = compute_logmel_spectrogram(segs[0], sr=sr, n_fft=2048, hop_length=512, n_mels=128)
                        except Exception:
                            spec_arr = np.zeros((128, 431), dtype=np.float32)
                    else:
                        spec_arr = np.zeros((128, 431), dtype=np.float32)

                im = ax.imshow(spec_arr, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
                ax.set_title(f"{genre} — Track #{t_id:06d} (Window 1)", fontsize=9, pad=4)
            else:
                ax.text(0.5, 0.5, "No sample", ha="center", va="center", transform=ax.transAxes)

            if col_idx == 0:
                ax.set_ylabel(f"{genre}\nMel Bins", fontsize=8)
            if row_idx == n_genres - 1:
                ax.set_xlabel("Time Frames (~431)", fontsize=8)

    fig.suptitle("Example 10-Second Log-Mel Spectrograms per Genre (PLAN §07)", fontsize=13, y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_duration_distribution(
    manifest_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot track duration distribution histogram."""
    durations: List[float] = []
    # If decodable_seconds exists in manifest, use it; otherwise approximate standard 30s
    if "duration" in manifest_df.columns:
        durations = manifest_df["duration"].dropna().tolist()
    elif "decodable_seconds" in manifest_df.columns:
        durations = manifest_df["decodable_seconds"].dropna().tolist()
    else:
        durations = [30.0] * len(manifest_df)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(durations, bins=30, color="#6366f1", edgecolor="black", alpha=0.8)
    ax.axvline(30.0, color="#ef4444", linestyle="--", label="Target Duration (30.0s)")
    ax.set_xlabel("Track Duration (seconds)", fontsize=11)
    ax.set_ylabel("Number of Tracks", fontsize=11)
    ax.set_title("FMA Small Audio Duration Distribution (PLAN §07)", fontsize=12, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_acoustic_feature_distributions(
    features_df: pd.DataFrame,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Plot spectral centroid, RMS energy, and tempo distributions by genre."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), dpi=150)

    target_features = [
        ("spectral_centroid_mean", "Spectral Centroid Mean (Hz)", "#06b6d4"),
        ("rms_mean", "RMS Energy Mean", "#f59e0b"),
        ("tempo", "Estimated Tempo (BPM)", "#ec4899"),
    ]

    for idx, (ax, (feat_col, feat_label, color)) in enumerate(zip(axes, target_features)):
        if feat_col in features_df.columns:
            data = [
                features_df[features_df["genre"] == c][feat_col].dropna().values
                for c in class_names
            ]
            # Replace empty arrays with [0.0] if a genre has no rows under limit
            data = [d if len(d) > 0 else np.array([0.0]) for d in data]
            bp = ax.boxplot(
                data,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
                medianprops={"color": "black", "linewidth": 1.5},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel(feat_label, fontsize=10)
            ax.set_xticks(range(1, len(class_names) + 1))
            if idx == len(axes) - 1:
                ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
            else:
                ax.set_xticklabels([])
            ax.grid(axis="y", linestyle="--", alpha=0.4)
        else:
            ax.text(0.5, 0.5, f"Feature '{feat_col}' not found in features.parquet", ha="center", va="center")

    axes[0].set_title("Acoustic Feature Distributions Across Genres (PLAN §07, §11)", fontsize=13, pad=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_eda(
    config_path: Optional[str] = None,
    limit: Optional[int] = None,
    out_dir_override: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute complete exploratory data analysis."""
    cfg = load_config(config_path)
    root = get_project_root()

    if out_dir_override:
        base_out = Path(out_dir_override).resolve()
        metrics_dir = base_out / "metrics"
        figures_dir = base_out / "figures"
    else:
        metrics_dir = resolve_path(cfg.paths.metrics_dir)
        figures_dir = resolve_path(cfg.paths.figures_dir)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_path(cfg.paths.manifest_path)
    features_path = resolve_path(cfg.paths.features_path)
    label_map_path = resolve_path(cfg.paths.label_map_path)
    cache_dir = resolve_path(cfg.paths.spectrogram_cache_dir)
    exclusions_path = resolve_path(cfg.paths.metrics_dir) / "data_exclusions.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features dataset not found at {features_path}")

    print(f"Loading manifest from {manifest_path}...")
    manifest_df = pd.read_csv(manifest_path)
    print(f"Loading features from {features_path}...")
    features_df = pd.read_parquet(features_path)

    if limit is not None and limit > 0:
        manifest_df = manifest_df.iloc[:limit].copy()
        features_df = features_df[features_df["track_id"].isin(manifest_df["track_id"])].copy()
        print(f"Limited dataset to {len(manifest_df)} tracks for EDA.")

    # Class names
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            lmap = json.load(f)
        class_names = sorted(lmap.keys(), key=lambda k: lmap[k])
    else:
        class_names = sorted(manifest_df["genre"].dropna().unique().tolist())

    # Count exclusions
    exclusions_count = 0
    if exclusions_path.exists():
        try:
            excl_df = pd.read_csv(exclusions_path)
            exclusions_count = len(excl_df)
        except Exception:
            pass

    # Per-split totals
    per_split_totals = {
        "training": int((manifest_df["split"] == "training").sum()),
        "validation": int((manifest_df["split"] == "validation").sum()),
        "test": int((manifest_df["split"] == "test").sum()),
        "total": len(manifest_df),
    }

    # Class counts by split
    genre_counts_by_split = {}
    for s in ["training", "validation", "test"]:
        split_subset = manifest_df[manifest_df["split"] == s]
        genre_counts_by_split[s] = {c: int((split_subset["genre"] == c).sum()) for c in class_names}

    # Mean duration
    mean_duration = 30.0
    if "duration" in manifest_df.columns:
        mean_duration = float(manifest_df["duration"].mean())

    # Key acoustic feature statistics by genre
    acoustic_stats = {}
    for c in class_names:
        c_feats = features_df[features_df["genre"] == c]
        acoustic_stats[c] = {
            "spectral_centroid_mean": float(round(float(c_feats["spectral_centroid_mean"].mean()), 2)) if "spectral_centroid_mean" in c_feats else 0.0,
            "rms_mean": float(round(float(c_feats["rms_mean"].mean()), 4)) if "rms_mean" in c_feats else 0.0,
            "tempo_mean": float(round(float(c_feats["tempo"].mean()), 2)) if "tempo" in c_feats else 0.0,
        }

    # 1. Plot Class Counts by Split
    print("Generating class counts by split figure...")
    plot_class_counts_by_split(manifest_df, class_names, figures_dir / "eda_class_counts_by_split.png")

    # 2. Plot Example Spectrograms
    print("Generating example spectrograms per genre...")
    plot_example_spectrograms(manifest_df, class_names, cache_dir, figures_dir / "eda_spectrogram_examples.png")

    # 3. Plot Duration Distribution
    print("Generating duration distribution figure...")
    plot_duration_distribution(manifest_df, figures_dir / "eda_duration_distribution.png")

    # 4. Plot Acoustic Feature Distributions
    print("Generating acoustic feature distributions figure...")
    plot_acoustic_feature_distributions(features_df, class_names, figures_dir / "eda_feature_distributions.png")

    # Summary payload
    eda_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_split_totals": per_split_totals,
        "exclusions_count": exclusions_count,
        "mean_duration_seconds": float(round(mean_duration, 2)),
        "class_names": class_names,
        "genre_counts_by_split": genre_counts_by_split,
        "acoustic_feature_stats_by_genre": acoustic_stats,
    }

    summary_file = metrics_dir / "eda_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    print(f"Saved EDA summary metrics to {summary_file}")
    print(f"Generated 4 EDA figures in {figures_dir}")
    return eda_summary


if __name__ == "__main__":
    args = parse_args()
    run_eda(config_path=args.config, limit=args.limit, out_dir_override=args.out_dir)

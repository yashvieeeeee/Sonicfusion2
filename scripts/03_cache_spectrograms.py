"""Compute training normalization statistics and cache float16 log-mel spectrogram arrays.

Follows PLAN §07 audio contract:
- 22,050 Hz mono waveform.
- 3 non-overlapping 10-second windows per track.
- 128 mel bands, n_fft=2048, hop_length=512, power=2.0, power_to_dB with ref=np.max.
- Computes mean and std from TRAINING windows only and saves to data/processed/norm_stats.json.
- Normalizes all windows (train, validation, test) using the saved training stats.
- Caches each window as float16 .npy in data/cache/logmels/{track_id:06d}_{window_idx}.npy.
- Supports --limit, --resume, and interactive progress.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.audio.loading import load_audio, segment_audio
from src.audio.spectrogram import (
    RunningStats,
    compute_logmel_spectrogram,
    load_norm_stats,
    normalize_spectrogram,
    save_norm_stats,
)
from src.config import get_project_root, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute training normalization stats and cache float16 log-mel spectrograms."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tracks to process from the manifest.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume caching, reusing existing norm_stats.json and skipping already cached .npy files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for spectrogram caching (default: 8).",
    )
    return parser.parse_args()


def cache_spectrograms(
    limit: Optional[int] = None,
    resume: bool = False,
    config_path: Optional[str] = None,
    workers: int = 8,
) -> None:
    """Execute normalization statistics computation and spectrogram caching."""
    cfg = load_config(config_path)
    root = get_project_root()

    manifest_path = resolve_path(cfg.paths.manifest_path)
    norm_stats_path = resolve_path(cfg.paths.norm_stats_path)
    cache_dir = resolve_path(cfg.paths.spectrogram_cache_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found at: {manifest_path}. "
            "Please run scripts/01_build_manifest.py first."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    if limit is not None and limit > 0:
        manifest_df = manifest_df.iloc[:limit]

    print(f"Loaded manifest with {len(manifest_df)} tracks.")

    sr = int(cfg.audio.sample_rate)
    duration = float(cfg.audio.duration)
    win_sec = float(cfg.audio.window_duration)
    num_windows = int(cfg.audio.windows_per_track)
    n_mels = int(cfg.audio.n_mels)
    n_fft = int(cfg.audio.n_fft)
    hop_length = int(cfg.audio.hop_length)

    # -------------------------------------------------------------
    # Step 1: Compute or load normalization statistics (TRAIN ONLY)
    # -------------------------------------------------------------
    mean: float
    std: float

    if resume and norm_stats_path.exists():
        stats = load_norm_stats(norm_stats_path)
        mean = stats["mean"]
        std = stats["std"]
        print(f"Loaded existing normalization stats from {norm_stats_path.relative_to(root)}: mean={mean:.4f}, std={std:.4f}")
    else:
        train_tracks = manifest_df[manifest_df["split"] == "training"]
        if train_tracks.empty:
            raise ValueError("No tracks with split == 'training' found in manifest to compute normalization stats!")

        print(f"\nComputing normalization statistics across {len(train_tracks)} TRAINING tracks ({len(train_tracks) * num_windows} windows)...")
        running_stats = RunningStats()

        progress_stats = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        with progress_stats:
            task_stats = progress_stats.add_task(
                "Calculating training stats...", total=len(train_tracks)
            )

            for _, row in train_tracks.iterrows():
                full_path = resolve_path(str(row["path"]))
                try:
                    y = load_audio(full_path, sr=sr, duration=duration, mono=True)
                    segments = segment_audio(y, sr=sr, window_sec=win_sec, num_windows=num_windows)
                    for seg in segments:
                        logmel = compute_logmel_spectrogram(
                            segment=seg,
                            sr=sr,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            n_mels=n_mels,
                        )
                        running_stats.update(logmel)
                except Exception as e:
                    print(f"Warning: Failed to process training track {row['track_id']} for stats: {e}")

                progress_stats.advance(task_stats)

        mean = running_stats.mean
        std = running_stats.std
        num_windows_computed = running_stats.count // (n_mels * 431)
        save_norm_stats(
            mean=mean,
            std=std,
            output_path=norm_stats_path,
            extra_metadata={
                "training_tracks_count": len(train_tracks),
                "training_windows_count": num_windows_computed,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "sr": sr,
            },
        )
        print(f"Computed & saved training normalization stats to {norm_stats_path.relative_to(root)}:")
        print(f"  Mean: {mean:.4f}, Std: {std:.4f} (across {num_windows_computed} training windows)")

    # -------------------------------------------------------------
    # Step 2: Cache float16 .npy per window (normalized with train stats)
    # -------------------------------------------------------------
    print(f"\nCaching float16 .npy spectrograms to {cache_dir.relative_to(root)}...")

    def _cache_track(row_dict: Dict[str, Any]) -> int:
        t_id = int(row_dict["track_id"])
        full_path = resolve_path(str(row_dict["path"]))
        win_paths = [cache_dir / f"{t_id:06d}_{w}.npy" for w in range(num_windows)]

        if resume and all(p.exists() for p in win_paths):
            return 0

        written = 0
        try:
            y = load_audio(full_path, sr=sr, duration=duration, mono=True)
            segments = segment_audio(y, sr=sr, window_sec=win_sec, num_windows=num_windows)

            for w_idx, seg in enumerate(segments):
                out_path = win_paths[w_idx]
                if resume and out_path.exists():
                    continue

                logmel = compute_logmel_spectrogram(
                    segment=seg,
                    sr=sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                )
                normed = normalize_spectrogram(logmel, mean=mean, std=std)
                np.save(out_path, normed.astype(np.float16))
                written += 1
        except Exception:
            pass

        return written

    progress_cache = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    cached_count = 0
    row_records = manifest_df.to_dict(orient="records")

    with progress_cache:
        task_cache = progress_cache.add_task(f"Caching spectrograms ({workers} workers)...", total=len(row_records))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i in range(0, len(row_records), 100):
                batch_slice = row_records[i : i + 100]
                futures = [executor.submit(_cache_track, r) for r in batch_slice]
                for fut in as_completed(futures):
                    cached_count += fut.result()
                    progress_cache.advance(task_cache)

    print(f"\nCaching complete: {cached_count} windows written.")
    print(f"Total files in cache directory: {len(list(cache_dir.glob('*.npy')))}")


if __name__ == "__main__":
    args = parse_args()
    cache_spectrograms(
        limit=args.limit,
        resume=args.resume,
        config_path=args.config,
        workers=args.workers,
    )

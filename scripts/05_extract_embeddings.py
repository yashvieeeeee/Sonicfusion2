"""Extract pretrained audio embeddings using MERT-v1-95M (5.0s windows, no overlap).

Processes audio tracks using 5-second windows with no overlap:
- 6 non-overlapping windows per 30-second track.
- Per-window embedding shape: (6, 768).
- Per-track mean embedding shape: (768,).
- Resumable: Skips tracks whose embedding artifacts already exist.
- Saves both per-window and mean embeddings to data/embeddings/MERT-v1-95M-5s/ as .npy with track index.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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

from src.config import get_project_root, load_config, resolve_path
from src.features.embeddings import (
    AudioEmbedder,
    DEFAULT_HOP_SEC,
    DEFAULT_WINDOW_SEC,
    MODEL_ALIAS,
)

DATASET_TOTAL_TRACKS = 7994


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract pretrained audio embeddings using MERT-v1-95M (5s windows, no overlap)."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest.csv (defaults to data/processed/manifest.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save extracted embeddings (defaults to data/embeddings/MERT-v1-95M-5s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tracks to process (e.g. --limit 10 for initial test).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "training", "validation", "test"],
        default="all",
        help="Optional split filter: 'training', 'validation', 'test', or 'all' (default: 'all').",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=DEFAULT_WINDOW_SEC,
        help="Window length in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=DEFAULT_HOP_SEC,
        help="Hop length in seconds between windows (default: 5.0, no overlap).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on: 'cpu', 'cuda', or None for auto-detect.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cached embeddings (default: resume/skip existing).",
    )
    return parser.parse_args()


def extract_embeddings_pipeline(
    manifest_path_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    limit: Optional[int] = None,
    split_filter: str = "all",
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    device: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Extract and persist per-window and track-mean embeddings for dataset tracks."""
    root = get_project_root()
    cfg = load_config()

    manifest_file = (
        resolve_path(manifest_path_override)
        if manifest_path_override
        else resolve_path(cfg.paths.manifest_path)
    )

    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found at: {manifest_file}")

    out_dir = (
        resolve_path(output_dir_override)
        if output_dir_override
        else root / "data" / "embeddings" / MODEL_ALIAS
    )
    windows_dir = out_dir / "windows"
    mean_dir = out_dir / "mean"

    windows_dir.mkdir(parents=True, exist_ok=True)
    mean_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {manifest_file.relative_to(root)}...")
    manifest_df = pd.read_csv(manifest_file)

    if split_filter != "all":
        manifest_df = manifest_df[manifest_df["split"] == split_filter].copy()

    if limit is not None and limit > 0:
        manifest_df = manifest_df.iloc[:limit].copy()

    total_requested = len(manifest_df)
    expected_windows = int((30.0 - window_sec) / hop_sec) + 1

    print(f"Target tracks to process: {total_requested} (split filter: {split_filter})")
    print(f"Window configuration   : {window_sec:.1f}s window, {hop_sec:.1f}s hop ({expected_windows} windows/track)")
    print(f"Output directory        : {out_dir.relative_to(root)}")

    # Initialize pretrained embedder
    print(f"\nInitializing AudioEmbedder (Model: {MODEL_ALIAS})...")
    embedder = AudioEmbedder(device=device)
    print(f"  Device in use: {embedder.device}")
    print(f"  Sample rate  : {embedder.sr} Hz")

    # Load existing index if available to support seamless resumption
    index_file = out_dir / "index.csv"
    existing_index: Dict[int, Dict[str, Any]] = {}
    if index_file.exists() and not force:
        try:
            prev_df = pd.read_csv(index_file)
            for _, r in prev_df.iterrows():
                existing_index[int(r["track_id"])] = r.to_dict()
        except Exception:
            existing_index = {}

    processed_records: List[Dict[str, Any]] = []
    skipped_count = 0
    extracted_count = 0
    durations: List[float] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    sample_window_shape: Optional[Tuple[int, ...]] = None
    sample_mean_shape: Optional[Tuple[int, ...]] = None

    with progress:
        task = progress.add_task(f"Extracting {MODEL_ALIAS} embeddings...", total=total_requested)

        for _, row in manifest_df.iterrows():
            t_id = int(row["track_id"])
            genre = str(row["genre"])
            split = str(row["split"])
            audio_path = resolve_path(str(row["path"]))

            win_file = windows_dir / f"{t_id:06d}.npy"
            mean_file = mean_dir / f"{t_id:06d}.npy"

            # Check resumability
            if not force and win_file.exists() and mean_file.exists():
                skipped_count += 1
                rec = existing_index.get(t_id, {
                    "track_id": t_id,
                    "genre": genre,
                    "split": split,
                    "windows_file": str(win_file.relative_to(root)),
                    "mean_file": str(mean_file.relative_to(root)),
                    "num_windows": expected_windows,
                    "embedding_dim": 768,
                })
                processed_records.append(rec)
                progress.advance(task)
                continue

            t0 = time.perf_counter()
            try:
                win_emb, mean_emb = embedder.extract_track_embeddings(
                    audio_path,
                    window_sec=window_sec,
                    hop_sec=hop_sec,
                )
            except Exception as exc:
                print(f"\nWarning: Failed to extract embeddings for track {t_id:06d}: {exc}")
                progress.advance(task)
                continue
            t1 = time.perf_counter()

            track_time = t1 - t0
            durations.append(track_time)

            if sample_window_shape is None:
                sample_window_shape = win_emb.shape
                sample_mean_shape = mean_emb.shape

            # Save per-window and per-track mean .npy arrays
            np.save(win_file, win_emb)
            np.save(mean_file, mean_emb)

            rec = {
                "track_id": t_id,
                "genre": genre,
                "split": split,
                "windows_file": str(win_file.relative_to(root)),
                "mean_file": str(mean_file.relative_to(root)),
                "num_windows": win_emb.shape[0],
                "embedding_dim": win_emb.shape[1],
            }
            processed_records.append(rec)
            extracted_count += 1

            progress.advance(task)

    # Save consolidated track_id index CSV
    if processed_records:
        index_df = pd.DataFrame(processed_records)
        index_df = index_df.drop_duplicates(subset=["track_id"], keep="last").sort_values("track_id")
        index_df.to_csv(index_file, index=False)

        # Also save track_ids.npy and mean_embeddings.npy for instantaneous downstream loading
        all_tids = index_df["track_id"].values.astype(np.int64)
        np.save(out_dir / "track_ids.npy", all_tids)

        # Assemble mean embeddings
        mean_arrays = [np.load(root / row["mean_file"]) for _, row in index_df.iterrows()]
        all_means = np.stack(mean_arrays, axis=0).astype(np.float32)
        np.save(out_dir / "mean_embeddings.npy", all_means)

    avg_time_per_track = float(np.mean(durations)) if durations else 0.0
    projected_total_seconds = avg_time_per_track * DATASET_TOTAL_TRACKS
    projected_total_hours = projected_total_seconds / 3600.0

    summary = {
        "model_name": MODEL_ALIAS,
        "huggingface_model_id": embedder.model_name,
        "device": embedder.device,
        "sample_rate": embedder.sr,
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "num_windows_per_track": expected_windows,
        "total_requested": total_requested,
        "extracted_count": extracted_count,
        "skipped_count": skipped_count,
        "per_window_shape": list(sample_window_shape) if sample_window_shape else [expected_windows, 768],
        "per_track_mean_shape": list(sample_mean_shape) if sample_mean_shape else [768],
        "avg_time_per_track_seconds": round(avg_time_per_track, 3),
        "total_extraction_time_seconds": round(float(np.sum(durations)), 3),
        "dataset_total_tracks": DATASET_TOTAL_TRACKS,
        "projected_total_seconds": round(projected_total_seconds, 1),
        "projected_total_hours": round(projected_total_hours, 2),
        "output_directory": str(out_dir.relative_to(root)),
    }

    _print_extraction_summary(summary)
    return summary


def _print_extraction_summary(s: Dict[str, Any]) -> None:
    """Print formatted summary of embedding extraction."""
    print("\n" + "=" * 65)
    print("      MERT-v1-95M-5s AUDIO EMBEDDING EXTRACTION SUMMARY       ")
    print("=" * 65)
    print(f"Pretrained Model        : {s['model_name']} ({s['huggingface_model_id']})")
    print(f"Device                  : {s['device']}")
    print(f"Audio Sample Rate       : {s['sample_rate']} Hz")
    print(f"Window Configuration    : {s['window_sec']:.1f}s window, {s['hop_sec']:.1f}s hop ({s['num_windows_per_track']} windows/track)")
    print(f"Total Tracks Requested  : {s['total_requested']}")
    print(f"Newly Extracted Tracks  : {s['extracted_count']}")
    print(f"Skipped (Already Cached): {s['skipped_count']}")
    print(f"Per-Window Shape        : {s['per_window_shape']} ({s['num_windows_per_track']} windows x 768 dims)")
    print(f"Per-Track Mean Shape    : {s['per_track_mean_shape']} (768 dims)")
    if s["extracted_count"] > 0:
        print(f"Average Time per Track  : {s['avg_time_per_track_seconds']:.3f} s / track")
        print(f"Total Benchmark Time    : {s['total_extraction_time_seconds']:.3f} s")
        print(f"Projected Full Dataset  : {s['dataset_total_tracks']} tracks")
        print(f"Projected Total Time    : {s['projected_total_hours']:.2f} hours ({s['projected_total_seconds']:.1f} seconds)")
    print(f"Output Directory        : {s['output_directory']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    args = parse_args()
    extract_embeddings_pipeline(
        manifest_path_override=args.manifest,
        output_dir_override=args.output_dir,
        limit=args.limit,
        split_filter=args.split,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        device=args.device,
        force=args.force,
    )

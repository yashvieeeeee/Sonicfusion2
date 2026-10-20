"""Extract handcrafted acoustic feature vectors for classical baseline and similarity.

Implements Gate G3 feature extraction from docs/PLAN.md (§05, §08, Addendum A4):
- Extracts 51 acoustic features per track (timbral, tonal, spectral, dynamics, rhythm).
- Handles NaN/Inf and silent clips explicitly.
- Saves results to data/processed/features.parquet.
- Supports --limit, --resume, and interactive progress reporting.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

from src.audio.loading import load_audio
from src.config import get_project_root, load_config, resolve_path
from src.features.extraction import FEATURE_NAMES, extract_track_features


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract acoustic feature vectors from manifest tracks into Parquet."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tracks to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing features.parquet if present, skipping completed tracks.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--batch-save",
        type=int,
        default=50,
        help="Checkpoint save to disk every N tracks.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for audio feature extraction (default: 8).",
    )
    return parser.parse_args()


def extract_features(
    limit: Optional[int] = None,
    resume: bool = False,
    config_path: Optional[str] = None,
    batch_save: int = 50,
    workers: int = 8,
) -> pd.DataFrame:
    """Execute feature extraction pipeline."""
    cfg = load_config(config_path)
    root = get_project_root()

    manifest_path = resolve_path(cfg.paths.manifest_path)
    features_path = resolve_path(cfg.paths.features_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found at: {manifest_path}. "
            "Please run scripts/01_build_manifest.py first."
        )

    features_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    if limit is not None and limit > 0:
        manifest_df = manifest_df.iloc[:limit]

    total_manifest = len(manifest_df)
    print(f"Loaded manifest with {total_manifest} tracks to process.")

    existing_df: Optional[pd.DataFrame] = None
    existing_ids: Set[int] = set()

    if resume and features_path.exists():
        try:
            existing_df = pd.read_parquet(features_path)
            existing_ids = set(existing_df["track_id"].astype(int).tolist())
            print(f"Resuming: found {len(existing_ids)} tracks already extracted in {features_path.relative_to(root)}.")
        except Exception as e:
            print(f"Could not load existing parquet for resume ({e}), starting fresh.")
            existing_df = None
            existing_ids = set()

    tracks_to_process = manifest_df[~manifest_df["track_id"].astype(int).isin(existing_ids)]
    print(f"Tracks remaining to extract: {len(tracks_to_process)}")

    if tracks_to_process.empty and existing_df is not None:
        print("All tracks already processed. Nothing to do.")
        return existing_df

    sr = int(cfg.audio.sample_rate)
    duration = float(cfg.audio.duration)

    new_rows: List[Dict[str, Any]] = []

    def _flush_to_disk() -> None:
        nonlocal existing_df, new_rows
        if not new_rows:
            return
        chunk_df = pd.DataFrame(new_rows)
        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, chunk_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["track_id"], keep="last")
        else:
            combined = chunk_df

        combined.to_parquet(features_path, index=False)
        existing_df = combined
        new_rows = []

    def _process_track(row_dict: Dict[str, Any]) -> Dict[str, Any]:
        t_id = int(row_dict["track_id"])
        rel_path = str(row_dict["path"])
        genre = str(row_dict["genre"])
        split = str(row_dict["split"])
        artist_id = row_dict.get("artist_id")
        artist_val = int(artist_id) if pd.notna(artist_id) else None

        full_path = resolve_path(rel_path)

        try:
            y = load_audio(full_path, sr=sr, duration=duration, mono=True)
            feats = extract_track_features(y, sr=sr)
        except Exception:
            feats = {name: 0.0 for name in FEATURE_NAMES}

        return {
            "track_id": t_id,
            "genre": genre,
            "split": split,
            "artist_id": artist_val,
            **feats,
        }

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    row_records = tracks_to_process.to_dict(orient="records")

    with progress:
        task = progress.add_task(f"Extracting features ({workers} workers)...", total=len(row_records))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i in range(0, len(row_records), batch_save):
                batch_slice = row_records[i : i + batch_save]
                futures = [executor.submit(_process_track, r) for r in batch_slice]
                for fut in as_completed(futures):
                    rec = fut.result()
                    new_rows.append(rec)
                    progress.advance(task)
                _flush_to_disk()

    # Final save
    _flush_to_disk()

    final_df = pd.read_parquet(features_path)
    print(f"\nSuccessfully saved {len(final_df)} feature vectors to {features_path.relative_to(root)}.")
    print(f"Features DataFrame shape: {final_df.shape} (columns: {len(final_df.columns)})")

    return final_df


if __name__ == "__main__":
    args = parse_args()
    extract_features(
        limit=args.limit,
        resume=args.resume,
        config_path=args.config,
        batch_save=args.batch_save,
        workers=args.workers,
    )

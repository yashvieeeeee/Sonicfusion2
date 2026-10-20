"""Build dataset manifest from FMA metadata and audio files.

Implements Gate G2 requirements from docs/PLAN.md (§05, §06, §16, Addenda A4):
- Reads data/raw/fma_metadata/tracks.csv with multi-row header.
- Filters subset == 'small'.
- Validates and decode-tests each MP3 using src.audio.validation.verify_audio_file_strict.
- Excludes and logs any dataset track whose strictly decodable duration is below
  dataset.min_track_seconds (default: 25.0s), strictly avoiding lenient ffmpeg salvage.
- Writes data/processed/manifest.csv (track_id, path, genre, split, artist_id).
- Writes data/processed/label_map.json (alphabetical mapping).
- Supports --out-dir (default data/processed) for isolated rebuilds.
- Supports --audit-only to read the existing manifest.csv and output fresh
  reports/metrics/manifest_audit.json and data_exclusions.csv without touching
  manifest.csv, label_map.json, features, or caches.
- Computes per-split metadata vs. manifest counts and artist split overlap as a documented limitation.
- Supports --limit N for fast debugging and testing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path for direct script invocation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.audio.validation import sanitize_path_str, verify_audio_file_strict
from src.config import load_config, resolve_path, get_project_root


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build FMA Small dataset manifest and label map with strict decode verification."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed",
        help="Output directory for manifest.csv and label_map.json (default: data/processed).",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit existing manifest against metadata and write manifest_audit.json and data_exclusions.csv without modifying manifest, label map, or caches.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Path to manifest.csv to audit when using --audit-only (defaults to data/processed/manifest.csv).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help="Directory to write data_exclusions.csv and manifest_audit.json (defaults to reports/metrics).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tracks processed (useful for rapid testing / development).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (defaults to configs/config.yaml).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for MP3 decode testing (default: 8).",
    )
    return parser.parse_args()


def build_manifest(
    out_dir: Union[str, Path] = "data/processed",
    audit_only: bool = False,
    limit: Optional[int] = None,
    config_path: Optional[str] = None,
    workers: int = 8,
    metrics_dir_override: Optional[Union[str, Path]] = None,
    manifest_path_override: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute manifest creation, audio verification, label map building, or audit.

    Args:
        out_dir: Output directory for manifest.csv and label_map.json (rebuild mode).
        audit_only: If True, only audits existing manifest.csv against metadata.
        limit: Optional maximum number of tracks to process.
        config_path: Optional path to config.yaml.
        workers: Concurrency level for audio verification.
        metrics_dir_override: Optional override for reports/metrics destination.
        manifest_path_override: Optional path to existing manifest.csv (used in audit-only).

    Returns:
        Audit report dictionary.
    """
    cfg = load_config(config_path)
    root = get_project_root()

    # Resolve dataset.min_track_seconds (PLAN §06, default: 25.0s)
    dataset_cfg = getattr(cfg, "dataset", {})
    if isinstance(dataset_cfg, dict):
        min_track_seconds = float(dataset_cfg.get("min_track_seconds", 25.0))
    else:
        min_track_seconds = float(getattr(dataset_cfg, "min_track_seconds", 25.0))

    tracks_csv_path = resolve_path(cfg.paths.fma_metadata_dir) / "tracks.csv"
    metrics_dir = (
        Path(metrics_dir_override)
        if metrics_dir_override is not None
        else resolve_path(cfg.paths.metrics_dir)
    )
    exclusions_csv_path = metrics_dir / "data_exclusions.csv"
    audit_json_path = metrics_dir / "manifest_audit.json"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from {tracks_csv_path.relative_to(root)}...")
    if not tracks_csv_path.exists():
        raise FileNotFoundError(
            f"FMA metadata tracks.csv not found at: {tracks_csv_path}. "
            "Please ensure data/raw/fma_metadata.zip is extracted into data/raw."
        )

    # Read tracks.csv with 2-level column header (PLAN §06)
    tracks = pd.read_csv(tracks_csv_path, index_col=0, header=[0, 1])

    # Filter strictly for subset == "small"
    small_all = tracks[tracks[("set", "subset")] == "small"]
    total_metadata_tracks = len(small_all)
    print(f"Found {total_metadata_tracks} tracks in FMA 'small' metadata subset.")

    # ---------------------------------------------------------
    # MODE 1: AUDIT-ONLY MODE
    # ---------------------------------------------------------
    if audit_only:
        print("\n[AUDIT-ONLY MODE] Reading existing manifest and auditing exclusions...")
        manifest_path = (
            Path(manifest_path_override)
            if manifest_path_override is not None
            else (resolve_path(out_dir) / "manifest.csv" if (resolve_path(out_dir) / "manifest.csv").exists() else resolve_path(cfg.paths.manifest_path))
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot run --audit-only: manifest file not found at {manifest_path}")

        print(f"Reading manifest from {manifest_path.relative_to(root) if manifest_path.is_relative_to(root) else manifest_path}...")
        manifest_df = pd.read_csv(manifest_path)
        manifest_tids = set(manifest_df["track_id"])
        print(f"Loaded {len(manifest_df)} tracks from existing manifest.")

        # Identify excluded tracks (tracks in metadata small subset but absent from manifest)
        excluded_tids = sorted([int(tid) for tid in small_all.index if int(tid) not in manifest_tids])
        print(f"Identified {len(excluded_tids)} excluded track IDs: {excluded_tids}")

        exclusion_rows: List[Dict[str, Any]] = []
        for tid in excluded_tids:
            row = small_all.loc[tid]
            genre = str(row[("track", "genre_top")])
            split = str(row[("set", "split")])
            rel_path = f"data/raw/fma_small/{tid // 1000:03d}/{tid:06d}.mp3"
            full_path = resolve_path(rel_path)

            # Strict decode test: check header, corruption, and duration without ffmpeg salvage
            is_valid, decodable_sec, file_size, reason = verify_audio_file_strict(
                full_path, min_duration=min_track_seconds
            )
            if not reason:
                reason = f"Decodable duration ({decodable_sec:.2f}s) is below min_track_seconds ({min_track_seconds:.1f}s)"
            else:
                reason = sanitize_path_str(reason, project_root=root)

            exclusion_rows.append({
                "track_id": tid,
                "genre": genre,
                "split": split,
                "file_size_bytes": file_size,
                "decodable_seconds": round(float(decodable_sec), 2),
                "reason": reason,
            })

        # Write data_exclusions.csv
        exclusions_df = pd.DataFrame(
            exclusion_rows,
            columns=["track_id", "genre", "split", "file_size_bytes", "decodable_seconds", "reason"],
        )
        exclusions_df.to_csv(exclusions_csv_path, index=False)
        print(f"Wrote {len(exclusions_df)} exclusions to {exclusions_csv_path.relative_to(root) if exclusions_csv_path.is_relative_to(root) else exclusions_csv_path}.")

        # Compute per-split counts: metadata vs manifest
        metadata_splits = {
            s: int((small_all[("set", "split")] == s).sum())
            for s in ["training", "validation", "test"]
        }
        manifest_splits = {
            s: int((manifest_df["split"] == s).sum())
            for s in ["training", "validation", "test"]
        }
        exclusion_splits = {
            s: metadata_splits[s] - manifest_splits.get(s, 0)
            for s in ["training", "validation", "test"]
        }

        # Artist split overlap analysis (PLAN §06 limitation documentation)
        multi_split_artists: Dict[int, List[str]] = {}
        tracks_with_shared_artist = 0
        missing_artist_count = 0
        unique_artists_per_split: Dict[str, int] = {}
        total_unique_artists = 0

        if "artist_id" in manifest_df.columns:
            missing_artist_count = int(manifest_df["artist_id"].isna().sum())
            total_unique_artists = int(manifest_df["artist_id"].nunique(dropna=True))
            for s in ["training", "validation", "test"]:
                split_sub = manifest_df[manifest_df["split"] == s]
                unique_artists_per_split[s] = int(split_sub["artist_id"].nunique(dropna=True))

            artist_groups = (
                manifest_df.dropna(subset=["artist_id"])
                .groupby("artist_id")["split"]
                .unique()
            )
            for a_id, splits in artist_groups.items():
                if len(splits) > 1:
                    multi_split_artists[int(a_id)] = sorted(list(splits))

            if multi_split_artists:
                tracks_with_shared_artist = int(
                    manifest_df["artist_id"].isin(multi_split_artists.keys()).sum()
                )

        if len(multi_split_artists) == 0:
            artist_note = (
                "No artist_id appears in more than one split (measured from FMA metadata artist_id). "
                "Album-level and near-duplicate overlap were not checked."
            )
        else:
            artist_note = (
                "The official FMA Small split is strictly assigned at the track level, "
                "not the artist level. Tracks from the same artist can appear in both training "
                "and evaluation splits. This is a documented dataset limitation."
            )

        audit_report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "min_track_seconds": min_track_seconds,
            "total_small_tracks_in_metadata": total_metadata_tracks,
            "total_small_tracks_in_manifest": len(manifest_df),
            "total_exclusions": len(exclusion_rows),
            "splits_metadata": metadata_splits,
            "splits_manifest": manifest_splits,
            "splits_exclusions": exclusion_splits,
            "exclusions": exclusion_rows,
            "artist_overlap_limitation": {
                "total_artists_in_manifest": total_unique_artists,
                "missing_artist_id_count": missing_artist_count,
                "unique_artists_per_split": unique_artists_per_split,
                "artists_in_multiple_splits_count": len(multi_split_artists),
                "tracks_affected_count": tracks_with_shared_artist,
                "sample_multi_split_artists": dict(list(multi_split_artists.items())[:10]),
                "note": artist_note,
            },
            # Backwards compatibility keys
            "processed_tracks": total_metadata_tracks,
            "valid_tracks": len(manifest_df),
            "excluded_tracks": len(exclusion_rows),
            "splits": manifest_splits,
            "genres": manifest_df["genre"].value_counts().to_dict(),
            "split_genre_matrix": pd.crosstab(manifest_df["split"], manifest_df["genre"]).to_dict(),
        }


        with open(audit_json_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)
        print(f"Saved manifest audit to {audit_json_path.relative_to(root) if audit_json_path.is_relative_to(root) else audit_json_path}.\n")

        _print_audit_summary(audit_report)
        return audit_report

    # ---------------------------------------------------------
    # MODE 2: REBUILD / BUILD MODE
    # ---------------------------------------------------------
    out_dir_path = resolve_path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_csv_path = out_dir_path / "manifest.csv"
    label_map_json_path = out_dir_path / "label_map.json"

    # Build alphabetical label_map.json from all 8 small genres (PLAN §06, Addendum A4)
    all_genres = sorted(small_all[("track", "genre_top")].dropna().unique().tolist())
    label_map = {genre: idx for idx, genre in enumerate(all_genres)}
    with open(label_map_json_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved alphabetical label map with {len(label_map)} genres to {label_map_json_path}.")

    # Apply limit if requested
    if limit is not None and limit > 0:
        small = small_all.iloc[:limit]
        print(f"Processing limited subset of {len(small)} tracks (--limit {limit}).")
    else:
        small = small_all
        print(f"Processing all {len(small)} tracks with min_track_seconds={min_track_seconds}s.")

    # Prepare item specifications
    items_to_verify: List[Dict[str, Any]] = []
    for track_id, row in small.iterrows():
        t_id = int(track_id)
        split = str(row[("set", "split")])
        genre = str(row[("track", "genre_top")])
        raw_artist = row[("artist", "id")]
        artist_id = int(raw_artist) if pd.notna(raw_artist) else None
        rel_path = f"data/raw/fma_small/{t_id // 1000:03d}/{t_id:06d}.mp3"
        full_path = resolve_path(rel_path)

        items_to_verify.append({
            "track_id": t_id,
            "path": rel_path,
            "full_path": full_path,
            "genre": genre,
            "split": split,
            "artist_id": artist_id,
        })

    print(f"Strict decode-testing {len(items_to_verify)} audio files with {workers} worker threads (min_duration={min_track_seconds}s)...")

    valid_rows: List[Dict[str, Any]] = []
    exclusion_rows: List[Dict[str, Any]] = []

    def _check_item(item: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, float, int, Optional[str]]:
        is_valid, dur, size, reason = verify_audio_file_strict(
            item["full_path"], min_duration=min_track_seconds
        )
        return item, is_valid, dur, size, reason

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for item, is_valid, dur, size, reason in executor.map(_check_item, items_to_verify):
            if is_valid:
                valid_rows.append({
                    "track_id": item["track_id"],
                    "path": item["path"],
                    "genre": item["genre"],
                    "split": item["split"],
                    "artist_id": item["artist_id"],
                })
            else:
                exclusion_rows.append({
                    "track_id": item["track_id"],
                    "genre": item["genre"],
                    "split": item["split"],
                    "file_size_bytes": size,
                    "decodable_seconds": round(float(dur), 2),
                    "reason": reason or f"Decodable duration ({dur:.2f}s) < {min_track_seconds:.1f}s",
                })

    # Sort manifest for deterministic output
    valid_rows.sort(key=lambda x: x["track_id"])
    exclusion_rows.sort(key=lambda x: x["track_id"])

    # Write manifest.csv
    manifest_df = pd.DataFrame(
        valid_rows,
        columns=["track_id", "path", "genre", "split", "artist_id"],
    )
    manifest_df.to_csv(manifest_csv_path, index=False)
    print(f"Wrote {len(manifest_df)} valid tracks to {manifest_csv_path}.")

    # Write data_exclusions.csv
    exclusions_df = pd.DataFrame(
        exclusion_rows,
        columns=["track_id", "genre", "split", "file_size_bytes", "decodable_seconds", "reason"],
    )
    exclusions_df.to_csv(exclusions_csv_path, index=False)
    print(f"Logged {len(exclusions_df)} exclusions to {exclusions_csv_path}.")

    # Compute per-split counts
    metadata_splits = {
        s: int((small_all[("set", "split")] == s).sum())
        for s in ["training", "validation", "test"]
    }
    manifest_splits = {
        s: int((manifest_df["split"] == s).sum())
        for s in ["training", "validation", "test"]
    }
    exclusion_splits = {
        s: metadata_splits[s] - manifest_splits.get(s, 0)
        for s in ["training", "validation", "test"]
    }

    # Artist split overlap analysis
    multi_split_artists: Dict[int, List[str]] = {}
    tracks_with_shared_artist = 0
    missing_artist_count = 0
    unique_artists_per_split: Dict[str, int] = {}
    total_unique_artists = 0

    if not manifest_df.empty and "artist_id" in manifest_df.columns:
        missing_artist_count = int(manifest_df["artist_id"].isna().sum())
        total_unique_artists = int(manifest_df["artist_id"].nunique(dropna=True))
        for s in ["training", "validation", "test"]:
            split_sub = manifest_df[manifest_df["split"] == s]
            unique_artists_per_split[s] = int(split_sub["artist_id"].nunique(dropna=True))

        artist_groups = (
            manifest_df.dropna(subset=["artist_id"])
            .groupby("artist_id")["split"]
            .unique()
        )
        for a_id, splits in artist_groups.items():
            if len(splits) > 1:
                multi_split_artists[int(a_id)] = sorted(list(splits))

        if multi_split_artists:
            tracks_with_shared_artist = int(
                manifest_df["artist_id"].isin(multi_split_artists.keys()).sum()
            )

    if len(multi_split_artists) == 0:
        artist_note = (
            "No artist_id appears in more than one split (measured from FMA metadata artist_id). "
            "Album-level and near-duplicate overlap were not checked."
        )
    else:
        artist_note = (
            "The official FMA Small split is strictly assigned at the track level, "
            "not the artist level. Tracks from the same artist can appear in both training "
            "and evaluation splits. This is a documented dataset limitation."
        )

    audit_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "min_track_seconds": min_track_seconds,
        "total_small_tracks_in_metadata": total_metadata_tracks,
        "total_small_tracks_in_manifest": len(manifest_df),
        "total_exclusions": len(exclusion_rows),
        "splits_metadata": metadata_splits,
        "splits_manifest": manifest_splits,
        "splits_exclusions": exclusion_splits,
        "exclusions": exclusion_rows,
        "artist_overlap_limitation": {
            "total_artists_in_manifest": total_unique_artists,
            "missing_artist_id_count": missing_artist_count,
            "unique_artists_per_split": unique_artists_per_split,
            "artists_in_multiple_splits_count": len(multi_split_artists),
            "tracks_affected_count": tracks_with_shared_artist,
            "sample_multi_split_artists": dict(list(multi_split_artists.items())[:10]),
            "note": artist_note,
        },
        # Backwards compatibility keys
        "processed_tracks": len(items_to_verify),
        "valid_tracks": len(manifest_df),
        "excluded_tracks": len(exclusion_rows),
        "splits": manifest_splits,
        "genres": manifest_df["genre"].value_counts().to_dict(),
        "split_genre_matrix": pd.crosstab(manifest_df["split"], manifest_df["genre"]).to_dict() if not manifest_df.empty else {},
    }

    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"Saved manifest audit to {audit_json_path}.\n")

    _print_audit_summary(audit_report)
    return audit_report


def _print_audit_summary(audit: Dict[str, Any]) -> None:
    """Print formatted summary of the manifest audit."""
    print("=" * 64)
    print("             SONICFUSION 2: MANIFEST AUDIT             ")
    print("=" * 64)
    print(f"Metadata tracks : {audit.get('total_small_tracks_in_metadata', 8000)}")
    print(f"Manifest tracks : {audit.get('total_small_tracks_in_manifest', audit.get('valid_tracks', 0))}")
    print(f"Exclusions      : {audit.get('total_exclusions', audit.get('excluded_tracks', 0))}")

    print("\n--- Per-Split Metadata vs. Manifest ---")
    splits_meta = audit.get("splits_metadata", {})
    splits_mani = audit.get("splits_manifest", audit.get("splits", {}))
    splits_excl = audit.get("splits_exclusions", {})
    for s in ["training", "validation", "test"]:
        m_count = splits_meta.get(s, "N/A")
        v_count = splits_mani.get(s, "N/A")
        e_count = splits_excl.get(s, 0)
        print(f"  {s:<12}: Metadata={m_count:<5} | Manifest={v_count:<5} | Excluded={e_count}")

    print("\n--- Exclusions Breakdown ---")
    exclusions = audit.get("exclusions", [])
    if exclusions:
        for ex in exclusions:
            tid = ex.get("track_id")
            genre = ex.get("genre")
            split = ex.get("split")
            sz = ex.get("file_size_bytes")
            dur = ex.get("decodable_seconds")
            reason = ex.get("reason")
            print(f"  Track {tid:<6} ({genre}, {split}): size={sz}B, dur={dur}s -> {reason}")
    else:
        print("  None")

    artist_audit = audit.get("artist_overlap_limitation", {})
    if artist_audit:
        print("\n--- Dataset Limitation: Artist Split Overlap ---")
        print(f"  Total unique artists  : {artist_audit.get('total_artists_in_manifest')}")
        print(f"  Missing artist_id     : {artist_audit.get('missing_artist_id_count', 0)}")
        per_split = artist_audit.get("unique_artists_per_split", {})
        if per_split:
            splits_str = ", ".join(f"{s}: {cnt}" for s, cnt in per_split.items())
            print(f"  Unique artists/split  : {splits_str}")
        print(f"  Multi-split artists   : {artist_audit.get('artists_in_multiple_splits_count')}")
        print(f"  Affected tracks       : {artist_audit.get('tracks_affected_count')}")
        print(f"  Note: {artist_audit.get('note')}")
    print("=" * 64)



if __name__ == "__main__":
    args = parse_args()
    build_manifest(
        out_dir=args.out_dir,
        audit_only=args.audit_only,
        limit=args.limit,
        config_path=args.config,
        workers=args.workers,
        metrics_dir_override=args.metrics_dir,
        manifest_path_override=args.manifest_path,
    )

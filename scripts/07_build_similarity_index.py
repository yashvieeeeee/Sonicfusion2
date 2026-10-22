"""Build content-based song similarity index on FMA catalog tracks.

Follows PLAN §12 and Gate 7 specification:
- Fixed catalog of 1,000–3,000 tracks selected from training/validation splits only (never test).
- Extracts standard 51 acoustic features (MFCC, chroma, spectral metrics, RMS, tempo).
- Fits StandardScaler on catalog tracks only.
- Computes normalized vectors for rapid cosine similarity ranking.
- Evaluates diagnostic retrieval proxy: same-genre precision@5 on held-out validation queries.
- Filters identical track_id to prevent self-retrieval.
- Persists index to data/processed/similarity_index.joblib.
- Persists proxy metrics to reports/metrics/similarity_metrics.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import get_project_root, load_config, resolve_path
from src.features.extraction import FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    """Parse CLI options for building similarity index."""
    parser = argparse.ArgumentParser(
        description="Build nearest-neighbor song similarity index on FMA catalog."
    )
    parser.add_argument(
        "--catalog-size",
        type=int,
        default=2000,
        help="Target number of catalog tracks from train/val splits (1,000 - 3,000).",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features.parquet (defaults to config).",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Path to raw fma_metadata/tracks.csv (defaults to config).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for similarity_index.joblib.",
    )
    return parser.parse_args()


def load_metadata_annotations(metadata_csv: Path) -> pd.DataFrame:
    """Load song titles and artist names from raw fma_metadata/tracks.csv."""
    if not metadata_csv.exists():
        print(f"Warning: Metadata file not found at {metadata_csv}. Using fallback names.")
        return pd.DataFrame(columns=["title", "artist"])

    try:
        raw_tracks = pd.read_csv(metadata_csv, index_col=0, header=[0, 1])
        small = raw_tracks[raw_tracks[("set", "subset")] == "small"]

        meta_df = pd.DataFrame(index=small.index)
        meta_df["title"] = small[("track", "title")].fillna("Unknown Title").astype(str)
        meta_df["artist"] = small[("artist", "name")].fillna("Unknown Artist").astype(str)
        return meta_df
    except Exception as exc:
        print(f"Warning: Failed to load track annotations from {metadata_csv}: {exc}")
        return pd.DataFrame(columns=["title", "artist"])


def evaluate_precision_at_k(
    catalog_vectors: np.ndarray,
    catalog_genres: List[str],
    catalog_tids: List[int],
    query_vectors: np.ndarray,
    query_genres: List[str],
    query_tids: List[int],
    k: int = 5,
) -> Dict[str, Any]:
    """Compute same-genre precision@k diagnostic proxy metric.

    For each query track, retrieves top-k most similar catalog tracks
    (excluding identical track_id to prevent self-retrieval) and computes
    the proportion sharing the same genre label.
    """
    precisions: List[float] = []
    genre_precisions: Dict[str, List[float]] = {}
    sample_queries: List[Dict[str, Any]] = []

    for i in range(len(query_tids)):
        q_vec = query_vectors[i]
        q_tid = query_tids[i]
        q_genre = query_genres[i]

        # Cosine similarity against catalog (dot product of unit normalized vectors)
        sims = np.dot(catalog_vectors, q_vec)

        # Mask out query track if it exists in the catalog (prevent self-retrieval)
        for cat_idx, cat_tid in enumerate(catalog_tids):
            if cat_tid == q_tid:
                sims[cat_idx] = -1.0  # Lowest possible cosine score

        # Top-k indices
        top_k_indices = np.argsort(sims)[::-1][:k]
        top_k_genres = [catalog_genres[idx] for idx in top_k_indices]
        top_k_scores = [float(sims[idx]) for idx in top_k_indices]

        # Calculate same-genre precision
        matches = sum(1 for g in top_k_genres if g == q_genre)
        p_at_k = matches / k
        precisions.append(p_at_k)

        if q_genre not in genre_precisions:
            genre_precisions[q_genre] = []
        genre_precisions[q_genre].append(p_at_k)

        # Collect sample queries for qualitative verification
        if len(sample_queries) < 5:
            sample_queries.append({
                "query_track_id": int(q_tid),
                "query_genre": q_genre,
                "precision_at_k": round(p_at_k, 2),
                "top_retrievals": [
                    {
                        "track_id": int(catalog_tids[idx]),
                        "genre": catalog_genres[idx],
                        "similarity": round(top_k_scores[rank], 4),
                        "genre_match": bool(catalog_genres[idx] == q_genre),
                    }
                    for rank, idx in enumerate(top_k_indices)
                ],
            })

    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    per_genre_means = {
        g: round(float(np.mean(vals)), 4) for g, vals in sorted(genre_precisions.items())
    }

    return {
        "k": k,
        "queries_count": len(query_tids),
        "mean_same_genre_precision_at_5": round(mean_precision, 4),
        "per_genre_precision_at_5": per_genre_means,
        "sample_queries": sample_queries,
    }


def build_similarity_index(
    catalog_size: int = 2000,
    features_path_override: Optional[str | Path] = None,
    metadata_path_override: Optional[str | Path] = None,
    output_path_override: Optional[str | Path] = None,
    metrics_path_override: Optional[str | Path] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build standardized nearest-neighbor similarity index and evaluation report."""
    cfg = load_config(config_path)
    root = get_project_root()

    features_path = (
        resolve_path(features_path_override)
        if features_path_override
        else resolve_path(cfg.paths.features_path)
    )
    metadata_path = (
        resolve_path(metadata_path_override)
        if metadata_path_override
        else resolve_path(cfg.paths.fma_metadata_dir) / "tracks.csv"
    )
    output_path = (
        resolve_path(output_path_override)
        if output_path_override
        else resolve_path(cfg.paths.similarity_index_path)
    )
    metrics_path = (
        resolve_path(metrics_path_override)
        if metrics_path_override
        else resolve_path(cfg.paths.metrics_dir) / "similarity_metrics.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features parquet not found at: {features_path}. "
            "Please run scripts/02_extract_features.py first."
        )

    print(f"Loading features from {features_path.relative_to(root)}...")
    df = pd.read_parquet(features_path)

    # Validate feature columns
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            raise KeyError(f"Missing required feature column: {feat}")

    # Enforce catalog selection strictly from training and validation (PLAN §12, Addendum A5)
    eligible_df = df[df["split"].isin(["training", "validation"])].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Catalog candidate pool: {len(eligible_df)} tracks (training + validation).")
    print(f"Test split tracks: {len(test_df)} (EXCLUDED: Never added to similarity index).")

    # Select bounded catalog: 1,000 <= catalog_size <= 3,000
    catalog_target = max(1000, min(catalog_size, 3000, len(eligible_df)))

    # Balance across genres evenly where possible
    genres = eligible_df["genre"].unique()
    per_genre_target = catalog_target // len(genres)

    selected_dfs = []
    for g in genres:
        g_slice = eligible_df[eligible_df["genre"] == g]
        take_count = min(len(g_slice), per_genre_target)
        selected_dfs.append(g_slice.iloc[:take_count])

    catalog_df = pd.concat(selected_dfs).reset_index(drop=True)

    # If any remainder needed to reach target, fill from remaining eligible tracks
    if len(catalog_df) < catalog_target:
        remaining = eligible_df[~eligible_df["track_id"].isin(catalog_df["track_id"])]
        fill_count = catalog_target - len(catalog_df)
        catalog_df = pd.concat([catalog_df, remaining.iloc[:fill_count]]).reset_index(drop=True)

    print(f"\nIndexed catalog contains {len(catalog_df)} tracks across {len(genres)} genres.")

    # Load title and artist metadata
    print(f"Loading track title and artist annotations from {metadata_path.name}...")
    meta_df = load_metadata_annotations(metadata_path)

    catalog_records: List[Dict[str, Any]] = []
    for _, row in catalog_df.iterrows():
        t_id = int(row["track_id"])
        genre = str(row["genre"])
        title = meta_df.loc[t_id, "title"] if t_id in meta_df.index else f"FMA Track {t_id:06d}"
        artist = meta_df.loc[t_id, "artist"] if t_id in meta_df.index else "Unknown Artist"
        tempo_val = float(row.get("tempo", 120.0))

        catalog_records.append({
            "track_id": t_id,
            "title": title,
            "artist": artist,
            "genre": genre,
            "duration": 30.0,
            "tempo": tempo_val,
        })

    # Fit scaler ON CATALOG TRACKS ONLY (PLAN §12 specification)
    X_catalog = catalog_df[FEATURE_NAMES].values.astype(np.float32)
    scaler = StandardScaler()
    X_catalog_scaled = scaler.fit_transform(X_catalog)

    # Compute unit-norm vectors for exact cosine similarity dot products
    norms = np.linalg.norm(X_catalog_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    catalog_vectors = (X_catalog_scaled / norms).astype(np.float32)

    # -------------------------------------------------------------
    # Proxy evaluation: same-genre precision@5 on validation queries
    # -------------------------------------------------------------
    val_df = df[df["split"] == "validation"].copy()
    print(f"\nEvaluating similarity retrieval on {len(val_df)} validation tracks (proxy metric)...")

    X_val = val_df[FEATURE_NAMES].values.astype(np.float32)
    X_val_scaled = scaler.transform(X_val)
    val_norms = np.linalg.norm(X_val_scaled, axis=1, keepdims=True)
    val_norms[val_norms == 0] = 1e-8
    val_vectors = (X_val_scaled / val_norms).astype(np.float32)

    val_tids = val_df["track_id"].astype(int).tolist()
    val_genres = val_df["genre"].astype(str).tolist()
    cat_tids = [r["track_id"] for r in catalog_records]
    cat_genres = [r["genre"] for r in catalog_records]

    proxy_metrics = evaluate_precision_at_k(
        catalog_vectors=catalog_vectors,
        catalog_genres=cat_genres,
        catalog_tids=cat_tids,
        query_vectors=val_vectors,
        query_genres=val_genres,
        query_tids=val_tids,
        k=5,
    )

    print(f"Results for same-genre Precision@5 (proxy diagnostic):")
    print(f"  Overall Mean Precision@5: {proxy_metrics['mean_same_genre_precision_at_5'] * 100:.2f}%")
    for g, score in proxy_metrics["per_genre_precision_at_5"].items():
        print(f"    - {g:<15}: {score * 100:.1f}%")

    # Persist index artifact
    index_payload = {
        "catalog_size": len(catalog_records),
        "feature_names": FEATURE_NAMES,
        "scaler": scaler,
        "catalog_vectors": catalog_vectors,
        "catalog_metadata": catalog_records,
        "track_id_to_idx": {r["track_id"]: idx for idx, r in enumerate(catalog_records)},
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(index_payload, output_path)
    try:
        out_display = output_path.relative_to(root)
    except ValueError:
        out_display = output_path
    print(f"\nSaved similarity index artifact to {out_display}.")

    # Persist evaluation metrics JSON
    metrics_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "catalog_size": len(catalog_records),
        "similarity_metric": "Cosine Similarity",
        "features_dimension": len(FEATURE_NAMES),
        "evaluation_protocol": "Proxy diagnostic: Same-genre precision@5 on held-out validation tracks with self-exclusion",
        "mean_same_genre_precision_at_5": proxy_metrics["mean_same_genre_precision_at_5"],
        "per_genre_precision_at_5": proxy_metrics["per_genre_precision_at_5"],
        "sample_retrievals": proxy_metrics["sample_queries"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    try:
        metrics_display = metrics_path.relative_to(root)
    except ValueError:
        metrics_display = metrics_path
    print(f"Saved similarity proxy metrics to {metrics_display}.")

    return index_payload


if __name__ == "__main__":
    args = parse_args()
    build_similarity_index(
        catalog_size=args.catalog_size,
        features_path_override=args.features_path,
        metadata_path_override=args.metadata_path,
        output_path_override=args.output_path,
    )

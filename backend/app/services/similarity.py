"""Song similarity service using cosine distance over standardized FMA catalog features.

Follows PLAN §12 and Gate 7 specification:
- Indexes 1,000 to 3,000 catalog tracks from training/validation splits only.
- Computes cosine similarity against fitted StandardScaler representation.
- Filters out identical track_id to prevent self-retrieval.
- Returns top-5 similar songs with similarity scores and comparative explanations.
- If the index is missing, returns status 'unavailable' - NEVER returns fake/mocked results.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np

from src.config import get_project_root, load_config, resolve_path
from src.features.extraction import FEATURE_NAMES


class SimilarityService:
    """Singleton service managing catalog nearest-neighbor similarity retrieval."""

    def __init__(self, index_path: Optional[str | Path] = None) -> None:
        self.cfg = load_config()
        self.index_path = (
            resolve_path(index_path)
            if index_path
            else resolve_path(self.cfg.paths.similarity_index_path)
        )
        self.is_ready: bool = False
        self._load_error: Optional[str] = None

        self.catalog_size: int = 0
        self.feature_names: List[str] = []
        self.scaler: Any = None
        self.catalog_vectors: Optional[np.ndarray] = None
        self.catalog_metadata: List[Dict[str, Any]] = []
        self.track_id_to_idx: Dict[int, int] = {}

        self._load_index()

    def _load_index(self) -> None:
        """Attempt to load the similarity index joblib artifact."""
        if not self.index_path.exists():
            self._load_error = (
                f"Similarity index not found at: {self.index_path}. "
                "Run scripts/07_build_similarity_index.py to generate it."
            )
            self.is_ready = False
            return

        try:
            payload = joblib.load(self.index_path)
            self.catalog_size = int(payload.get("catalog_size", 0))
            self.feature_names = list(payload.get("feature_names", FEATURE_NAMES))
            self.scaler = payload.get("scaler")
            self.catalog_vectors = payload.get("catalog_vectors")
            self.catalog_metadata = list(payload.get("catalog_metadata", []))
            self.track_id_to_idx = dict(payload.get("track_id_to_idx", {}))

            self.is_ready = True
            self._load_error = None
        except Exception as exc:
            self._load_error = f"Failed to load similarity index artifact: {exc}"
            self.is_ready = False

    def is_available(self) -> bool:
        """Return True if index is loaded and ready for retrieval queries."""
        return self.is_ready

    def get_load_error(self) -> Optional[str]:
        """Return diagnostic error message if index is unavailable."""
        return self._load_error

    def reload(self) -> bool:
        """Reload similarity index from disk."""
        self._load_index()
        return self.is_available()

    def get_catalog_track(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve approved display metadata for a catalog track by ID."""
        if not self.is_ready or track_id not in self.track_id_to_idx:
            return None
        idx = self.track_id_to_idx[track_id]
        return self.catalog_metadata[idx]

    def _generate_explanation(
        self, query_feats: Dict[str, float], cat_record: Dict[str, Any], sim_score: float
    ) -> str:
        """Generate human-interpretable rationale for why songs appear similar."""
        q_tempo = query_feats.get("tempo", 0.0)
        c_tempo = float(cat_record.get("tempo", 0.0))
        tempo_diff = abs(q_tempo - c_tempo)

        notes: List[str] = []

        # Rhythmic tempo proximity
        if q_tempo > 0 and c_tempo > 0 and tempo_diff <= 8.0:
            notes.append(f"matching tempo (~{c_tempo:.0f} BPM)")
        elif q_tempo > 0 and c_tempo > 0 and tempo_diff <= 18.0:
            notes.append(f"close tempo profile ({c_tempo:.0f} vs {q_tempo:.0f} BPM)")

        # Genre alignment
        q_genre = query_feats.get("genre", "")
        c_genre = cat_record.get("genre", "")
        if q_genre and c_genre and q_genre == c_genre:
            notes.append(f"shared {c_genre} genre styling")

        # Fallback timbre / dynamics description
        if not notes:
            if sim_score >= 0.85:
                notes.append("high timbral texture and spectral envelope correspondence")
            elif sim_score >= 0.70:
                notes.append("comparable acoustic energy and harmonic pitch characteristics")
            else:
                notes.append("moderate acoustic feature proximity")

        return " & ".join(notes).capitalize()

    def query_similar(
        self,
        features: Union[Dict[str, float], np.ndarray],
        query_track_id: Optional[int] = None,
        top_k: int = 5,
        filter_exact_matches: bool = False,
        exact_match_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Query top-k nearest catalog tracks using cosine similarity.

        Args:
            features: Dictionary or 1D array of extracted acoustic features.
            query_track_id: Optional track ID of the query song to prevent self-retrieval.
            top_k: Number of nearest neighbors to retrieve (default 5).
            filter_exact_matches: Whether to exclude exact matches with similarity >= threshold (for uploads).
            exact_match_threshold: Cosine similarity cutoff for exact match exclusion (default 0.999 from config).

        Returns:
            Dictionary with status, similar_tracks list, and identical_catalog_track_excluded boolean.
            Returns status 'unavailable' if index is not built - NEVER fakes results.
        """
        # Strict requirement: If index missing, return unavailable, never fake results
        if not self.is_available() or self.catalog_vectors is None or self.scaler is None:
            return {
                "status": "unavailable",
                "similar_tracks": [],
                "identical_catalog_track_excluded": False,
                "detail": self._load_error or "Similarity index not loaded.",
            }

        # Convert input features to ordered 1D vector
        if isinstance(features, dict):
            feat_dict = features
            vec = np.array([float(features.get(f, 0.0)) for f in self.feature_names], dtype=np.float32)
        else:
            feat_dict = {}
            vec = np.asarray(features, dtype=np.float32)
            if len(vec) != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, got {len(vec)}."
                )

        # Handle any NaN/Inf in query vector
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale with scaler fitted ON CATALOG ONLY
        scaled_vec = self.scaler.transform([vec])[0]

        # Normalize to unit length for cosine dot product
        norm = float(np.linalg.norm(scaled_vec))
        if norm <= 0:
            norm = 1e-8
        unit_vec = (scaled_vec / norm).astype(np.float32)

        # Compute cosine similarity against all catalog tracks
        sims = np.dot(self.catalog_vectors, unit_vec)

        # Filter out self-retrieval if query_track_id matches a catalog entry (catalog-query mode)
        if query_track_id is not None and query_track_id in self.track_id_to_idx:
            self_idx = self.track_id_to_idx[query_track_id]
            sims[self_idx] = -1.0  # Lowest score to exclude

        # Rank indices
        ranked_indices = np.argsort(sims)[::-1]

        thresh = (
            float(exact_match_threshold)
            if exact_match_threshold is not None
            else float(self.cfg.get("similarity", {}).get("exact_match_threshold", 0.999))
        )

        similar_tracks: List[Dict[str, Any]] = []
        identical_catalog_track_excluded = False

        for idx in ranked_indices:
            raw_score = float(sims[idx])
            # If all valid candidates are exhausted, stop
            if raw_score <= -1.0:
                break

            # Filter exact matches for uploads: drop and fill from next candidates
            if filter_exact_matches and raw_score >= thresh:
                identical_catalog_track_excluded = True
                continue

            record = self.catalog_metadata[idx]
            # Normalise cosine score [-1, 1] to a readable [0, 1] range
            normalized_similarity = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
            explanation = self._generate_explanation(feat_dict, record, normalized_similarity)

            similar_tracks.append({
                "track_id": int(record["track_id"]),
                "title": str(record.get("title", f"Track {record['track_id']}")),
                "artist": str(record.get("artist", "Unknown Artist")),
                "genre": str(record.get("genre", "Unknown")),
                "feature_similarity_score": round(normalized_similarity, 4),
                "explanation": explanation,
            })

            if len(similar_tracks) == top_k:
                break

        return {
            "status": "available",
            "catalog_size": self.catalog_size,
            "similar_tracks": similar_tracks,
            "identical_catalog_track_excluded": identical_catalog_track_excluded,
        }


# Global singleton instance
_similarity_service: Optional[SimilarityService] = None


def get_similarity_service(index_path: Optional[str | Path] = None) -> SimilarityService:
    """Return global singleton SimilarityService instance."""
    global _similarity_service
    if _similarity_service is None or index_path is not None:
        _similarity_service = SimilarityService(index_path=index_path)
    return _similarity_service

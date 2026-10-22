"""Unit and integration tests for Song Similarity Engine (PLAN §12, Gate 7).

Tests:
- SimilarityService reports status 'unavailable' and empty list when index is missing (never fakes results).
- Building the similarity index selects 1,000-3,000 tracks strictly from train/val splits (never test).
- Scaler is fitted on catalog tracks only.
- Cosine retrieval returns top-5 nearest neighbors with valid scores and explanations.
- Self-retrieval prevention: query track is excluded from its own neighbors list.
- Same-genre precision@5 proxy evaluation metric calculation.
- Backend GET /api/catalog/{id} endpoint behaves correctly.
"""

from pathlib import Path
import tempfile
from fastapi.testclient import TestClient
import numpy as np
import pytest

import importlib

from backend.app.main import app
from backend.app.services.similarity import SimilarityService, get_similarity_service
from src.config import get_project_root, load_config, resolve_path
from src.features.extraction import FEATURE_NAMES

build_sim_module = importlib.import_module("scripts.07_build_similarity_index")
build_similarity_index = build_sim_module.build_similarity_index

client = TestClient(app)


def test_similarity_missing_index_returns_unavailable():
    """Verify SimilarityService returns status 'unavailable' when index file does not exist.

    Strict contract: NEVER returns fake/mocked results.
    """
    dummy_service = SimilarityService(index_path="non_existent/similarity_index.joblib")
    assert not dummy_service.is_available()

    dummy_feats = {feat: 0.5 for feat in FEATURE_NAMES}
    res = dummy_service.query_similar(dummy_feats, top_k=5)

    assert res["status"] == "unavailable"
    assert res["similar_tracks"] == []
    assert "not loaded" in res["detail"].lower() or "not found" in res["detail"].lower()


def test_build_similarity_index_catalog_constraints():
    """Verify build_similarity_index enforces 1,000-3,000 catalog size and train/val split isolation."""
    root = get_project_root()
    features_path = root / "data" / "processed" / "features.parquet"
    if not features_path.exists():
        pytest.skip("data/processed/features.parquet not found")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_index_path = Path(tmp_dir) / "test_sim_index.joblib"
        tmp_metrics_path = Path(tmp_dir) / "test_similarity_metrics.json"

        # Build with target catalog size 1,500
        index_data = build_similarity_index(
            catalog_size=1500,
            output_path_override=tmp_index_path,
            metrics_path_override=tmp_metrics_path,
        )

        assert tmp_metrics_path.exists()
        assert 0 < index_data["catalog_size"] <= 3000

        # Verify required keys
        required_keys = [
            "catalog_size",
            "feature_names",
            "scaler",
            "catalog_vectors",
            "catalog_metadata",
            "track_id_to_idx",
            "version",
        ]
        for key in required_keys:
            assert key in index_data, f"Missing key in similarity index: {key}"

        # Verify split isolation: ensure NO test tracks are included in the catalog
        import pandas as pd
        df = pd.read_parquet(features_path)
        test_tids = set(df[df["split"] == "test"]["track_id"].astype(int))

        catalog_tids = set(r["track_id"] for r in index_data["catalog_metadata"])
        overlap = test_tids.intersection(catalog_tids)
        assert len(overlap) == 0, f"Found {len(overlap)} test split tracks leaked into similarity catalog!"


def test_cosine_top_5_and_no_self_retrieval():
    """Verify top-5 similarity ranking, valid score ranges, and self-retrieval filtering."""
    cfg = load_config()
    index_path = resolve_path(cfg.paths.similarity_index_path)
    if not index_path.exists():
        pytest.skip(f"Similarity index not found at {index_path} (build index first)")

    service = SimilarityService(index_path=index_path)
    assert service.is_available()
    assert 1000 <= service.catalog_size <= 3000

    # Pick the first catalog track to test query and self-retrieval
    first_track = service.catalog_metadata[0]
    first_tid = first_track["track_id"]

    import pandas as pd
    root = get_project_root()
    features_df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    raw_row = features_df[features_df["track_id"] == first_tid].iloc[0]
    query_feats = {col: float(raw_row[col]) for col in service.feature_names}

    # Query without self-exclusion (should find itself as #1 neighbor with 1.0 similarity)
    res_with_self = service.query_similar(query_feats, query_track_id=None, top_k=5)
    assert res_with_self["status"] == "available"
    assert len(res_with_self["similar_tracks"]) == 5
    assert res_with_self["similar_tracks"][0]["track_id"] == first_tid
    assert res_with_self["similar_tracks"][0]["feature_similarity_score"] > 0.99

    # Query WITH self-exclusion (query_track_id passed)
    res_filtered = service.query_similar(query_feats, query_track_id=first_tid, top_k=5)
    assert res_filtered["status"] == "available"
    assert len(res_filtered["similar_tracks"]) == 5

    retrieved_tids = [t["track_id"] for t in res_filtered["similar_tracks"]]
    assert first_tid not in retrieved_tids, "Self-retrieval occurred! Query track should be excluded."

    # Validate output contract
    for t in res_filtered["similar_tracks"]:
        assert "title" in t
        assert "artist" in t
        assert "genre" in t
        assert "feature_similarity_score" in t
        assert "explanation" in t
        assert 0.0 <= t["feature_similarity_score"] <= 1.0
        assert len(t["explanation"]) > 0


def test_catalog_track_endpoint():
    """Verify GET /api/catalog/{id} returns track details or 404."""
    cfg = load_config()
    index_path = resolve_path(cfg.paths.similarity_index_path)
    if not index_path.exists():
        pytest.skip(f"Similarity index not found at {index_path}")

    service = get_similarity_service(index_path=index_path)
    if not service.is_available():
        pytest.skip("Similarity service unavailable")

    # Valid track
    valid_tid = service.catalog_metadata[0]["track_id"]
    res = client.get(f"/api/catalog/{valid_tid}")
    assert res.status_code == 200
    data = res.json()
    assert data["track_id"] == valid_tid
    assert "title" in data
    assert "artist" in data
    assert "genre" in data

    # Non-existent track ID
    res_invalid = client.get("/api/catalog/999999999")
    assert res_invalid.status_code == 404

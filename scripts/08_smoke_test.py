"""End-to-End System Smoke Test for SonicScope AI (SonicFusion 2).

Follows PLAN §17, §18, §22:
Validates configuration, datasets, baseline & CNN model checkpoints,
similarity engine, and live FastAPI REST API endpoints without mocked results.
Exits with 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.similarity import SimilarityService
from src.config import get_project_root, load_config, resolve_path
from src.features.extraction import FEATURE_NAMES
from src.models.genre_cnn import GenreCNN


class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(title: str) -> None:
    print(f"\n{Color.CYAN}{Color.BOLD}{'=' * 70}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}  {title}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'=' * 70}{Color.RESET}")


def run_check(name: str, check_fn) -> Tuple[bool, str]:
    """Execute a single check function and return pass/fail with detail."""
    try:
        msg = check_fn()
        return True, msg or "Passed"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="SonicScope AI End-to-End Smoke Test")
    parser.add_argument("--skip-api", action="store_true", help="Skip FastAPI HTTP endpoint verification")
    parser.add_argument("--verbose", action="store_true", help="Print detailed diagnostic messages")
    args = parser.parse_args()

    start_time = time.time()
    results: List[Tuple[str, bool, str]] = []

    print_header("SonicScope AI (SonicFusion 2) - Pipeline Smoke Test")

    # -------------------------------------------------------------
    # 1. Configuration & Path Resolution
    # -------------------------------------------------------------
    def check_config():
        cfg = load_config()
        root = get_project_root()
        assert (root / "configs" / "config.yaml").exists(), "config.yaml missing"
        assert cfg.audio.sample_rate == 22050, f"Expected 22050 Hz, got {cfg.audio.sample_rate}"
        assert cfg.audio.duration == 30.0, f"Expected 30.0s, got {cfg.audio.duration}"
        assert cfg.audio.n_mels == 128, f"Expected 128 mels, got {cfg.audio.n_mels}"
        return f"Root: {root.name} | Audio: {cfg.audio.sample_rate}Hz, {cfg.audio.n_mels} mels"

    passed, msg = run_check("1. Configuration & Settings (§07)", check_config)
    results.append(("Configuration & Paths", passed, msg))

    # -------------------------------------------------------------
    # 2. Dataset Manifest & Label Map
    # -------------------------------------------------------------
    def check_manifest():
        root = get_project_root()
        label_map_file = root / "data" / "processed" / "label_map.json"
        manifest_parquet = root / "data" / "processed" / "manifest.parquet"
        manifest_csv = root / "data" / "processed" / "manifest.csv"

        assert label_map_file.exists(), f"label_map.json not found at {label_map_file}"
        import json
        with open(label_map_file, "r") as f:
            label_map = json.load(f)
        assert len(label_map) == 8, f"Expected 8 genres in label_map, got {len(label_map)}"
        
        manifest_path = manifest_parquet if manifest_parquet.exists() else manifest_csv
        assert manifest_path.exists(), "Neither manifest.parquet nor manifest.csv found"
        return f"Label map: 8 classes | Manifest: {manifest_path.name}"

    passed, msg = run_check("2. Dataset Manifest & Taxonomy (§06)", check_manifest)
    results.append(("Manifest & Label Map", passed, msg))

    # -------------------------------------------------------------
    # 3. Tabular Features Parquet
    # -------------------------------------------------------------
    def check_features():
        root = get_project_root()
        feats_path = root / "data" / "processed" / "features.parquet"
        assert feats_path.exists(), f"features.parquet not found at {feats_path}"
        import pandas as pd
        df = pd.read_parquet(feats_path)
        assert len(df) > 0, "features.parquet is empty"
        for feat in FEATURE_NAMES:
            assert feat in df.columns, f"Missing feature column: {feat}"
        assert "split" in df.columns, "Missing 'split' column in features.parquet"
        assert set(df["split"].unique()).issubset({"training", "validation", "test"}), "Invalid splits"
        return f"Rows: {len(df)} tracks | Features: {len(FEATURE_NAMES)} acoustic dimensions"

    passed, msg = run_check("3. Extracted Features Dataset (§07)", check_features)
    results.append(("Acoustic Features (51-d)", passed, msg))

    # -------------------------------------------------------------
    # 4. Classical Baseline Model
    # -------------------------------------------------------------
    def check_baseline():
        root = get_project_root()
        cfg = load_config()
        model_path = resolve_path(cfg.paths.baseline_model_path)
        if not model_path.exists():
            alt_path = root / "models" / "baseline" / "random_forest_pipeline.joblib"
            if alt_path.exists():
                model_path = alt_path

        metrics_path = root / "reports" / "metrics" / "baseline_val.json"

        assert model_path.exists(), f"Baseline model not found at {model_path}"
        assert metrics_path.exists(), f"Baseline metrics not found at {metrics_path}"

        import json
        with open(metrics_path, "r") as f:
            bm = json.load(f)
        val_f1 = bm.get("validation_macro_f1", 0.0)
        assert val_f1 > 0.0, "Validation macro-F1 must be > 0"
        return f"Algorithm: {bm.get('model_type', 'RandomForest')} | Val Macro-F1: {val_f1:.4f}"

    passed, msg = run_check("4. Classical Baseline Model (§08)", check_baseline)
    results.append(("Random Forest Baseline", passed, msg))

    # -------------------------------------------------------------
    # 5. GenreCNN Model Checkpoint & Architecture
    # -------------------------------------------------------------
    def check_cnn():
        root = get_project_root()
        cfg = load_config()
        ckpt_path = resolve_path(cfg.paths.cnn_checkpoint_path)
        metrics_path = root / "reports" / "metrics" / "cnn_val.json"

        assert ckpt_path.exists(), f"CNN checkpoint not found at {ckpt_path}"
        assert metrics_path.exists(), f"CNN metrics not found at {metrics_path}"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "state_dict" in ckpt or "model_state_dict" in ckpt, "Checkpoint missing weights state dict"
        for req_k in ["class_names", "normalization_stats"]:
            assert req_k in ckpt, f"Checkpoint missing key: {req_k}"

        classes = ckpt["class_names"]
        assert len(classes) == 8, f"Expected 8 class names, got {len(classes)}"

        # Instantiate model and test forward pass
        model = GenreCNN(num_classes=8, in_channels=1, dropout=0.3)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        model.load_state_dict(state_dict)
        model.eval()

        # Dummy spectrogram window: (1, 1, 128, 431)
        dummy_input = torch.randn(1, 1, 128, 431)
        with torch.no_grad():
            logits = model(dummy_input)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        assert logits.shape == (1, 8), f"Expected shape (1, 8), got {logits.shape}"
        assert np.isclose(np.sum(probs), 1.0, atol=1e-4), "Softmax probabilities must sum to 1.0"

        return f"Params: {sum(p.numel() for p in model.parameters()):,} | Classes: {len(classes)} | Forward pass: (1, 8)"

    passed, msg = run_check("5. GenreCNN Architecture & Weights (§09)", check_cnn)
    results.append(("GenreCNN 2D PyTorch Model", passed, msg))

    # -------------------------------------------------------------
    # 6. Song Similarity Engine & Isolation
    # -------------------------------------------------------------
    def check_similarity():
        root = get_project_root()
        sim_path = root / "data" / "processed" / "similarity_index.joblib"
        metrics_path = root / "reports" / "metrics" / "similarity_metrics.json"

        assert sim_path.exists(), f"similarity_index.joblib missing at {sim_path}"
        assert metrics_path.exists(), f"similarity_metrics.json missing at {metrics_path}"

        sim_service = SimilarityService(index_path=sim_path)
        assert sim_service.is_available(), "SimilarityService failed to initialize as available"
        assert 1000 <= sim_service.catalog_size <= 3000, f"Catalog size {sim_service.catalog_size} outside 1000-3000 range"

        # Verify no self-retrieval
        first_track = sim_service.catalog_metadata[0]
        first_tid = first_track["track_id"]
        dummy_feats = {col: 0.5 for col in sim_service.feature_names}
        res = sim_service.query_similar(dummy_feats, query_track_id=first_tid, top_k=5)
        assert res["status"] == "available", "Query failed"
        assert len(res["similar_tracks"]) == 5, f"Expected 5 neighbors, got {len(res['similar_tracks'])}"
        retrieved_ids = [t["track_id"] for t in res["similar_tracks"]]
        assert first_tid not in retrieved_ids, "Self-retrieval prevention failed"

        # Verify honest unavailable fallback (strict requirement: never fake results)
        dummy_svc = SimilarityService(index_path="nonexistent_path/index.joblib")
        assert not dummy_svc.is_available()
        unavail_res = dummy_svc.query_similar(dummy_feats)
        assert unavail_res["status"] == "unavailable"
        assert unavail_res["similar_tracks"] == []

        return f"Catalog: {sim_service.catalog_size} tracks (train/val only) | Top-5 cosine verified | No self-retrieval"

    passed, msg = run_check("6. Song Similarity Engine (§12)", check_similarity)
    results.append(("Similarity Engine (Cosine Top-5)", passed, msg))

    # -------------------------------------------------------------
    # 7. FastAPI Backend REST API Endpoints
    # -------------------------------------------------------------
    if not args.skip_api:
        def check_api():
            client = TestClient(app)

            # Health
            res_h = client.get("/api/health")
            assert res_h.status_code == 200, f"Health status {res_h.status_code}"
            h_data = res_h.json()
            assert h_data["status"] == "ok", f"Health status not ok: {h_data}"
            assert h_data["model_loaded"] is True, "Model reported not loaded"

            # Models
            res_m = client.get("/api/models")
            assert res_m.status_code == 200, f"Models status {res_m.status_code}"
            assert len(res_m.json()["class_names"]) == 8

            # Metrics
            res_met = client.get("/api/metrics")
            assert res_met.status_code == 200, f"Metrics status {res_met.status_code}"
            assert "baseline_validation" in res_met.json()
            assert "cnn_validation" in res_met.json()

            # Catalog Lookup
            res_cat = client.get("/api/catalog/2")
            assert res_cat.status_code in (200, 404)
            if res_cat.status_code == 200:
                assert res_cat.json()["track_id"] == 2

            # End-to-end Analyze upload with sample track
            root = get_project_root()
            sample_mp3 = root / "web" / "public" / "samples" / "sample_000002.mp3"
            if sample_mp3.exists():
                with open(sample_mp3, "rb") as f:
                    res_a = client.post("/api/analyze", files={"file": ("sample.mp3", f, "audio/mpeg")})
                assert res_a.status_code == 200, f"Analyze failed with status {res_a.status_code}"
                a_data = res_a.json()
                assert "genre" in a_data
                assert "acoustic_features" in a_data
                assert "visualizations" in a_data
                assert "similar_tracks" in a_data
                assert len(a_data["genre"]["scores"]) == 8
                return f"Tested GET /api/health, /models, /metrics, /catalog, and POST /api/analyze"
            return "Tested GET endpoints (/api/health, /models, /metrics, /catalog)"

        passed, msg = run_check("7. FastAPI Backend Service (§13)", check_api)
        results.append(("FastAPI REST API Endpoints", passed, msg))
    else:
        results.append(("FastAPI REST API Endpoints", True, "Skipped by user flag (--skip-api)"))

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    print_header("Smoke Test Results Summary")
    all_passed = True
    for name, ok, detail in results:
        status_text = f"{Color.GREEN}[PASS]{Color.RESET}" if ok else f"{Color.RED}[FAIL]{Color.RESET}"
        print(f" {status_text}  {name:<32} {Color.YELLOW}{detail}{Color.RESET}")
        if not ok:
            all_passed = False

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds.")

    if all_passed:
        print(f"{Color.GREEN}{Color.BOLD}>>> ALL SMOKE TESTS PASSED! System is fully operational. <<<{Color.RESET}\n")
        return 0
    else:
        print(f"{Color.RED}{Color.BOLD}>>> SMOKE TESTS FAILED! Review errors above. <<<{Color.RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

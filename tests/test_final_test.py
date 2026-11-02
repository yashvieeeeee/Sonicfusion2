"""Unit tests for the one-time final test evaluation output."""

import json
from pathlib import Path
import pytest


def test_final_test_json_artifact():
    """Verify that final_test.json exists, contains all models and ensembles, and conforms to schema."""
    metrics_path = Path("reports/metrics/final_test.json")
    if not metrics_path.exists():
        pytest.skip("final_test.json not yet generated")

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "models" in data
    assert data["evaluated_tracks"] == 800

    models = data["models"]
    for expected_key in [
        "random_forest",
        "rbf_svc",
        "extra_trees",
        "genre_cnn",
        "ensemble_rf_svc_cnn",
    ]:
        assert expected_key in models
        m = models[expected_key]
        assert "test_accuracy" in m
        assert "test_macro_f1" in m
        assert "test_top_3_accuracy" in m
        assert 0.0 <= m["test_accuracy"] <= 1.0
        assert 0.0 <= m["test_macro_f1"] <= 1.0
        assert 0.0 <= m["test_top_3_accuracy"] <= 1.0

    # Verify ensemble weights
    ens = models["ensemble_rf_svc_cnn"]
    assert ens["weights"]["svc"] == 0.40
    assert ens["weights"]["cnn"] == 0.60
    assert ens["weights"]["rf"] == 0.00


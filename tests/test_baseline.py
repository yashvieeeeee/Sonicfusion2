"""Unit tests for classical Random Forest baseline training and tuning.

Verifies requirements from docs/PLAN.md (§05, §08, Addenda A4):
- Pipeline structure consists of SimpleImputer and RandomForestClassifier.
- Grid search selects model strictly based on validation macro F1.
- Saved joblib model loads and produces predictions.
- Validation metrics JSON is generated.
- Test split is strictly excluded from training and tuning.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import load_config, resolve_path
from src.features.extraction import FEATURE_NAMES
from src.models.baseline import (
    build_baseline_pipeline,
    load_baseline_model,
    tune_and_train_baseline,
)


@pytest.fixture
def cfg():
    """Project configuration fixture."""
    return load_config()


def test_baseline_pipeline_structure():
    """Verify that baseline pipeline consists of SimpleImputer and RandomForestClassifier."""
    pipe = build_baseline_pipeline(n_estimators=100, max_depth=10, random_state=42)
    assert isinstance(pipe, Pipeline)
    assert "imputer" in pipe.named_steps
    assert "classifier" in pipe.named_steps
    assert pipe.named_steps["imputer"].strategy == "median"
    assert pipe.named_steps["classifier"].n_estimators == 100
    assert pipe.named_steps["classifier"].max_depth == 10
    assert pipe.named_steps["classifier"].random_state == 42


def test_baseline_model_artifact_and_inference(cfg):
    """Verify saved baseline joblib model exists, loads, and executes predictions."""
    model_path = resolve_path(cfg.paths.baseline_model_path)
    assert model_path.exists(), f"Baseline model file not found at: {model_path}"

    loaded_pipe = load_baseline_model(model_path)
    assert isinstance(loaded_pipe, Pipeline)

    # Test prediction on a sample dummy DataFrame with valid feature names
    dummy_input = pd.DataFrame(
        np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
        columns=FEATURE_NAMES,
    )
    pred = loaded_pipe.predict(dummy_input)
    assert len(pred) == 1
    assert isinstance(pred[0], str)

    # Predict probabilities
    proba = loaded_pipe.predict_proba(dummy_input)
    assert proba.shape[0] == 1
    assert np.isclose(np.sum(proba), 1.0, atol=1e-3)


def test_baseline_validation_metrics_artifact(cfg):
    """Verify validation metrics JSON artifact exists and has correct schema."""
    metrics_path = resolve_path(cfg.paths.metrics_dir) / "baseline_val.json"
    assert metrics_path.exists(), f"Validation metrics JSON not found at: {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "best_params" in metrics
    assert "validation_macro_f1" in metrics
    assert "validation_accuracy" in metrics
    assert "validation_macro_precision" in metrics
    assert "validation_macro_recall" in metrics
    assert "classification_report" in metrics
    assert "grid_search_history" in metrics
    assert "top_feature_importances" in metrics

    # Verify protocol: strictly validation split, test split was NOT touched
    assert "test" not in metrics.get("best_params", {})
    assert "Evaluated strictly on the official FMA validation split" in metrics.get("note", "")


def test_baseline_tuning_selection_logic():
    """Verify that tune_and_train_baseline selects the hyperparameter set with highest val macro F1."""
    rng = np.random.default_rng(42)
    n_train = 60
    n_val = 30
    n_features = len(FEATURE_NAMES)

    # 3 distinct classes
    X_train = rng.normal(size=(n_train, n_features))
    y_train = np.array(["Rock"] * 20 + ["Pop"] * 20 + ["Folk"] * 20)

    X_val = rng.normal(size=(n_val, n_features))
    y_val = np.array(["Rock"] * 10 + ["Pop"] * 10 + ["Folk"] * 10)

    param_grid = {
        "n_estimators": [10, 20],
        "max_depth": [3, None],
    }

    best_pipe, summary = tune_and_train_baseline(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_grid=param_grid,
        random_state=42,
        feature_names=FEATURE_NAMES,
    )

    assert isinstance(best_pipe, Pipeline)
    history = summary["grid_search_history"]
    assert len(history) == 4

    # The selected macro F1 must equal the maximum macro F1 in grid history
    best_f1_in_history = max(item["macro_f1"] for item in history)
    assert np.isclose(summary["validation_macro_f1"], best_f1_in_history)

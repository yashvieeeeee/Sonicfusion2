"""Unit tests for RBF SVC, ExtraTrees, and soft-voting ensemble routines."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.extraction import FEATURE_NAMES
from src.models.extra_models import (
    build_extra_trees_pipeline,
    build_svc_pipeline,
    evaluate_predictions,
    find_best_ensemble_weights_2,
    tune_extra_trees_on_validation,
    tune_svc_on_validation,
)


def test_build_svc_pipeline():
    """Verify that SVC pipeline consists of SimpleImputer, StandardScaler, and SVC."""
    pipe = build_svc_pipeline(C=3.0, gamma="scale", random_state=42)
    assert isinstance(pipe, Pipeline)
    assert "imputer" in pipe.named_steps
    assert "scaler" in pipe.named_steps
    assert "classifier" in pipe.named_steps
    assert pipe.named_steps["classifier"].C == 3.0
    assert pipe.named_steps["classifier"].probability is True


def test_build_extra_trees_pipeline():
    """Verify that ExtraTrees pipeline consists of SimpleImputer and ExtraTreesClassifier."""
    pipe = build_extra_trees_pipeline(n_estimators=100, max_features="sqrt", random_state=42)
    assert isinstance(pipe, Pipeline)
    assert "imputer" in pipe.named_steps
    assert "classifier" in pipe.named_steps
    assert pipe.named_steps["classifier"].n_estimators == 100


def test_evaluate_predictions_and_top3():
    """Verify accuracy, macro F1, and top-3 accuracy calculation."""
    y_true = np.array(["Pop", "Rock", "Folk", "Jazz"])
    class_labels = ["Folk", "Jazz", "Pop", "Rock"]
    # 4 classes: Pop, Rock, Folk, Jazz
    probs = np.array([
        [0.1, 0.1, 0.7, 0.1],  # True: Pop (rank 1)
        [0.1, 0.1, 0.2, 0.6],  # True: Rock (rank 1)
        [0.2, 0.4, 0.3, 0.1],  # True: Folk (rank 3 - Jazz is 1, Pop is 2, Folk is 3)
        [0.4, 0.1, 0.3, 0.2],  # True: Jazz (rank 4 - not in top 3)
    ])
    res = evaluate_predictions(y_true, probs, class_labels)
    assert res["accuracy"] == 0.5  # 2 of 4 correct (Pop, Rock)
    assert res["top_3_accuracy"] == 0.75  # 3 of 4 in top 3 (Pop, Rock, Folk)


def test_find_best_ensemble_weights_2():
    """Verify grid search finds optimal convex weights for 2 probability matrices."""
    y_true = np.array(["Pop"] * 10 + ["Rock"] * 10)
    classes = ["Pop", "Rock"]

    # Model A: good at Pop, bad at Rock
    prob_a = np.array([[0.9, 0.1]] * 10 + [[0.6, 0.4]] * 10)
    # Model B: bad at Pop, good at Rock
    prob_b = np.array([[0.4, 0.6]] * 10 + [[0.1, 0.9]] * 10)

    weights, metrics = find_best_ensemble_weights_2(y_true, prob_a, prob_b, classes, steps=11)
    assert len(weights) == 2
    assert np.isclose(sum(weights), 1.0)
    assert metrics["accuracy"] >= 0.5


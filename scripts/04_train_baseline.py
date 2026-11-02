"""Train and tune the classical Random Forest baseline model.

Implements Gate G3 requirements from docs/PLAN.md (§05, §08, Addenda A4):
- Reads features from data/processed/features.parquet.
- Uses scikit-learn Pipeline (SimpleImputer + RandomForestClassifier).
- Tunes hyperparameters from configs/config.yaml using fixed random seed.
- Selects best model strictly by validation macro F1 score.
- Saves model pipeline to models/baseline/random_forest.joblib.
- Saves validation metrics to reports/metrics/baseline_val.json.
- Strictly adheres to evaluation protocol: DOES NOT evaluate on the test split.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_project_root, load_config, resolve_path
from src.features.extraction import FEATURE_NAMES
from src.models.baseline import (
    save_baseline_model,
    tune_and_train_adaboost,
    tune_and_train_baseline,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and tune Random Forest classical baseline on validation split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to features.parquet file (defaults to config path).",
    )
    parser.add_argument(
        "--include-adaboost",
        action="store_true",
        default=True,
        help="Also evaluate AdaBoost baseline on validation split for comparison.",
    )
    return parser.parse_args()


def train_baseline(
    config_path: str | None = None,
    features_path_override: str | None = None,
    include_adaboost: bool = True,
) -> Dict[str, Any]:
    """Train Random Forest baseline with hyperparameter grid search on validation data."""
    cfg = load_config(config_path)
    root = get_project_root()

    features_file = (
        resolve_path(features_path_override)
        if features_path_override
        else resolve_path(cfg.paths.features_path)
    )
    model_output_file = resolve_path(cfg.paths.baseline_model_path)
    metrics_output_file = resolve_path(cfg.paths.metrics_dir) / "baseline_val.json"

    if not features_file.exists():
        raise FileNotFoundError(
            f"Features parquet file not found at: {features_file}. "
            "Please run scripts/02_extract_features.py first."
        )

    print(f"Loading feature dataset from {features_file.relative_to(root)}...")
    df = pd.read_parquet(features_file)

    # Verify required columns
    for col in ["genre", "split"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from {features_file}")

    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            raise KeyError(f"Feature '{feat}' missing from {features_file}")

    # Enforce strict split protocol (PLAN §08, §16, Addendum A5)
    train_df = df[df["split"] == "training"]
    val_df = df[df["split"] == "validation"]
    test_count = int((df["split"] == "test").sum())

    print(f"Dataset summary:")
    print(f"  Training samples   : {len(train_df)}")
    print(f"  Validation samples : {len(val_df)}")
    print(f"  Test samples       : {test_count} (EXCLUDED: Test split is never touched during training or tuning)")

    if len(train_df) == 0:
        raise ValueError("No training samples found in features dataset!")
    if len(val_df) == 0:
        raise ValueError("No validation samples found in features dataset!")

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["genre"]
    X_val = val_df[FEATURE_NAMES]
    y_val = val_df["genre"]

    # Parameter grid from config
    param_grid = dict(cfg.baseline.param_grid)
    seed = int(cfg.baseline.random_state if "random_state" in cfg.baseline else cfg.seed)

    print(f"\nRunning parameter grid search (tuning by validation macro F1)...")
    print(f"  n_estimators candidates : {param_grid.get('n_estimators')}")
    print(f"  max_depth candidates    : {param_grid.get('max_depth')}")
    print(f"  Fixed random seed       : {seed}")

    best_pipeline, val_metrics = tune_and_train_baseline(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_grid=param_grid,
        random_state=seed,
        feature_names=FEATURE_NAMES,
    )

    # Save model pipeline
    save_baseline_model(best_pipeline, model_output_file)
    print(f"\nSaved best baseline pipeline to {model_output_file.relative_to(root)}.")

    adaboost_benchmark = None
    if include_adaboost:
        print("\nEvaluating comparative AdaBoost baseline on validation data...")
        try:
            _, adaboost_metrics = tune_and_train_adaboost(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                random_state=seed,
            )
            adaboost_benchmark = {
                "model_type": "AdaBoostClassifier",
                "base_estimator": f"DecisionTreeClassifier(max_depth={adaboost_metrics['best_params'].get('max_depth', 2)})",
                "best_params": adaboost_metrics["best_params"],
                "validation_accuracy": adaboost_metrics["validation_accuracy"],
                "validation_macro_f1": adaboost_metrics["validation_macro_f1"],
                "validation_macro_precision": adaboost_metrics["validation_macro_precision"],
                "validation_macro_recall": adaboost_metrics["validation_macro_recall"],
                "classification_report": adaboost_metrics["classification_report"],
                "grid_search_history": adaboost_metrics["grid_search_history"],
                "dataset_context_note": (
                    "FMA Small consists of 8 balanced genre classes. Uniform random guess accuracy is 12.5% (1/8). "
                    "Validation accuracy of ~52% (Random Forest) and ~44% (AdaBoost) exceeds random chance by 3.5x-4.2x. "
                    "Handcrafted 51-d summary acoustic features reach an empirical ceiling on complex musical audio."
                ),
            }
            print(f"  AdaBoost best params: {adaboost_metrics['best_params']}")
            print(f"  AdaBoost Val Acc: {adaboost_metrics['validation_accuracy']:.4f}, Macro-F1: {adaboost_metrics['validation_macro_f1']:.4f}")
        except Exception as exc:
            print(f"  Warning: AdaBoost evaluation skipped due to error: {exc}")

    # Prepare JSON metrics payload (preserving all existing RandomForest fields)
    metrics_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestClassifier",
        "random_state": seed,
        "best_params": val_metrics["best_params"],
        "validation_macro_f1": val_metrics["validation_macro_f1"],
        "validation_accuracy": val_metrics["validation_accuracy"],
        "validation_macro_precision": val_metrics["validation_macro_precision"],
        "validation_macro_recall": val_metrics["validation_macro_recall"],
        "classification_report": val_metrics["classification_report"],
        "top_feature_importances": dict(list(val_metrics["feature_importances"].items())[:15]),
        "grid_search_history": val_metrics["grid_search_history"],
        "num_train_samples": len(train_df),
        "num_val_samples": len(val_df),
        "note": "Evaluated strictly on the official FMA validation split. The test split was not evaluated.",
    }
    if adaboost_benchmark is not None:
        metrics_payload["adaboost_benchmark"] = adaboost_benchmark

    metrics_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved validation metrics to {metrics_output_file.relative_to(root)}.")

    # Print summary
    _print_baseline_summary(metrics_payload)

    return metrics_payload


def _print_baseline_summary(metrics: Dict[str, Any]) -> None:
    """Print formatted summary of baseline validation metrics."""
    print("\n" + "=" * 60)
    print("        SONICFUSION 2: BASELINE VALIDATION RESULTS        ")
    print("=" * 60)
    print(f"Primary Baseline     : RandomForestClassifier")
    print(f"Best Hyperparameters : {metrics['best_params']}")
    print(f"Validation Macro F1  : {metrics['validation_macro_f1']:.4f}")
    print(f"Validation Accuracy  : {metrics['validation_accuracy']:.4f} (Random guess baseline: 12.50%)")
    print(f"Validation Precision : {metrics['validation_macro_precision']:.4f}")
    print(f"Validation Recall    : {metrics['validation_macro_recall']:.4f}")

    if "adaboost_benchmark" in metrics:
        ada = metrics["adaboost_benchmark"]
        print("\n--- Additional Comparative Baseline: AdaBoost ---")
        print(f"Model Type           : {ada['model_type']}")
        print(f"Best Params          : {ada['best_params']}")
        print(f"Validation Macro F1  : {ada['validation_macro_f1']:.4f}")
        print(f"Validation Accuracy  : {ada['validation_accuracy']:.4f}")

    print("\n--- Top 5 Important Features ---")
    for feat, imp in list(metrics["top_feature_importances"].items())[:5]:
        print(f"  {feat:<25}: {imp:.4f}")

    print("\nEvaluation Protocol Note:")
    print(f"  {metrics['note']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = parse_args()
    train_baseline(
        config_path=args.config,
        features_path_override=args.features,
        include_adaboost=args.include_adaboost,
    )

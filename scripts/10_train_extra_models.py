"""Train, tune, and ensemble additional classical models (RBF SVC, ExtraTrees, and Ensembles).

Follows Task 9 specifications:
1. RBF SVC (probability=True) with StandardScaler in Pipeline.
   Tuned on validation over C in [1, 3, 10, 30] and gamma in ['scale', 0.01, 0.003].
2. ExtraTreesClassifier in Pipeline.
   Tuned on validation over n_estimators=400, max_features in ['sqrt', 0.3].
3. Soft-voting ensemble of RF + SVC (+ GenreCNN if available).
   Ensemble weights selected strictly on validation data.
4. Saves metrics (val accuracy, macro-F1, top-3 accuracy) to reports/metrics/extra_models_val.json.
5. Strictly adheres to evaluation protocol: DOES NOT evaluate on the test split.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.audio.spectrogram import to_cnn_tensor
from src.config import get_project_root, load_config, resolve_path
from src.features.extraction import FEATURE_NAMES
from src.models.baseline import load_baseline_model
from src.models.extra_models import (
    evaluate_predictions,
    find_best_ensemble_weights_2,
    find_best_ensemble_weights_3,
    tune_extra_trees_on_validation,
    tune_svc_on_validation,
)
from src.models.genre_cnn import (
    GenreCNN,
    aggregate_window_probabilities,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate extra models (RBF SVC, ExtraTrees) and ensembles on validation split."
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to features.parquet (defaults to config path: data/processed/features.parquet).",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Output metrics path (defaults to reports/metrics/extra_models_val.json).",
    )
    return parser.parse_args()


def load_cnn_val_probabilities(
    val_df: pd.DataFrame,
    class_names: List[str],
) -> Optional[np.ndarray]:
    """Extract GenreCNN validation probabilities from cached log-mel spectrograms if checkpoint exists."""
    root = get_project_root()
    cfg = load_config()

    cnn_ckpt_path = resolve_path(cfg.paths.cnn_checkpoint_path)
    cache_dir = resolve_path(cfg.paths.spectrogram_cache_dir)

    if not cnn_ckpt_path.exists():
        print(f"GenreCNN checkpoint not found at: {cnn_ckpt_path.relative_to(root)}. Skipping CNN in ensemble.")
        return None

    if not cache_dir.exists():
        print(f"Spectrogram cache dir not found at: {cache_dir.relative_to(root)}. Skipping CNN in ensemble.")
        return None

    print(f"\nLoading GenreCNN checkpoint from {cnn_ckpt_path.relative_to(root)}...")
    device = torch.device("cpu")
    checkpoint = torch.load(cnn_ckpt_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    # Clean possible prefix
    cleaned_sd = {}
    for k, v in state_dict.items():
        k_clean = k.replace("module.", "").replace("model.", "")
        cleaned_sd[k_clean] = v

    cnn_classes = checkpoint.get("class_names", class_names)
    num_classes = len(cnn_classes)
    dropout = float(cfg.cnn.architecture.dropout if "cnn" in cfg else 0.3)

    model = GenreCNN(num_classes=num_classes, in_channels=1, dropout=dropout)
    try:
        model.load_state_dict(cleaned_sd, strict=False)
    except Exception as exc:
        print(f"Warning: Could not load exact weights into GenreCNN: {exc}")
        return None

    model.eval()

    num_windows = int(cfg.audio.windows_per_track if "audio" in cfg else 3)
    val_probs_list: List[np.ndarray] = []

    print(f"Evaluating GenreCNN on {len(val_df)} validation tracks using cached spectrograms...")
    with torch.no_grad():
        for _, row in val_df.iterrows():
            t_id = int(row["track_id"])
            win_tensors: List[torch.Tensor] = []

            all_windows_found = True
            for w_idx in range(num_windows):
                cache_file = cache_dir / f"{t_id:06d}_{w_idx}.npy"
                if not cache_file.exists():
                    all_windows_found = False
                    break
                spec_arr = np.load(cache_file).astype(np.float32)
                win_tensors.append(to_cnn_tensor(spec_arr))

            if not all_windows_found or len(win_tensors) == 0:
                val_probs_list.append(np.ones(num_classes) / num_classes)
                continue

            batch = torch.stack(win_tensors, dim=0)
            logits = model(batch)
            window_probs = F.softmax(logits, dim=-1)
            track_prob = aggregate_window_probabilities(window_probs)
            val_probs_list.append(track_prob)

    cnn_probs_raw = np.stack(val_probs_list, axis=0)

    # Reorder columns to match class_names if necessary
    if list(cnn_classes) != list(class_names):
        reorder_idx = [cnn_classes.index(c) for c in class_names]
        cnn_probs_aligned = cnn_probs_raw[:, reorder_idx]
    else:
        cnn_probs_aligned = cnn_probs_raw

    return cnn_probs_aligned


def train_extra_models(
    features_path_override: Optional[str] = None,
    output_metrics_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Train, tune, and evaluate RBF SVC, ExtraTrees, and ensembles on validation split."""
    root = get_project_root()
    cfg = load_config()
    seed = int(cfg.seed)

    features_file = (
        resolve_path(features_path_override)
        if features_path_override
        else resolve_path(cfg.paths.features_path)
    )
    metrics_file = (
        resolve_path(output_metrics_path_override)
        if output_metrics_path_override
        else root / "reports" / "metrics" / "extra_models_val.json"
    )

    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found at: {features_file}")

    print(f"Loading feature dataset from {features_file.relative_to(root)}...")
    df = pd.read_parquet(features_file)

    # Enforce strict split protocol: Test split is never touched!
    train_df = df[df["split"] == "training"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_count = int((df["split"] == "test").sum())

    print(f"Dataset summary:")
    print(f"  Training samples   : {len(train_df)}")
    print(f"  Validation samples : {len(val_df)}")
    print(f"  Test samples       : {test_count} (STRICTLY EXCLUDED: Never evaluated or tuned)")

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["genre"]
    X_val = val_df[FEATURE_NAMES]
    y_val = val_df["genre"]

    # Canonical sorted class names
    class_names = sorted(list(y_train.unique()))

    # Ensure output model directory exists
    models_extra_dir = root / "models" / "extra"
    models_extra_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Train and Tune RBF SVC
    # =========================================================================
    print("\n" + "=" * 60)
    print(" 1. TUNING RBF SVC (StandardScaler + SimpleImputer + SVC)")
    print("=" * 60)
    svc_grid = {
        "C": [1, 3, 10, 30],
        "gamma": ["scale", 0.01, 0.003],
    }
    t0 = time.perf_counter()
    best_svc_pipe, svc_summary = tune_svc_on_validation(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_grid=svc_grid,
        random_state=seed,
        n_jobs=-1,
    )
    t_svc = time.perf_counter() - t0
    svc_model_path = models_extra_dir / "rbf_svc.joblib"
    joblib.dump(best_svc_pipe, svc_model_path)
    print(f"Best RBF SVC: {svc_summary['best_params']} (tuning took {t_svc:.1f}s)")
    print(f"  Val Accuracy  : {svc_summary['validation_accuracy']:.4f}")
    print(f"  Val Macro-F1  : {svc_summary['validation_macro_f1']:.4f}")
    print(f"  Top-3 Accuracy: {svc_summary['validation_top_3_accuracy']:.4f}")

    svc_val_probs = best_svc_pipe.predict_proba(X_val)

    # =========================================================================
    # 2. Train and Tune ExtraTreesClassifier
    # =========================================================================
    print("\n" + "=" * 60)
    print(" 2. TUNING EXTRATREES (SimpleImputer + ExtraTreesClassifier)")
    print("=" * 60)
    et_grid = {
        "n_estimators": [400],
        "max_features": ["sqrt", 0.3],
    }
    t0 = time.perf_counter()
    best_et_pipe, et_summary = tune_extra_trees_on_validation(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_grid=et_grid,
        random_state=seed,
    )
    t_et = time.perf_counter() - t0
    et_model_path = models_extra_dir / "extra_trees.joblib"
    joblib.dump(best_et_pipe, et_model_path)
    print(f"Best ExtraTrees: {et_summary['best_params']} (tuning took {t_et:.1f}s)")
    print(f"  Val Accuracy  : {et_summary['validation_accuracy']:.4f}")
    print(f"  Val Macro-F1  : {et_summary['validation_macro_f1']:.4f}")
    print(f"  Top-3 Accuracy: {et_summary['validation_top_3_accuracy']:.4f}")

    et_val_probs = best_et_pipe.predict_proba(X_val)

    # =========================================================================
    # 3. Load Existing Random Forest Baseline
    # =========================================================================
    print("\n" + "=" * 60)
    print(" 3. LOADING EXISTING RANDOM FOREST BASELINE")
    print("=" * 60)
    rf_model_path = resolve_path(cfg.paths.baseline_model_path)
    if not rf_model_path.exists():
        raise FileNotFoundError(f"Baseline Random Forest model not found at: {rf_model_path}")
    rf_pipe = load_baseline_model(rf_model_path)
    rf_val_probs = rf_pipe.predict_proba(X_val)
    rf_eval = evaluate_predictions(y_val, rf_val_probs, rf_pipe.classes_)
    print(f"Random Forest (Baseline):")
    print(f"  Val Accuracy  : {rf_eval['accuracy']:.4f}")
    print(f"  Val Macro-F1  : {rf_eval['macro_f1']:.4f}")
    print(f"  Top-3 Accuracy: {rf_eval['top_3_accuracy']:.4f}")

    # =========================================================================
    # 4. Load GenreCNN Validation Probabilities (if available)
    # =========================================================================
    cnn_val_probs = load_cnn_val_probabilities(val_df=val_df, class_names=class_names)
    cnn_eval: Optional[Dict[str, Any]] = None
    if cnn_val_probs is not None:
        cnn_eval = evaluate_predictions(y_val, cnn_val_probs, class_names)
        print(f"\nGenreCNN (Validation):")
        print(f"  Val Accuracy  : {cnn_eval['accuracy']:.4f}")
        print(f"  Val Macro-F1  : {cnn_eval['macro_f1']:.4f}")
        print(f"  Top-3 Accuracy: {cnn_eval['top_3_accuracy']:.4f}")

    # =========================================================================
    # 5. Soft-Voting Ensembles
    # =========================================================================
    print("\n" + "=" * 60)
    print(" 5. SOFT-VOTING ENSEMBLES (Optimized strictly on validation)")
    print("=" * 60)

    # Ensure all class orders align
    assert list(rf_pipe.classes_) == class_names
    assert list(best_svc_pipe.classes_) == class_names
    assert list(best_et_pipe.classes_) == class_names

    # Ensemble A: RF + SVC
    weights_rf_svc, eval_rf_svc = find_best_ensemble_weights_2(
        y_val=y_val,
        prob_a=rf_val_probs,
        prob_b=svc_val_probs,
        class_labels=class_names,
    )
    print(f"\nEnsemble [RF + SVC]:")
    print(f"  Optimal Weights (RF, SVC) : {weights_rf_svc}")
    print(f"  Val Accuracy              : {eval_rf_svc['accuracy']:.4f}")
    print(f"  Val Macro-F1              : {eval_rf_svc['macro_f1']:.4f}")
    print(f"  Top-3 Accuracy            : {eval_rf_svc['top_3_accuracy']:.4f}")

    # Ensemble B: RF + SVC + ExtraTrees
    weights_rf_svc_et, eval_rf_svc_et = find_best_ensemble_weights_3(
        y_val=y_val,
        prob_a=rf_val_probs,
        prob_b=svc_val_probs,
        prob_c=et_val_probs,
        class_labels=class_names,
    )
    print(f"\nEnsemble [RF + SVC + ExtraTrees]:")
    print(f"  Optimal Weights (RF, SVC, ET): {weights_rf_svc_et}")
    print(f"  Val Accuracy                 : {eval_rf_svc_et['accuracy']:.4f}")
    print(f"  Val Macro-F1                 : {eval_rf_svc_et['macro_f1']:.4f}")
    print(f"  Top-3 Accuracy               : {eval_rf_svc_et['top_3_accuracy']:.4f}")

    # Ensemble C: RF + SVC + GenreCNN (if CNN available)
    eval_rf_svc_cnn: Optional[Dict[str, Any]] = None
    weights_rf_svc_cnn: Optional[Tuple[float, float, float]] = None
    if cnn_val_probs is not None:
        weights_rf_svc_cnn, eval_rf_svc_cnn = find_best_ensemble_weights_3(
            y_val=y_val,
            prob_a=rf_val_probs,
            prob_b=svc_val_probs,
            prob_c=cnn_val_probs,
            class_labels=class_names,
        )
        print(f"\nEnsemble [RF + SVC + GenreCNN]:")
        print(f"  Optimal Weights (RF, SVC, CNN): {weights_rf_svc_cnn}")
        print(f"  Val Accuracy                  : {eval_rf_svc_cnn['accuracy']:.4f}")
        print(f"  Val Macro-F1                  : {eval_rf_svc_cnn['macro_f1']:.4f}")
        print(f"  Top-3 Accuracy                : {eval_rf_svc_cnn['top_3_accuracy']:.4f}")

    # =========================================================================
    # 6. Assemble JSON Output Payload
    # =========================================================================
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": seed,
        "note": "Evaluated strictly on the official FMA validation split. The test split was not evaluated.",
        "dataset": {
            "num_train_tracks": len(train_df),
            "num_val_tracks": len(val_df),
            "class_names": class_names,
            "random_guess_accuracy": 1.0 / len(class_names),
        },
        "single_models": {
            "random_forest": {
                "model_type": "RandomForestClassifier",
                "validation_accuracy": rf_eval["accuracy"],
                "validation_macro_f1": rf_eval["macro_f1"],
                "validation_top_3_accuracy": rf_eval["top_3_accuracy"],
            },
            "rbf_svc": {
                "model_type": "SVC(kernel='rbf')",
                "best_params": svc_summary["best_params"],
                "validation_accuracy": svc_summary["validation_accuracy"],
                "validation_macro_f1": svc_summary["validation_macro_f1"],
                "validation_top_3_accuracy": svc_summary["validation_top_3_accuracy"],
                "grid_search_history": svc_summary["grid_search_history"],
                "saved_model_path": str(svc_model_path.relative_to(root)),
            },
            "extra_trees": {
                "model_type": "ExtraTreesClassifier",
                "best_params": et_summary["best_params"],
                "validation_accuracy": et_summary["validation_accuracy"],
                "validation_macro_f1": et_summary["validation_macro_f1"],
                "validation_top_3_accuracy": et_summary["validation_top_3_accuracy"],
                "grid_search_history": et_summary["grid_search_history"],
                "saved_model_path": str(et_model_path.relative_to(root)),
            },
        },
        "ensembles": {
            "rf_plus_svc": {
                "models": ["RandomForestClassifier", "SVC(kernel='rbf')"],
                "weights": list(weights_rf_svc),
                "validation_accuracy": eval_rf_svc["accuracy"],
                "validation_macro_f1": eval_rf_svc["macro_f1"],
                "validation_top_3_accuracy": eval_rf_svc["top_3_accuracy"],
            },
            "rf_plus_svc_plus_extratrees": {
                "models": ["RandomForestClassifier", "SVC(kernel='rbf')", "ExtraTreesClassifier"],
                "weights": list(weights_rf_svc_et),
                "validation_accuracy": eval_rf_svc_et["accuracy"],
                "validation_macro_f1": eval_rf_svc_et["macro_f1"],
                "validation_top_3_accuracy": eval_rf_svc_et["top_3_accuracy"],
            },
        },
    }

    if cnn_eval is not None and eval_rf_svc_cnn is not None:
        payload["single_models"]["genre_cnn"] = {
            "model_type": "GenreCNN",
            "validation_accuracy": cnn_eval["accuracy"],
            "validation_macro_f1": cnn_eval["macro_f1"],
            "validation_top_3_accuracy": cnn_eval["top_3_accuracy"],
        }
        payload["ensembles"]["rf_plus_svc_plus_cnn"] = {
            "models": ["RandomForestClassifier", "SVC(kernel='rbf')", "GenreCNN"],
            "weights": list(weights_rf_svc_cnn),
            "validation_accuracy": eval_rf_svc_cnn["accuracy"],
            "validation_macro_f1": eval_rf_svc_cnn["macro_f1"],
            "validation_top_3_accuracy": eval_rf_svc_cnn["top_3_accuracy"],
        }

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved validation metrics to {metrics_file.relative_to(root)}.")

    return payload


if __name__ == "__main__":
    args = parse_args()
    train_extra_models(
        features_path_override=args.features,
        output_metrics_path_override=args.output_metrics,
    )

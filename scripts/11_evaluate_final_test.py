"""One-time final evaluation on the official FMA held-out TEST split.

Evaluates frozen models without any hyperparameter tuning or retuning:
- Random Forest Baseline (models/baseline/random_forest.joblib)
- RBF SVC (models/extra/rbf_svc.joblib)
- ExtraTrees (models/extra/extra_trees.joblib)
- GenreCNN (models/checkpoints/genre_cnn.pt)
- Soft-Voting Ensemble: SVC 0.40 + CNN 0.60 + RF 0.00 (validation-chosen weights)

Saves results to reports/metrics/final_test.json and prints Validation vs. Test comparison table.
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
import torch
import torch.nn.functional as F

from src.audio.spectrogram import to_cnn_tensor
from src.config import get_project_root, load_config, resolve_path
from src.features.extraction import FEATURE_NAMES
from src.models.extra_models import evaluate_predictions
from src.models.genre_cnn import GenreCNN, aggregate_window_probabilities


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="One-time frozen model evaluation on official held-out test split."
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to features.parquet (defaults to data/processed/features.parquet).",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Output metrics path (defaults to reports/metrics/final_test.json).",
    )
    return parser.parse_args()


def load_cnn_test_probabilities(
    test_df: pd.DataFrame,
    class_names: List[str],
) -> np.ndarray:
    """Compute GenreCNN test probabilities from cached log-mel spectrograms."""
    root = get_project_root()
    cfg = load_config()

    cnn_ckpt_path = resolve_path(cfg.paths.cnn_checkpoint_path)
    cache_dir = resolve_path(cfg.paths.spectrogram_cache_dir)

    if not cnn_ckpt_path.exists():
        raise FileNotFoundError(f"GenreCNN checkpoint not found at: {cnn_ckpt_path}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"Spectrogram cache dir not found at: {cache_dir}")

    device = torch.device("cpu")
    checkpoint = torch.load(cnn_ckpt_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    cleaned_sd = {}
    for k, v in state_dict.items():
        k_clean = k.replace("module.", "").replace("model.", "")
        cleaned_sd[k_clean] = v

    cnn_classes = checkpoint.get("class_names", class_names)
    num_classes = len(cnn_classes)
    dropout = float(cfg.cnn.architecture.dropout if "cnn" in cfg else 0.3)

    model = GenreCNN(num_classes=num_classes, in_channels=1, dropout=dropout)
    model.load_state_dict(cleaned_sd, strict=False)
    model.eval()

    num_windows = int(cfg.audio.windows_per_track if "audio" in cfg else 3)
    test_probs_list: List[np.ndarray] = []

    print(f"Evaluating GenreCNN on {len(test_df)} held-out TEST tracks...")
    with torch.no_grad():
        for _, row in test_df.iterrows():
            t_id = int(row["track_id"])
            win_tensors: List[torch.Tensor] = []

            for w_idx in range(num_windows):
                cache_file = cache_dir / f"{t_id:06d}_{w_idx}.npy"
                if not cache_file.exists():
                    raise FileNotFoundError(f"Cached spectrogram missing: {cache_file}")
                spec_arr = np.load(cache_file).astype(np.float32)
                win_tensors.append(to_cnn_tensor(spec_arr))

            batch = torch.stack(win_tensors, dim=0)
            logits = model(batch)
            window_probs = F.softmax(logits, dim=-1)
            track_prob = aggregate_window_probabilities(window_probs)
            test_probs_list.append(track_prob)

    cnn_probs_raw = np.stack(test_probs_list, axis=0)

    # Align columns to canonical class_names
    if list(cnn_classes) != list(class_names):
        reorder_idx = [cnn_classes.index(c) for c in class_names]
        cnn_probs = cnn_probs_raw[:, reorder_idx]
    else:
        cnn_probs = cnn_probs_raw

    return cnn_probs


def evaluate_final_test(
    features_path_override: Optional[str] = None,
    output_metrics_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one-time evaluation on the official test split and save results."""
    root = get_project_root()
    cfg = load_config()

    features_file = (
        resolve_path(features_path_override)
        if features_path_override
        else resolve_path(cfg.paths.features_path)
    )
    output_file = (
        resolve_path(output_metrics_path_override)
        if output_metrics_path_override
        else root / "reports" / "metrics" / "final_test.json"
    )

    # Load validation metrics for side-by-side comparison table
    val_metrics_file = root / "reports" / "metrics" / "extra_models_val.json"
    val_data: Dict[str, Any] = {}
    if val_metrics_file.exists():
        with open(val_metrics_file, "r", encoding="utf-8") as f:
            val_data = json.load(f)

    print(f"Loading feature dataset from {features_file.relative_to(root)}...")
    df = pd.read_parquet(features_file)

    test_df = df[df["split"] == "test"].copy()
    if len(test_df) != 800:
        raise ValueError(f"Expected 800 test tracks, found {len(test_df)}")

    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["genre"]
    class_names = sorted(list(y_test.unique()))

    print(f"Test split size: {len(test_df)} tracks across {len(class_names)} genres.")

    # Model paths
    rf_path = resolve_path(cfg.paths.baseline_model_path)
    svc_path = root / "models" / "extra" / "rbf_svc.joblib"
    et_path = root / "models" / "extra" / "extra_trees.joblib"

    # 1. Random Forest
    print("\n[1/5] Evaluating Random Forest on test split...")
    rf_pipe = joblib.load(rf_path)
    rf_test_probs = rf_pipe.predict_proba(X_test)
    rf_eval = evaluate_predictions(y_test, rf_test_probs, rf_pipe.classes_)

    # 2. RBF SVC
    print("[2/5] Evaluating RBF SVC on test split...")
    svc_pipe = joblib.load(svc_path)
    svc_test_probs = svc_pipe.predict_proba(X_test)
    svc_eval = evaluate_predictions(y_test, svc_test_probs, svc_pipe.classes_)

    # 3. ExtraTrees
    print("[3/5] Evaluating ExtraTrees on test split...")
    et_pipe = joblib.load(et_path)
    et_test_probs = et_pipe.predict_proba(X_test)
    et_eval = evaluate_predictions(y_test, et_test_probs, et_pipe.classes_)

    # 4. GenreCNN
    print("[4/5] Evaluating GenreCNN on test split...")
    cnn_test_probs = load_cnn_test_probabilities(test_df, class_names)
    cnn_eval = evaluate_predictions(y_test, cnn_test_probs, class_names)

    # 5. Soft-Voting Ensemble (SVC 0.40, CNN 0.60, RF 0.00)
    print("[5/5] Evaluating Ensemble (SVC 0.40 + CNN 0.60 + RF 0.00) on test split...")
    ensemble_weights = {"svc": 0.40, "cnn": 0.60, "rf": 0.00}
    ensemble_test_probs = (
        0.40 * svc_test_probs +
        0.60 * cnn_test_probs +
        0.00 * rf_test_probs
    )
    ensemble_eval = evaluate_predictions(y_test, ensemble_test_probs, class_names)

    # Assemble JSON payload
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "One-time final evaluation on official FMA test split. Zero test retuning.",
        "evaluated_tracks": len(test_df),
        "class_names": class_names,
        "models": {
            "random_forest": {
                "name": "Random Forest Baseline",
                "test_accuracy": rf_eval["accuracy"],
                "test_macro_f1": rf_eval["macro_f1"],
                "test_top_3_accuracy": rf_eval["top_3_accuracy"],
                "classification_report": rf_eval["classification_report"],
            },
            "rbf_svc": {
                "name": "RBF SVC (C=10, gamma=0.01)",
                "test_accuracy": svc_eval["accuracy"],
                "test_macro_f1": svc_eval["macro_f1"],
                "test_top_3_accuracy": svc_eval["top_3_accuracy"],
                "classification_report": svc_eval["classification_report"],
            },
            "extra_trees": {
                "name": "ExtraTrees (n=400, sqrt)",
                "test_accuracy": et_eval["accuracy"],
                "test_macro_f1": et_eval["macro_f1"],
                "test_top_3_accuracy": et_eval["top_3_accuracy"],
                "classification_report": et_eval["classification_report"],
            },
            "genre_cnn": {
                "name": "GenreCNN (Best Checkpoint)",
                "test_accuracy": cnn_eval["accuracy"],
                "test_macro_f1": cnn_eval["macro_f1"],
                "test_top_3_accuracy": cnn_eval["top_3_accuracy"],
                "classification_report": cnn_eval["classification_report"],
            },
            "ensemble_rf_svc_cnn": {
                "name": "Ensemble (SVC 0.40 + CNN 0.60 + RF 0.00)",
                "weights": ensemble_weights,
                "test_accuracy": ensemble_eval["accuracy"],
                "test_macro_f1": ensemble_eval["macro_f1"],
                "test_top_3_accuracy": ensemble_eval["top_3_accuracy"],
                "classification_report": ensemble_eval["classification_report"],
            },
        },
    }

    # Save to reports/metrics/final_test.json
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved final test metrics to {output_file.relative_to(root)}.")

    # Print Validation vs Test comparison table
    _print_val_vs_test_table(val_data, payload)

    return payload


def _print_val_vs_test_table(val_data: Dict[str, Any], test_payload: Dict[str, Any]) -> None:
    """Print formatted Validation vs Test comparison table."""
    val_singles = val_data.get("single_models", {})
    val_ensembles = val_data.get("ensembles", {})
    test_models = test_payload["models"]

    rows = [
        (
            "Random Forest",
            val_singles.get("random_forest", {}).get("validation_accuracy"),
            test_models["random_forest"]["test_accuracy"],
            val_singles.get("random_forest", {}).get("validation_macro_f1"),
            test_models["random_forest"]["test_macro_f1"],
            val_singles.get("random_forest", {}).get("validation_top_3_accuracy"),
            test_models["random_forest"]["test_top_3_accuracy"],
        ),
        (
            "RBF SVC",
            val_singles.get("rbf_svc", {}).get("validation_accuracy"),
            test_models["rbf_svc"]["test_accuracy"],
            val_singles.get("rbf_svc", {}).get("validation_macro_f1"),
            test_models["rbf_svc"]["test_macro_f1"],
            val_singles.get("rbf_svc", {}).get("validation_top_3_accuracy"),
            test_models["rbf_svc"]["test_top_3_accuracy"],
        ),
        (
            "ExtraTrees",
            val_singles.get("extra_trees", {}).get("validation_accuracy"),
            test_models["extra_trees"]["test_accuracy"],
            val_singles.get("extra_trees", {}).get("validation_macro_f1"),
            test_models["extra_trees"]["test_macro_f1"],
            val_singles.get("extra_trees", {}).get("validation_top_3_accuracy"),
            test_models["extra_trees"]["test_top_3_accuracy"],
        ),
        (
            "GenreCNN",
            val_singles.get("genre_cnn", {}).get("validation_accuracy"),
            test_models["genre_cnn"]["test_accuracy"],
            val_singles.get("genre_cnn", {}).get("validation_macro_f1"),
            test_models["genre_cnn"]["test_macro_f1"],
            val_singles.get("genre_cnn", {}).get("validation_top_3_accuracy"),
            test_models["genre_cnn"]["test_top_3_accuracy"],
        ),
        (
            "Ensemble (SVC.40+CNN.60)",
            val_ensembles.get("rf_plus_svc_plus_cnn", {}).get("validation_accuracy"),
            test_models["ensemble_rf_svc_cnn"]["test_accuracy"],
            val_ensembles.get("rf_plus_svc_plus_cnn", {}).get("validation_macro_f1"),
            test_models["ensemble_rf_svc_cnn"]["test_macro_f1"],
            val_ensembles.get("rf_plus_svc_plus_cnn", {}).get("validation_top_3_accuracy"),
            test_models["ensemble_rf_svc_cnn"]["test_top_3_accuracy"],
        ),
    ]

    print("\n" + "=" * 84)
    print("            OFFICIAL FMA EVALUATION: VALIDATION vs. TEST COMPARISON            ")
    print("=" * 84)
    header = f"{'Model / Ensemble':<25} | {'Val Acc':<8} {'Test Acc':<8} | {'Val F1':<8} {'Test F1':<8} | {'Val Top3':<8} {'Test Top3':<8}"
    print(header)
    print("-" * 84)
    for name, v_acc, t_acc, v_f1, t_f1, v_t3, t_t3 in rows:
        v_acc_s = f"{v_acc*100:.2f}%" if v_acc is not None else "—"
        t_acc_s = f"{t_acc*100:.2f}%" if t_acc is not None else "—"
        v_f1_s = f"{v_f1:.4f}" if v_f1 is not None else "—"
        t_f1_s = f"{t_f1:.4f}" if t_f1 is not None else "—"
        v_t3_s = f"{v_t3*100:.2f}%" if v_t3 is not None else "—"
        t_t3_s = f"{t_t3*100:.2f}%" if t_t3 is not None else "—"
        print(f"{name:<25} | {v_acc_s:<8} {t_acc_s:<8} | {v_f1_s:<8} {t_f1_s:<8} | {v_t3_s:<8} {t_t3_s:<8}")
    print("=" * 84)
    print("Note: 8 balanced classes (100 tracks/class). Pure random guessing is 12.50%.\n")


if __name__ == "__main__":
    args = parse_args()
    evaluate_final_test(
        features_path_override=args.features,
        output_metrics_path_override=args.output_metrics,
    )


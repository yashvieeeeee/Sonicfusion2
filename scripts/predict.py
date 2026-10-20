"""Classify an MP3 audio track using a trained GenreCNN checkpoint.

Follows PLAN §09 inference contract:
- Loads self-contained checkpoint containing model weights, class names,
  preprocessing parameters, and normalization statistics.
- Decodes and segments audio into standardized 10-second windows.
- Normalizes spectrograms with saved training statistics.
- Averages per-window softmax probabilities for track-level inference.
- Flags prediction uncertainty if top score is weak or margin is narrow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.audio.loading import load_audio, segment_audio
from src.audio.spectrogram import (
    compute_logmel_spectrogram,
    normalize_spectrogram,
    to_cnn_tensor,
)
from src.config import get_project_root, load_config, resolve_device, resolve_path
from src.models.genre_cnn import GenreCNN


def parse_args() -> argparse.Namespace:
    """Parse CLI options for prediction."""
    parser = argparse.ArgumentParser(description="Classify music genre from an audio MP3 file.")
    parser.add_argument("audio_path", type=str, help="Path to input audio file (e.g. MP3 or WAV).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained genre_cnn.pt checkpoint (defaults to config path).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top genre predictions to return (default: 3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (e.g., 'cpu', 'cuda', or 'auto').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of human-readable text.",
    )
    return parser.parse_args()


def predict_genre(
    audio_path: str | Path,
    checkpoint_path: str | Path | None = None,
    top_k: int = 3,
    device_str: str | None = None,
) -> Dict[str, Any]:
    """Classify genre of an audio file using GenreCNN checkpoint.

    Returns:
        Structured prediction result dictionary.
    """
    cfg = load_config()
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found at: {audio_file}")

    ckpt_file = (
        resolve_path(checkpoint_path)
        if checkpoint_path
        else resolve_path(cfg.paths.cnn_checkpoint_path)
    )
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found at: {ckpt_file}. "
            "Please train the model first using scripts/05_train_cnn.py."
        )

    # Resolve device
    if device_str is None:
        device_str = resolve_device(cfg)
    else:
        device_str = resolve_device(device_str)
    device = torch.device(device_str)

    # Load self-contained checkpoint
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)

    class_names: List[str] = checkpoint["class_names"]
    num_classes = len(class_names)
    prep_params: Dict[str, Any] = (
        checkpoint.get("preprocessing_params")
        or checkpoint.get("preprocessing params")
        or {}
    )
    norm_stats: Dict[str, float] = (
        checkpoint.get("normalization_stats")
        or checkpoint.get("normalization stats")
        or {}
    )

    sr = int(checkpoint.get("sample_rate", 22050))
    duration = float(prep_params.get("duration", 30.0))
    win_sec = float(prep_params.get("window_duration", 10.0))
    num_windows = int(prep_params.get("windows_per_track", 3))
    n_mels = int(prep_params.get("n_mels", 128))
    n_fft = int(prep_params.get("n_fft", 2048))
    hop_length = int(prep_params.get("hop_length", 512))

    mean = float(norm_stats.get("mean", -43.0))
    std = float(norm_stats.get("std", 15.0))

    # Instantiate model and load state
    model = GenreCNN(num_classes=num_classes, in_channels=1, dropout=0.3)
    state_dict = checkpoint.get("state_dict") if "state_dict" in checkpoint else checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Audio decode, pad/truncate, and segment
    y = load_audio(audio_file, sr=sr, duration=duration, mono=True)
    segments = segment_audio(y, sr=sr, window_sec=win_sec, num_windows=num_windows)

    # Process all windows
    window_tensors: List[torch.Tensor] = []
    for seg in segments:
        raw_logmel = compute_logmel_spectrogram(
            segment=seg,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        normed = normalize_spectrogram(raw_logmel, mean=mean, std=std)
        window_tensors.append(to_cnn_tensor(normed))

    batch = torch.stack(window_tensors, dim=0).to(device)  # [num_windows, 1, 128, 431]

    with torch.no_grad():
        logits = model(batch)  # [num_windows, num_classes]
        probs = F.softmax(logits, dim=-1)  # [num_windows, num_classes]
        avg_probs = probs.mean(dim=0).cpu().numpy()  # [num_classes]

    # Rank classes by aggregated probability
    ranked_indices = np.argsort(avg_probs)[::-1]
    top_indices = ranked_indices[: min(top_k, num_classes)]

    top_predictions = [
        {"genre": class_names[idx], "score": float(avg_probs[idx])}
        for idx in top_indices
    ]

    predicted_genre = top_predictions[0]["genre"]
    top_confidence = top_predictions[0]["score"]
    second_confidence = top_predictions[1]["score"] if len(top_predictions) > 1 else 0.0

    # Uncertainty detection (PLAN §09)
    # Weak top confidence (<0.40) or narrow gap (<0.15) indicates ambiguity/multiple influences
    confidence_margin = top_confidence - second_confidence
    is_uncertain = (top_confidence < 0.40) or (confidence_margin < 0.15)

    result = {
        "audio_file": str(audio_file),
        "predicted_genre": predicted_genre,
        "confidence": top_confidence,
        "is_uncertain": bool(is_uncertain),
        "status": "uncertain / multiple influences" if is_uncertain else "confident",
        "top_predictions": top_predictions,
        "all_class_scores": {
            class_names[i]: float(avg_probs[i]) for i in ranked_indices
        },
        "model_version": checkpoint.get("version", "1.0"),
    }

    return result


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    res = predict_genre(
        audio_path=args.audio_path,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        device_str=args.device,
    )

    if args.json:
        print(json.dumps(res, indent=2))
        return

    print("=" * 60)
    print("             SONICSCOPE AI: GENRE PREDICTION           ")
    print("=" * 60)
    print(f"File             : {res['audio_file']}")
    print(f"Predicted Genre  : {res['predicted_genre']}")
    print(f"Top Confidence   : {res['confidence'] * 100:.1f}%")
    print(f"Classification   : {res['status'].upper()}")

    print(f"\n--- Top Predictions ---")
    for i, item in enumerate(res["top_predictions"], 1):
        bar = "#" * int(item["score"] * 30)
        print(f"  {i}. {item['genre']:<15} {item['score']*100:>5.1f}% | {bar}")

    if res["is_uncertain"]:
        print(
            "\n[NOTE] Prediction uncertainty detected: model score is below threshold "
            "or margin between top genres is narrow. Song may feature blended musical styles."
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

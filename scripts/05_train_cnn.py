"""Train the original 2D Log-Mel Spectrogram CNN for music genre recognition.

Follows PLAN §09 (Architecture & Reproducibility) and PLAN §18 (Pilot Disclosure):
- Exact 3-block 2D CNN architecture.
- Optimizer: AdamW (lr=1e-3, weight_decay=0.01).
- Optional LR scheduler: ReduceLROnPlateau.
- Loss: CrossEntropyLoss.
- Batch size: 16 (default).
- Training-only time-shift and light gain augmentation.
- Early stopping on validation track-level macro F1.
- Evaluates validation metrics by averaging window probabilities per track.
- Per-epoch atomic history logging to reports/metrics/cnn_history.json.
- Per-epoch state checkpointing to models/checkpoints/genre_cnn_last.pt.
- Resumability (--resume) with SHA-256 validation, RNG, scheduler, and early-stopping counter restoration.
- Safe archiving: fresh runs archive existing artifacts (*_prev_<timestamp>), never overwrite or delete.
- Graceful exit on KeyboardInterrupt / SystemExit (status: interrupted) or errors (status: failed).
- Comprehensive reproducibility metadata saved to checkpoint and reports/metrics/cnn_run_metadata.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.audio.loading import load_audio, segment_audio
from src.audio.spectrogram import (
    compute_logmel_spectrogram,
    load_norm_stats,
    normalize_spectrogram,
    to_cnn_tensor,
)
from src.config import get_project_root, load_config, resolve_device, resolve_path
from src.models.genre_cnn import GenreCNN, apply_spectrogram_augmentation


def compute_file_sha256(filepath: Path) -> str:
    """Compute hex SHA-256 digest of a file; returns empty string if not found."""
    if not filepath.exists() or not filepath.is_file():
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_state(project_root: Path) -> Tuple[str, bool]:
    """Retrieve current git commit hash and dirty flag computed on TRACKED files only."""
    commit_hash = "unknown"
    is_dirty = False
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            commit_hash = res.stdout.strip()

        # Tracked files only (--untracked-files=no)
        res_dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res_dirty.returncode == 0:
            is_dirty = len(res_dirty.stdout.strip()) > 0
    except Exception:
        pass
    return commit_hash, is_dirty


def get_package_versions() -> Dict[str, str]:
    """Collect runtime library versions for reproducibility tracking."""
    sklearn_ver = "unknown"
    try:
        import sklearn
        sklearn_ver = sklearn.__version__
    except Exception:
        pass

    torchaudio_ver = "unknown"
    try:
        import torchaudio
        torchaudio_ver = torchaudio.__version__
    except Exception:
        pass

    librosa_ver = "unknown"
    try:
        import librosa
        librosa_ver = librosa.__version__
    except Exception:
        pass

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchaudio": torchaudio_ver,
        "librosa": librosa_ver,
        "numpy": np.__version__,
        "scikit-learn": sklearn_ver,
    }


def write_json_atomic(data: Any, target_path: Path) -> None:
    """Atomically write JSON data using a temporary file replaced in-place."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f".tmp_{os.getpid()}_{int(time.time() * 1000)}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def handle_fresh_run_archiving(
    checkpoint_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    cnn_checkpoint_path: Path,
    last_checkpoint_path: Path,
    fresh: bool,
) -> None:
    """Archive previous run artifacts before writing fresh data. Nothing is ever deleted."""
    if last_checkpoint_path.exists() and not fresh:
        raise RuntimeError(
            f"Existing checkpoint found at {last_checkpoint_path}. "
            "Pass --resume to continue training, or --fresh to archive previous run artifacts and start fresh."
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # 1. Archive genre_cnn.pt -> pilot_cnn_12epoch_interrupted.pt or genre_cnn_prev_<ts>.pt
    if cnn_checkpoint_path.exists():
        target_pilot = checkpoint_dir / "pilot_cnn_12epoch_interrupted.pt"
        if not target_pilot.exists():
            shutil.move(str(cnn_checkpoint_path), str(target_pilot))
            print(f"Archived existing checkpoint: {cnn_checkpoint_path.name} -> {target_pilot.name}")
        else:
            target_prev = checkpoint_dir / f"genre_cnn_prev_{ts}.pt"
            shutil.move(str(cnn_checkpoint_path), str(target_prev))
            print(f"Archived existing checkpoint: {cnn_checkpoint_path.name} -> {target_prev.name}")

    # 2. Archive genre_cnn_last.pt if fresh was requested
    if last_checkpoint_path.exists() and fresh:
        target_last_prev = checkpoint_dir / f"genre_cnn_last_prev_{ts}.pt"
        shutil.move(str(last_checkpoint_path), str(target_last_prev))
        print(f"Archived existing checkpoint: {last_checkpoint_path.name} -> {target_last_prev.name}")

    # 3. Archive metrics files
    for fname in ["cnn_history.json", "cnn_val.json", "cnn_run_metadata.json"]:
        fpath = metrics_dir / fname
        if fpath.exists():
            target_fpath = metrics_dir / f"{fpath.stem}_prev_{ts}{fpath.suffix}"
            shutil.move(str(fpath), str(target_fpath))
            print(f"Archived previous metrics: {fpath.name} -> {target_fpath.name}")

    # 4. Archive training curves
    curves_path = figures_dir / "cnn_training_curves.png"
    if curves_path.exists():
        target_curves = figures_dir / f"cnn_training_curves_prev_{ts}.png"
        shutil.move(str(curves_path), str(target_curves))
        print(f"Archived previous curves: {curves_path.name} -> {target_curves.name}")


class SpectrogramWindowDataset(Dataset):
    """Dataset serving 10-second normalized spectrogram windows."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        label_to_idx: Dict[str, int],
        cache_dir: Path,
        norm_mean: float,
        norm_std: float,
        sr: int = 22050,
        duration: float = 30.0,
        window_sec: float = 10.0,
        num_windows: int = 3,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        is_train: bool = False,
    ) -> None:
        self.label_to_idx = label_to_idx
        self.cache_dir = cache_dir
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sr = sr
        self.duration = duration
        self.window_sec = window_sec
        self.num_windows = num_windows
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.is_train = is_train

        # Build list of (track_id, window_idx, path, label_idx)
        self.samples: List[Tuple[int, int, str, int]] = []
        for _, row in manifest_df.iterrows():
            t_id = int(row["track_id"])
            genre = str(row["genre"])
            p_str = str(row["path"])
            label_idx = self.label_to_idx[genre]
            for w in range(self.num_windows):
                self.samples.append((t_id, w, p_str, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        t_id, w_idx, rel_path, label_idx = self.samples[idx]

        cache_file = self.cache_dir / f"{t_id:06d}_{w_idx}.npy"
        if cache_file.exists():
            spec_arr = np.load(cache_file).astype(np.float32)
        else:
            # Fallback: compute on the fly if not yet cached
            full_path = resolve_path(rel_path)
            y = load_audio(full_path, sr=self.sr, duration=self.duration, mono=True)
            segments = segment_audio(y, sr=self.sr, window_sec=self.window_sec, num_windows=self.num_windows)
            raw_logmel = compute_logmel_spectrogram(
                segments[w_idx],
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            spec_arr = normalize_spectrogram(raw_logmel, mean=self.norm_mean, std=self.norm_std)

        tensor = to_cnn_tensor(spec_arr)  # [1, 128, 431]

        if self.is_train:
            tensor = apply_spectrogram_augmentation(tensor, max_time_shift=40, gain_db_range=2.0)

        return tensor, label_idx, t_id, w_idx


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Train original 2D CNN on log-mel spectrograms.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of manifest tracks for testing.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on validation macro F1.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', 'auto'). Defaults to config.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument("--resume", action="store_true", help="Resume training from models/checkpoints/genre_cnn_last.pt.")
    parser.add_argument("--fresh", action="store_true", help="Archive existing checkpoint/metrics and start fresh run.")
    return parser.parse_args()


def train_cnn(
    epochs_override: Optional[int] = None,
    batch_size: int = 16,
    limit: Optional[int] = None,
    patience: int = 5,
    device_override: Optional[str] = None,
    config_path: Optional[str] = None,
    resume: bool = False,
    fresh: bool = False,
    checkpoint_dir_override: Optional[Path] = None,
    metrics_dir_override: Optional[Path] = None,
    figures_dir_override: Optional[Path] = None,
    manifest_path_override: Optional[Path] = None,
    cache_dir_override: Optional[Path] = None,
    norm_stats_path_override: Optional[Path] = None,
    label_map_path_override: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train 2D CNN with atomic per-epoch logging, early stopping, and resumability."""
    cfg = load_config(config_path)
    root = get_project_root()

    manifest_path = manifest_path_override or resolve_path(cfg.paths.manifest_path)
    label_map_path = label_map_path_override or resolve_path(cfg.paths.label_map_path)
    norm_stats_path = norm_stats_path_override or resolve_path(cfg.paths.norm_stats_path)
    cache_dir = cache_dir_override or resolve_path(cfg.paths.spectrogram_cache_dir)
    config_file_path = resolve_path(config_path) if config_path else root / "configs" / "config.yaml"

    checkpoint_dir = checkpoint_dir_override or resolve_path(cfg.paths.cnn_checkpoint_path).parent
    checkpoint_path = checkpoint_dir / "genre_cnn.pt"
    last_checkpoint_path = checkpoint_dir / "genre_cnn_last.pt"

    figures_dir = figures_dir_override or resolve_path(cfg.paths.figures_dir)
    metrics_dir = metrics_dir_override or resolve_path(cfg.paths.metrics_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history_json_path = metrics_dir / "cnn_history.json"
    metrics_json_path = metrics_dir / "cnn_val.json"
    run_metadata_path = metrics_dir / "cnn_run_metadata.json"
    curves_plot_path = figures_dir / "cnn_training_curves.png"

    # Compute current SHA-256 hashes of input artifacts
    current_sha256 = {
        "config": compute_file_sha256(config_file_path),
        "manifest": compute_file_sha256(manifest_path),
        "label_map": compute_file_sha256(label_map_path),
        "norm_stats": compute_file_sha256(norm_stats_path),
    }

    # Handle Archiving / Fresh Run Validation
    if not resume:
        handle_fresh_run_archiving(
            checkpoint_dir=checkpoint_dir,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            cnn_checkpoint_path=checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            fresh=fresh,
        )

    # Load metadata
    manifest_df = pd.read_csv(manifest_path)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_to_idx: Dict[str, int] = json.load(f)

    class_names = [k for k, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    norm_stats = load_norm_stats(norm_stats_path)
    mean = norm_stats["mean"]
    std = norm_stats["std"]

    # Filter splits
    train_manifest = manifest_df[manifest_df["split"] == "training"]
    val_manifest = manifest_df[manifest_df["split"] == "validation"]
    test_manifest = manifest_df[manifest_df["split"] == "test"]

    if limit is not None and limit > 0:
        train_count = max(2, int(limit * 0.8))
        val_count = max(2, int(limit * 0.2))
        train_manifest = train_manifest.iloc[:train_count]
        val_manifest = val_manifest.iloc[:val_count]

    print(f"Dataset partitions for CNN training:")
    print(f"  Training tracks    : {len(train_manifest)} ({len(train_manifest) * 3} windows)")
    print(f"  Validation tracks  : {len(val_manifest)} ({len(val_manifest) * 3} windows)")
    print(f"  Test tracks        : {len(test_manifest)} (EXCLUDED: Not touched during training)")

    train_dataset = SpectrogramWindowDataset(
        manifest_df=train_manifest,
        label_to_idx=label_to_idx,
        cache_dir=cache_dir,
        norm_mean=mean,
        norm_std=std,
        is_train=True,
    )
    val_dataset = SpectrogramWindowDataset(
        manifest_df=val_manifest,
        label_to_idx=label_to_idx,
        cache_dir=cache_dir,
        norm_mean=mean,
        norm_std=std,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Device & Model Setup
    device_str = resolve_device(device_override if device_override is not None else cfg)
    device = torch.device(device_str)
    print(f"\nUsing device: {device}")

    epochs = (
        epochs_override
        if epochs_override is not None
        else int(cfg.cnn.training.epochs)
    )
    lr = float(cfg.cnn.training.learning_rate)
    weight_decay = float(cfg.cnn.training.weight_decay)
    seed = int(getattr(cfg, "seed", 42))

    # Set seeds initially
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = GenreCNN(num_classes=num_classes, in_channels=1, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )

    history_records: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_val_acc = 0.0
    best_classification_report: Dict[str, Any] = {}
    epochs_no_improve = 0
    start_epoch = 1

    # Checkpoint Resume logic
    if resume:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume: checkpoint {last_checkpoint_path} not found.")

        print(f"\nResuming training from {last_checkpoint_path.name}...")
        ckpt = torch.load(last_checkpoint_path, map_location=device, weights_only=False)

        # 1. SHA-256 validation
        ckpt_meta = ckpt.get("metadata", {})
        ckpt_hashes = ckpt_meta.get("sha256_hashes", {})
        mismatches = []
        for k in ["config", "manifest", "label_map", "norm_stats"]:
            if k in ckpt_hashes and current_sha256[k] != ckpt_hashes[k]:
                mismatches.append(f"{k} (checkpoint: {ckpt_hashes[k][:8]}..., current: {current_sha256[k][:8]}...)")

        if mismatches:
            raise ValueError(
                f"Cannot resume training: SHA-256 hash mismatch for: {', '.join(mismatches)}. "
                "The training data, label map, norm stats, or config has changed since the checkpoint was saved."
            )

        # 2. Restore weights, optimizer, scheduler
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # 3. Restore RNG states
        if "rng_state_python" in ckpt:
            random.setstate(ckpt["rng_state_python"])
        if "rng_state_numpy" in ckpt:
            np.random.set_state(ckpt["rng_state_numpy"])
        if "rng_state_torch" in ckpt:
            torch.set_rng_state(ckpt["rng_state_torch"])
        if torch.cuda.is_available() and ckpt.get("rng_state_torch_cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_state_torch_cuda"])

        # 4. Restore counters and history
        start_epoch = int(ckpt["epoch"]) + 1
        best_epoch = int(ckpt.get("best_epoch", -1))
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        best_val_acc = float(ckpt.get("best_val_accuracy", 0.0))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        history_records = list(ckpt.get("history", []))

        print(f"Resumed at Epoch {start_epoch} (Best Epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}, Patience Counter: {epochs_no_improve}/{patience})")

    # Common preprocessing params contract
    prep_params = {
        "duration": float(cfg.audio.duration),
        "window_duration": float(cfg.audio.window_duration),
        "windows_per_track": int(cfg.audio.windows_per_track),
        "n_mels": int(cfg.audio.n_mels),
        "n_fft": int(cfg.audio.n_fft),
        "hop_length": int(cfg.audio.hop_length),
        "power": float(cfg.audio.power),
        "mono": bool(cfg.audio.mono),
    }
    norm_stats_dict = {
        "mean": float(mean),
        "std": float(std),
    }

    git_commit, git_dirty = get_git_state(root)
    pkg_versions = get_package_versions()

    status = "in_progress"
    error_info: Optional[Dict[str, str]] = None

    print(f"\nBeginning CNN training for {epochs} epochs (starting at {start_epoch}, patience={patience})...")

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_t0 = time.time()

            # 1. Training Phase
            model.train()
            running_train_loss = 0.0
            train_batches = 0

            for batch_x, batch_y, _, _ in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                train_batches += 1

            avg_train_loss = running_train_loss / max(1, train_batches)

            # 2. Validation Phase (track-level probability aggregation)
            model.eval()
            running_val_loss = 0.0
            val_batches = 0

            track_window_probs: Dict[int, List[np.ndarray]] = {}
            track_true_labels: Dict[int, int] = {}

            with torch.no_grad():
                for batch_x, batch_y, batch_tids, _ in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    running_val_loss += loss.item()
                    val_batches += 1

                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    y_cpu = batch_y.cpu().numpy()
                    tids_cpu = batch_tids.numpy()

                    for p, true_y, t_id in zip(probs, y_cpu, tids_cpu):
                        if t_id not in track_window_probs:
                            track_window_probs[t_id] = []
                            track_true_labels[t_id] = int(true_y)
                        track_window_probs[t_id].append(p)

            avg_val_loss = running_val_loss / max(1, val_batches)

            # Aggregate track-level predictions
            y_val_true: List[int] = []
            y_val_pred: List[int] = []

            for t_id, prob_list in track_window_probs.items():
                avg_prob = np.mean(prob_list, axis=0)
                pred_class = int(np.argmax(avg_prob))
                y_val_true.append(track_true_labels[t_id])
                y_val_pred.append(pred_class)

            val_macro_f1 = float(f1_score(y_val_true, y_val_pred, average="macro", zero_division=0))
            val_accuracy = float(accuracy_score(y_val_true, y_val_pred))

            # Step scheduler based on validation macro F1
            scheduler.step(val_macro_f1)
            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_elapsed_sec = time.time() - epoch_t0

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "val_accuracy": float(val_accuracy),
                "val_macro_f1": float(val_macro_f1),
                "learning_rate": float(current_lr),
                "seconds": float(round(epoch_elapsed_sec, 2)),
            }
            history_records.append(epoch_record)

            # 3. ATOMIC WRITE: cnn_history.json after EVERY epoch
            write_json_atomic(history_records, history_json_path)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Macro F1: {val_macro_f1:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"{epoch_elapsed_sec:.1f}s"
            )

            # Check if this epoch is the new best
            is_best = val_macro_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_macro_f1
                best_epoch = epoch
                best_val_acc = val_accuracy
                epochs_no_improve = 0

                target_names = [class_names[i] for i in sorted(list(set(y_val_true + y_val_pred)))]
                best_classification_report = classification_report(
                    y_val_true,
                    y_val_pred,
                    labels=sorted(list(set(y_val_true + y_val_pred))),
                    target_names=target_names,
                    output_dict=True,
                    zero_division=0,
                )

                # Save best checkpoint (genre_cnn.pt)
                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "sample_rate": int(cfg.audio.sample_rate),
                    "preprocessing_params": prep_params,
                    "normalization_stats": norm_stats_dict,
                    "version": "1.0",
                    "best_epoch": best_epoch,
                    "validation_macro_f1": best_val_f1,
                    "validation_accuracy": best_val_acc,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": {
                        "seed": seed,
                        "device": str(device),
                        "epochs_configured": epochs,
                        "best_epoch": best_epoch,
                        "package_versions": pkg_versions,
                        "git": {"commit_hash": git_commit, "is_dirty": git_dirty},
                        "sha256_hashes": current_sha256,
                        "label_to_idx": label_to_idx,
                    },
                }
                torch.save(best_checkpoint, checkpoint_path)
                print(f"  -> Best checkpoint updated (Epoch {best_epoch}, Val Macro F1: {best_val_f1:.4f})")
            else:
                epochs_no_improve += 1

            # 4. Save genre_cnn_last.pt after EVERY epoch
            last_checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "rng_state_python": random.getstate(),
                "rng_state_numpy": np.random.get_state(),
                "rng_state_torch": torch.get_rng_state(),
                "rng_state_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "best_val_accuracy": best_val_acc,
                "epochs_no_improve": epochs_no_improve,
                "history": history_records,
                "class_names": class_names,
                "sample_rate": int(cfg.audio.sample_rate),
                "preprocessing_params": prep_params,
                "normalization_stats": norm_stats_dict,
                "version": "1.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "seed": seed,
                    "device": str(device),
                    "epochs_configured": epochs,
                    "epochs_run": len(history_records),
                    "best_epoch": best_epoch,
                    "package_versions": pkg_versions,
                    "git": {"commit_hash": git_commit, "is_dirty": git_dirty},
                    "sha256_hashes": current_sha256,
                    "label_to_idx": label_to_idx,
                },
            }
            torch.save(last_checkpoint, last_checkpoint_path)

            # Early stopping check
            if epochs_no_improve >= patience:
                status = "early_stopped"
                print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                break
        else:
            if status == "in_progress":
                status = "completed"

    except (KeyboardInterrupt, SystemExit):
        status = "interrupted"
        print("\n[NOTICE] CNN training interrupted by user/signal. Finalizing artifacts...")
    except Exception as exc:
        status = "failed"
        error_info = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        print(f"\n[ERROR] CNN training failed with {type(exc).__name__}: {exc}. Finalizing artifacts...")
        raise
    finally:
        # Finalize and save artifacts even on interruption or failure
        _finalize_run_artifacts(
            status=status,
            error_info=error_info,
            history_records=history_records,
            best_epoch=best_epoch,
            best_val_f1=best_val_f1,
            best_classification_report=best_classification_report,
            num_classes=num_classes,
            class_names=class_names,
            train_count=len(train_manifest),
            val_count=len(val_manifest),
            epochs_configured=epochs,
            seed=seed,
            device_str=str(device),
            pkg_versions=pkg_versions,
            git_commit=git_commit,
            git_dirty=git_dirty,
            current_sha256=current_sha256,
            label_to_idx=label_to_idx,
            cfg=cfg,
            metrics_json_path=metrics_json_path,
            run_metadata_path=run_metadata_path,
            curves_plot_path=curves_plot_path,
            history_json_path=history_json_path,
            root=root,
        )

    return {
        "status": status,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "epochs_run": len(history_records),
    }


def _finalize_run_artifacts(
    status: str,
    error_info: Optional[Dict[str, str]],
    history_records: List[Dict[str, Any]],
    best_epoch: int,
    best_val_f1: float,
    best_classification_report: Dict[str, Any],
    num_classes: int,
    class_names: List[str],
    train_count: int,
    val_count: int,
    epochs_configured: int,
    seed: int,
    device_str: str,
    pkg_versions: Dict[str, str],
    git_commit: str,
    git_dirty: bool,
    current_sha256: Dict[str, str],
    label_to_idx: Dict[str, int],
    cfg: Any,
    metrics_json_path: Path,
    run_metadata_path: Path,
    curves_plot_path: Path,
    history_json_path: Path,
    root: Path,
) -> None:
    """Save training curves, cnn_val.json, and cnn_run_metadata.json with run status."""
    # Ensure history is flushed
    write_json_atomic(history_records, history_json_path)

    # 1. Save training curves if at least one epoch was completed
    if history_records:
        history_dict = {
            "train_loss": [r["train_loss"] for r in history_records],
            "val_loss": [r["val_loss"] for r in history_records],
            "val_macro_f1": [r["val_macro_f1"] for r in history_records],
            "val_accuracy": [r["val_accuracy"] for r in history_records],
        }
        _plot_and_save_curves(history_dict, curves_plot_path, best_epoch)
        print(f"Saved CNN training curves to {curves_plot_path}.")

    # 2. Save cnn_val.json if at least one epoch ran or best_epoch > 0
    if history_records or best_epoch > 0:
        history_dict = {
            "train_loss": [r["train_loss"] for r in history_records],
            "val_loss": [r["val_loss"] for r in history_records],
            "val_macro_f1": [r["val_macro_f1"] for r in history_records],
            "val_accuracy": [r["val_accuracy"] for r in history_records],
        }
        val_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "error_info": error_info,
            "model_architecture": "GenreCNN (3-block 2D CNN)",
            "num_classes": num_classes,
            "class_names": class_names,
            "best_epoch": best_epoch,
            "epochs_run": len(history_records),
            "epochs_configured": epochs_configured,
            "best_validation_macro_f1": best_val_f1,
            "classification_report": best_classification_report,
            "training_history": history_dict,
            "num_train_tracks": train_count,
            "num_val_tracks": val_count,
            "note": "Evaluated strictly on the official FMA validation split. The test split was not evaluated.",
        }
        write_json_atomic(val_payload, metrics_json_path)
        print(f"Saved CNN validation metrics (status: {status}) to {metrics_json_path}.")

    # 3. Save cnn_run_metadata.json
    config_dict = cfg.dict() if hasattr(cfg, "dict") else (dict(cfg) if isinstance(cfg, dict) else {})
    meta_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error_info": error_info,
        "seed": seed,
        "device": device_str,
        "epochs_configured": epochs_configured,
        "epochs_run": len(history_records),
        "best_epoch": best_epoch,
        "early_stopped": status == "early_stopped",
        "git": {
            "commit_hash": git_commit,
            "is_dirty": git_dirty,
        },
        "package_versions": pkg_versions,
        "sha256_hashes": current_sha256,
        "label_to_idx": label_to_idx,
        "class_names": class_names,
        "config_snapshot": config_dict,
    }
    write_json_atomic(meta_payload, run_metadata_path)
    print(f"Saved CNN run metadata (status: {status}) to {run_metadata_path}.")


def _plot_and_save_curves(
    history: Dict[str, List[float]],
    output_path: Path,
    best_epoch: int,
) -> None:
    """Render and save training loss and validation macro-F1 plots."""
    epochs = range(1, len(history["train_loss"]) + 1)
    if not epochs:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Metric curve
    ax2.plot(epochs, history["val_macro_f1"], "g-o", label="Val Macro F1")
    ax2.plot(epochs, history["val_accuracy"], "m--^", label="Val Accuracy")
    if best_epoch > 0:
        ax2.axvline(best_epoch, color="k", linestyle=":", label=f"Best Checkpoint (Ep {best_epoch})")
    ax2.set_title("Validation Evaluation Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    train_cnn(
        epochs_override=args.epochs,
        batch_size=args.batch_size,
        limit=args.limit,
        patience=args.patience,
        device_override=args.device,
        config_path=args.config,
        resume=args.resume,
        fresh=args.fresh,
    )

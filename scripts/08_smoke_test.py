                                                """End-to-End System Smoke Test for SonicScope AI (SonicFusion 2).

Follows docs/PLAN.md §17, §18, §22:
Validates FFmpeg availability, configuration, dataset manifest & label map,
classical baseline & GenreCNN checkpoints, API /api/health, and a live /api/analyze call.
Exits with 0 on success, 1 on failure, printing clear actionable error messages.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import soundfile as sf
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
    print(f"\n{Color.CYAN}{Color.BOLD}{'=' * 72}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}  {title}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'=' * 72}{Color.RESET}")


def run_check(name: str, check_fn) -> Tuple[bool, str]:
    """Execute a single check function and return pass/fail with detail."""
    try:
        msg = check_fn()
        return True, msg or "Passed"
    except Exception as exc:
        return False, str(exc)


def create_in_memory_mp3(duration_sec: float = 3.0, sr: int = 22050, freq: float = 440.0) -> bytes:
    """Generate a clean synthetic MP3 in memory for analyze route testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    signal = 0.3 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="MP3")
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="SonicScope AI (SonicFusion 2) End-to-End Smoke Test")
    parser.add_argument("--skip-api", action="store_true", help="Skip FastAPI HTTP endpoint verification")
    parser.add_argument("--verbose", action="store_true", help="Print detailed diagnostic messages")
    args = parser.parse_args()

    start_time = time.time()
    results: List[Tuple[str, bool, str]] = []

    print_header("SonicFusion 2")

    # -------------------------------------------------------------
    # 1. FFmpeg Audio Codec Verification (PLAN §03, §17)
    # -------------------------------------------------------------
    def check_ffmpeg():
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(
                "FFmpeg executable not found on system PATH. "
                "FFmpeg is required for audio decoding and magic-byte stream validation. "
                "Please install FFmpeg and verify 'ffmpeg -version' works in your shell."
            )
        res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"FFmpeg binary failed execution with code {res.returncode}: {res.stderr.strip()}"
            )
        first_line = res.stdout.splitlines()[0] if res.stdout else "FFmpeg present"
        return f"Binary: ffmpeg | {first_line[:48]}"

    passed, msg = run_check("1. Audio Codec (FFmpeg)", check_ffmpeg)
    results.append(("Audio Codec (FFmpeg)", passed, msg))

    # -------------------------------------------------------------
    # 2. Configuration & Path Resolution (PLAN §05, §07)
    # -------------------------------------------------------------
    def check_config():
        cfg = load_config()
        root = get_project_root()
        cfg_file = root / "configs" / "config.yaml"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Configuration file missing at {cfg_file.name}")
        if cfg.audio.sample_rate != 22050:
            raise ValueError(f"Expected 22,050 Hz sample rate, got {cfg.audio.sample_rate}")
        if cfg.audio.duration != 30.0:
            raise ValueError(f"Expected 30.0s window duration contract, got {cfg.audio.duration}")
        if cfg.audio.n_mels != 128:
            raise ValueError(f"Expected 128 mel frequency bins, got {cfg.audio.n_mels}")
        return f"Loaded configs/config.yaml | SR: {cfg.audio.sample_rate}Hz, Mels: {cfg.audio.n_mels}"

    passed, msg = run_check("2. Configuration & Paths", check_config)
    results.append(("Configuration & Paths", passed, msg))

    # -------------------------------------------------------------
    # 3. Dataset Manifest & Label Map (PLAN §06)
    # -------------------------------------------------------------
    def check_manifest():
        root = get_project_root()
        label_map_file = root / "data" / "processed" / "label_map.json"
        manifest_csv = root / "data" / "processed" / "manifest.csv"

        if not label_map_file.exists():
            raise FileNotFoundError("data/processed/label_map.json missing. Run scripts/01_build_manifest.py.")
        import json
        with open(label_map_file, "r", encoding="utf-8") as f:
            lmap = json.load(f)
        if len(lmap) != 8:
            raise ValueError(f"Expected 8 genres in label map, got {len(lmap)}")

        if not manifest_csv.exists():
            raise FileNotFoundError("data/processed/manifest.csv missing. Run scripts/01_build_manifest.py.")
        import pandas as pd
        df = pd.read_csv(manifest_csv)
        if len(df) == 0:
            raise ValueError("manifest.csv is empty")
        for col in ["track_id", "genre", "split", "path"]:
            if col not in df.columns:
                raise KeyError(f"Missing required column '{col}' in manifest.csv")

        return f"Label map: 8 classes | Manifest: {len(df)} tracks ({manifest_csv.name})"

    passed, msg = run_check("3. Dataset Manifest & Taxonomy", check_manifest)
    results.append(("Manifest & Label Map", passed, msg))

    # -------------------------------------------------------------
    # 4. Checkpoints & Trained Models (PLAN §08, §09)
    # -------------------------------------------------------------
    def check_checkpoints():
        root = get_project_root()
        cfg = load_config()

        # 4a. Classical Baseline
        rf_path = resolve_path(cfg.paths.baseline_model_path)
        if not rf_path.exists():
            raise FileNotFoundError(
                f"Random Forest baseline model not found at {rf_path.name}. "
                "Train using scripts/04_train_baseline.py."
            )
        rf_pipe = joblib.load(rf_path)
        if not hasattr(rf_pipe, "predict"):
            raise TypeError("Loaded baseline artifact is not a valid scikit-learn estimator")

        # 4b. 2D CNN Checkpoint
        cnn_path = resolve_path(cfg.paths.cnn_checkpoint_path)
        if not cnn_path.exists():
            raise FileNotFoundError(
                f"GenreCNN checkpoint not found at {cnn_path.name}. "
                "Train using scripts/05_train_cnn.py."
            )
        ckpt = torch.load(cnn_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if not state_dict:
            raise KeyError("Checkpoint missing 'model_state_dict' or 'state_dict'")

        model = GenreCNN(num_classes=8, in_channels=1, dropout=0.3)
        model.load_state_dict(state_dict)
        model.eval()

        # Dry-run forward pass
        dummy_tensor = torch.randn(1, 1, 128, 431)
        with torch.no_grad():
            logits = model(dummy_tensor)
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        if logits.shape != (1, 8):
            raise ValueError(f"Expected logits shape (1, 8), got {logits.shape}")

        return f"RF: {rf_path.name} | CNN: {cnn_path.name} (Params: {sum(p.numel() for p in model.parameters()):,})"

    passed, msg = run_check("4. Trained Model Checkpoints", check_checkpoints)
    results.append(("Model Checkpoints (RF & CNN)", passed, msg))

    # -------------------------------------------------------------
    # 5. FastAPI /api/health Endpoint (PLAN §13)
    # -------------------------------------------------------------
    if not args.skip_api:
        def check_health_api():
            client = TestClient(app)
            res = client.get("/api/health")
            if res.status_code != 200:
                raise RuntimeError(f"GET /api/health returned HTTP {res.status_code}: {res.text}")
            data = res.json()
            if data.get("status") != "ok":
                raise RuntimeError(f"Backend status not 'ok': {data.get('detail', 'Unknown error')}")
            if not data.get("model_loaded"):
                raise RuntimeError("Backend reported model_loaded=False")
            return f"Status: {data['status']} | Model Loaded: True | Device: {data.get('device', 'cpu')}"

        passed, msg = run_check("5. API Health (/api/health)", check_health_api)
        results.append(("API Health (/api/health)", passed, msg))

        # -------------------------------------------------------------
        # 6. Live /api/analyze Upload Inference Call (PLAN §13)
        # -------------------------------------------------------------
        def check_analyze_api():
            client = TestClient(app)
            root = get_project_root()
            sample_file = root / "web" / "public" / "samples" / "sample_000002.mp3"

            if sample_file.exists():
                with open(sample_file, "rb") as f:
                    file_bytes = f.read()
                filename = sample_file.name
            else:
                # Fallback to generated valid MP3
                file_bytes = create_in_memory_mp3(duration_sec=5.0)
                filename = "smoke_test_clip.mp3"

            res = client.post("/api/analyze", files={"file": (filename, file_bytes, "audio/mpeg")})
            if res.status_code != 200:
                raise RuntimeError(
                    f"POST /api/analyze failed with HTTP {res.status_code}: {res.text}"
                )
            data = res.json()
            for key in ["genre", "acoustic_features", "visualizations", "analysis_id"]:
                if key not in data:
                    raise KeyError(f"Response missing expected key '{key}'")

            top_label = data["genre"]["top_label"]
            conf = data["genre"]["confidence"]
            scores_count = len(data["genre"]["scores"])
            return f"Top Genre: {top_label} ({conf * 100:.1f}%) | Scores: {scores_count} classes | Status 200"

        passed, msg = run_check("6. End-to-End Inference (/api/analyze)", check_analyze_api)
        results.append(("Live Analyze Call (/api/analyze)", passed, msg))
    else:
        results.append(("API Health (/api/health)", True, "Skipped (--skip-api)"))
        results.append(("Live Analyze Call (/api/analyze)", True, "Skipped (--skip-api)"))

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    print_header("Smoke Test Results Summary")
    all_passed = True
    for name, ok, detail in results:
        status_text = f"{Color.GREEN}[PASS]{Color.RESET}" if ok else f"{Color.RED}[FAIL]{Color.RESET}"
        print(f" {status_text}  {name:<34} {Color.YELLOW}{detail}{Color.RESET}")
        if not ok:
            all_passed = False

    elapsed = time.time() - start_time
    print(f"\nExecution finished in {elapsed:.2f} seconds.")

    if all_passed:
        print(f"{Color.GREEN}{Color.BOLD}>>> ALL SMOKE TESTS PASSED! Pipeline is fully verified. <<<{Color.RESET}\n")
        return 0
    else:
        print(f"{Color.RED}{Color.BOLD}>>> SMOKE TEST FAILED! Review failure messages above. <<<{Color.RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Master pipeline runner for the complete FMA Small dataset (~8,000 tracks).

Runs sequentially:
1. scripts/02_extract_features.py --resume --workers 8
2. scripts/03_cache_spectrograms.py --resume --workers 8
3. scripts/04_train_baseline.py
4. scripts/05_train_cnn.py
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = sys.executable


def run_stage(description: str, cmd: list[str]) -> None:
    print("\n" + "=" * 70)
    print(f"STAGE: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"STARTED AT: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n", flush=True)

    t0 = time.time()
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"\n[ERROR] Stage '{description}' failed with exit code {res.returncode}.", flush=True)
        sys.exit(res.returncode)

    print(f"\n[SUCCESS] Stage '{description}' completed in {elapsed:.1f}s ({elapsed/60:.1f} min).\n", flush=True)


def main() -> None:
    print("=" * 70)
    print("   SONICSCOPE AI: COMPLETE DATASET PIPELINE (FMA SMALL ~8,000 TRACKS)  ")
    print("=" * 70)

    # 1. Feature extraction
    run_stage(
        "Handcrafted Acoustic Feature Extraction",
        [PYTHON_EXE, "scripts/02_extract_features.py", "--resume", "--workers", "8"],
    )

    # 2. Spectrogram caching
    run_stage(
        "Log-Mel Spectrogram Window Caching",
        [PYTHON_EXE, "scripts/03_cache_spectrograms.py", "--resume", "--workers", "8"],
    )

    # 3. Train Baseline
    run_stage(
        "Random Forest Classical Baseline Training & Tuning",
        [PYTHON_EXE, "scripts/04_train_baseline.py"],
    )

    # 4. Train CNN
    run_stage(
        "2D Log-Mel GenreCNN Deep Learning Training",
        [PYTHON_EXE, "scripts/05_train_cnn.py"],
    )

    print("\n" + "=" * 70)
    print("ALL STAGES COMPLETED SUCCESSFULLY ON FULL DATASET!")
    print("=" * 70)


if __name__ == "__main__":
    main()

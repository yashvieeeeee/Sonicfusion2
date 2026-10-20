# SonicScope AI (SonicFusion 2)

> **Music Intelligence & Genre Recognition System:** Comparative study evaluating classical feature-engineered machine learning (Random Forest) against spectrogram-based deep learning (2D CNN) on the Free Music Archive (FMA Small).

The single source of truth for architectural decisions, contracts, and requirements is [`docs/PLAN.md`](docs/PLAN.md).

---

## 1. Project Overview

SonicScope AI is an end-to-end full-stack music information retrieval (MIR) platform:
- **Audio Processing**: Standardized 22,050 Hz mono decoding, 10-second non-overlapping windows, and 128 mel-band log-mel spectrograms `(1, 128, 431)`.
- **Classical Baseline**: Tuned Random Forest (400 estimators) operating on 51-dimensional handcrafted acoustic features (MFCCs, chroma, spectral centroid/rolloff/bandwidth/flatness, RMS energy, and tempo).
- **Original Deep Learning**: Custom 3-block 2D CNN with Batch Normalization, Max Pooling, Adaptive Average Pooling, and Dropout for rapid, CPU-compatible inference.
- **Uncertainty Calibration**: Automated prediction calibration marking outputs as uncertain when confidence < 40% or top-two margin < 15%.
- **Content-Based Song Similarity**: Nearest-neighbor retrieval over a 2,000-track FMA Small catalog (strictly isolated from training/validation splits) using cosine distance with guaranteed self-exclusion.
- **Interactive Web Dashboard**: Production-grade Next.js (App Router, Tailwind CSS) UI with interactive audio playback, waveform peak envelope, log-mel spectrogram heatmaps, 12-pitch chroma distribution, and genre ranking cards.
- **FastAPI REST Backend**: Validates audio signatures, decodes streams, enforces 30 MB / 5 min payload caps, automatically cleans temporary files, and serves real model predictions.

---

## 2. System Architecture

```
SonicScopeAI/
├── configs/
│   └── config.yaml              # Central configuration (paths, audio parameters, model grids)
├── data/
│   ├── downloads/               # Source archives (fma_small.zip, fma_metadata.zip)
│   ├── raw/                     # Extracted datasets (fma_small/, fma_metadata/)
│   ├── processed/               # Clean manifest, 51-d features.parquet, label_map.json
│   └── cache/                   # Spectrogram window cache (logmels/)
├── models/
│   ├── baseline/                # Random Forest pipeline (random_forest.joblib)
│   └── checkpoints/             # Trained GenreCNN checkpoint (genre_cnn.pt)
├── reports/
│   ├── figures/                 # EDA plots, spectrogram samples, CNN training curves
│   ├── metrics/                 # Official JSON evaluation reports (baseline_val, cnn_val, similarity)
│   └── screenshots/             # Interface validation screenshots
├── scripts/
│   ├── 01_build_manifest.py     # Clean manifest & FMA split assignment (Gate 1)
│   ├── 02_extract_features.py   # Handcrafted 51-d acoustic feature extraction (Gate 2)
│   ├── 03_cache_spectrograms.py # Precompute log-mel spectrogram arrays (Gate 3)
│   ├── 04_train_baseline.py     # Grid search & train Random Forest baseline (Gate 4)
│   ├── 05_train_cnn.py          # Train 2D CNN with early stopping & augmentations (Gate 5)
│   ├── predict.py               # Standalone single-track inference CLI (Gate 6)
│   ├── 07_build_similarity_index.py # Nearest-neighbor catalog similarity index (Gate 7)
│   └── 08_smoke_test.py         # End-to-end integration smoke test (Gate 8)
├── src/
│   ├── audio/                   # Decoding, window segmentation, log-mel transforms
│   ├── features/                # Tabular feature extraction (51 dimensions)
│   ├── models/                  # PyTorch GenreCNN architecture
│   └── config.py                # Configuration resolution and validation
├── backend/                     # FastAPI backend application
│   └── app/
│       ├── main.py              # Application entry point & CORS configuration
│       ├── schemas.py           # Typed Pydantic request/response contracts
│       ├── routes/analyze.py    # Audio upload & analysis pipeline
│       └── services/            # Audio, GenreCNN, and Similarity services
├── web/                         # Next.js frontend application (TypeScript, Tailwind)
└── tests/                       # Pytest test suite (51 unit and integration tests)
```

---

## 3. Environment & Prerequisites

- **Python**: 3.11+ (tested on Python 3.11 and Python 3.13)
- **Node.js**: Node 18+ LTS and npm
- **Audio Codec**: **FFmpeg** must be installed and accessible on system `PATH` (verify via `ffmpeg -version`)

---

## 4. Installation & Setup

### 4.1 Clone Repository & Setup Virtual Environment

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd sonicfusion2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
git clone <repo-url>
cd sonicfusion2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 Frontend Dependencies

```bash
cd web
npm install
cd ..
```

### 4.3 Environment Configuration

```bash
# Copy example environment configuration
cp .env.example .env
```

---

## 5. Dataset Acquisition & Preparation

Download the official Free Music Archive (FMA) Small dataset and metadata archives:

1. **FMA Small Audio (7.2 GiB)**: [https://os.unil.cloud.switch.ch/fma/fma_small.zip](https://os.unil.cloud.switch.ch/fma/fma_small.zip)
2. **FMA Metadata (342 MiB)**: [https://os.unil.cloud.switch.ch/fma/fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)

Extract the archives into `data/raw/`:
```
data/raw/
├── fma_small/       # Folders 000/ to 155/ containing .mp3 audio clips
└── fma_metadata/    # tracks.csv, genres.csv, features.csv
```

---

## 6. Step-by-Step Pipeline Execution (Gates 1 to 8)

Run the pipeline sequentially. All scripts support optional `--limit N` flags for rapid local pilot testing.

### Gate 1: Dataset Manifest & Split Validation
Builds verified dataset manifest with 8 balanced genres and official train/validation/test splits:
```powershell
python scripts/01_build_manifest.py
```
*Output*: `data/processed/manifest.csv`, `data/processed/label_map.json`, `reports/metrics/manifest_audit.json`.

### Gate 2: Tabular Feature Extraction (51 Acoustic Dimensions)
Extracts 51 standardized features (MFCCs, chroma, spectral, dynamics, tempo) with NaN/silent audio handling:
```powershell
python scripts/02_extract_features.py
```
*Output*: `data/processed/features.parquet`.

### Gate 3: Spectrogram Precomputation & Caching
Generates `(1, 128, 431)` log-mel spectrogram windows (3 non-overlapping 10s windows per track):
```powershell
python scripts/03_cache_spectrograms.py
```
*Output*: `data/cache/logmels/` cache directory and `data/processed/norm_stats.json`.

### Gate 4: Train Classical Random Forest Baseline
Performs grid search tuning across `n_estimators` and `max_depth` on training split only:
```powershell
python scripts/04_train_baseline.py
```
*Output*: `models/baseline/random_forest.joblib`, `reports/metrics/baseline_val.json`.

### Gate 5: Train 2D GenreCNN Deep Learning Model
Trains 3-block 2D CNN with AdamW (1e-3), batch size 16, time-shift/gain augmentations, and early stopping on validation macro-F1:
```powershell
python scripts/05_train_cnn.py
```
*Output*: `models/checkpoints/genre_cnn.pt`, `reports/metrics/cnn_val.json`, `reports/figures/cnn_training_curves.png`.

### Gate 6: Single-Track CLI Inference Verification
Classifies any arbitrary MP3 file using the trained GenreCNN checkpoint:
```powershell
python scripts/predict.py web/public/samples/sample_000002.mp3
```

### Gate 7: Build Song Similarity Index
Constructs nearest-neighbor index over 2,000 catalog tracks from training/validation splits only (never test), fits catalog scaler, and evaluates same-genre precision@5 proxy metric:
```powershell
python scripts/07_build_similarity_index.py --catalog-size 2000
```
*Output*: `data/processed/similarity_index.joblib`, `reports/metrics/similarity_metrics.json`.

### Gate 8: End-to-End System Smoke Test
Executes complete automated validation across config, manifest, features, models, similarity retrieval, and live API endpoints:
```powershell
python scripts/08_smoke_test.py
```

---

## 7. Running the Full-Stack Application

### 7.1 Start the FastAPI Backend

```powershell
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API is accessible at:
- **API Root**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 7.2 Start the Next.js Web Frontend

In a separate terminal window:
```powershell
cd web
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser:
- **Upload & Analysis Page**: [http://localhost:3000](http://localhost:3000)
- **Model Card & Architecture**: [http://localhost:3000/about](http://localhost:3000/about)
- **Evaluation & Benchmark Comparison**: [http://localhost:3000/evaluation](http://localhost:3000/evaluation)

---

## 8. Running Automated Tests

Run the full Pytest test suite (51 tests):
```powershell
# Run all tests
pytest -v

# Run targeted subsystem test suites
pytest tests/test_similarity.py -v
pytest tests/test_cnn.py -v
pytest tests/test_backend.py -v
pytest tests/test_baseline.py -v
pytest tests/test_audio_features.py -v
pytest tests/test_config.py -v
pytest tests/test_manifest.py -v
```

---

## 9. Research & Evaluation Summary

| Model | Input Representation | Validation Macro-F1 | Validation Accuracy | Parameters / Complexity |
|---|---|---|---|---|
| **Random Forest Baseline** | 51-d Handcrafted Acoustic Features | **0.5093** (50.93%) | **0.5238** (52.38%) | 400 Trees (`max_depth=None`) |
| **GenreCNN (2D CNN)** | 128 Mel-Band Log-Mel Spectrogram | Pilot Run: 0.1964 | Pilot Run: 0.2500 | 23,928 Parameters (CPU Safe) |
| **Similarity Engine** | 51-d Standardized Scaler (Catalog-Fitted) | **0.3893** (Precision@5) | Top-5 Cosine | 2,000 Catalog Tracks |

> **Note**: Full-dataset training metrics are recorded directly in [`reports/metrics/`](reports/metrics/). SonicScope AI strictly adheres to zero metric fabrication: all scores displayed on the dashboard are read directly from disk JSON artifacts.

---

## 10. Academic Disclosures & Errata

1. **Split Integrity**: Official FMA Small track-level assignments (`train`, `val`, `test`) are strictly honored. No custom random re-splitting was performed.
2. **Artist Leakage Limitation**: Documented in FMA Small (Defferrard et al., 2017), track-level assignment allows tracks from the same artist to appear across both training and validation splits.
3. **CPU-Safe Mode**: Pretrained Essentia Discogs-EffNet models are decoupled to avoid TensorFlow/Essentia binary conflicts on Windows CPU environments; GenreCNN serves verified 8-genre recognition natively.

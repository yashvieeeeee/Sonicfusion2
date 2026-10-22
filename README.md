# SonicScope AI (SonicFusion 2)

> **Music Intelligence & Genre Recognition System:** Comparative study evaluating classical feature-engineered machine learning (Random Forest) against spectrogram-based deep learning (2D CNN) on the Free Music Archive (FMA Small).

The single source of truth for architectural decisions, contracts, and requirements is [`docs/PLAN.md`](docs/PLAN.md).

---

## 1. Project Overview

SonicScope AI is an end-to-end music information retrieval (MIR) platform and research testbed:
- **Audio Processing**: Standardized 22,050 Hz mono decoding, 10-second non-overlapping windows, and 128 mel-band log-mel spectrogram representations `(1, 128, 431)`.
- **Classical Baseline**: Tuned Random Forest operating on 51-dimensional handcrafted acoustic features (MFCCs, chroma, spectral centroid/rolloff/bandwidth/flatness, RMS energy, and tempo).
- **Original Deep Learning**: Custom 3-block 2D CNN with Batch Normalization, Max Pooling, Adaptive Average Pooling, and Dropout for efficient, CPU-compatible inference.
- **Content-Based Similarity**: Nearest-neighbor audio retrieval over a 2,000-track catalog (strictly isolated from the test split) using cosine distance with guaranteed self-exclusion.
- **Production Web Dashboard**: Next.js (App Router, Tailwind CSS) UI featuring interactive waveform visualization, log-mel spectrogram heatmaps, 12-pitch chroma distribution, and genre ranking cards.
- **FastAPI REST Backend**: Validates audio signatures, decodes streams, enforces 30 MB / 5 min payload caps, automatically cleans temporary files, and serves real model predictions.

---

## 2. Directory Layout

```
SonicScopeAI/
├── configs/
│   └── config.yaml              # Central configuration (paths, audio parameters, model grids)
├── data/                        # [NOT COMMITTED] Datasets and processing caches
│   ├── downloads/               # Source archives (fma_small.zip, fma_metadata.zip)
│   ├── raw/                     # Extracted datasets (fma_small/, fma_metadata/)
│   ├── processed/               # Clean manifest, 51-d features.parquet, label_map.json
│   └── cache/                   # Precomputed spectrogram window cache (logmels/)
├── models/                      # [NOT COMMITTED] Trained weights and pipeline joblibs
│   ├── baseline/                # Random Forest pipeline (random_forest.joblib)
│   └── checkpoints/             # Trained GenreCNN checkpoint (genre_cnn.pt)
├── reports/
│   ├── figures/                 # EDA plots, spectrogram samples, confusion matrices
│   ├── metrics/                 # Official JSON evaluation reports (test_metrics, cnn_val, eda_summary)
│   └── screenshots/             # Interface validation screenshots
├── scripts/
│   ├── 01_build_manifest.py     # Clean manifest & FMA split assignment
│   ├── 02_extract_features.py   # Handcrafted 51-d acoustic feature extraction
│   ├── 03_cache_spectrograms.py # Precompute log-mel spectrogram arrays
│   ├── 04_train_baseline.py     # Grid search & train Random Forest baseline
│   ├── 05_train_cnn.py          # Train 2D CNN with early stopping & augmentations
│   ├── 06_evaluate.py           # Single-run test split evaluation on frozen models
│   ├── 07_build_similarity_index.py # Nearest-neighbor catalog similarity index
│   ├── 08_smoke_test.py         # End-to-end integration and API smoke test
│   ├── 09_eda.py                # Exploratory Data Analysis figures and summary
│   └── predict.py               # Standalone single-track inference CLI
├── src/
│   ├── audio/                   # Decoding, window segmentation, log-mel transforms
│   ├── features/                # Tabular feature extraction (51 dimensions)
│   ├── models/                  # PyTorch GenreCNN architecture
│   └── config.py                # Configuration resolution and validation
├── backend/                     # FastAPI backend application
│   └── app/
│       ├── main.py              # Application entry point, CORS, and figures serving
│       ├── schemas.py           # Typed Pydantic request/response contracts
│       ├── routes/analyze.py    # Audio upload & analysis pipeline
│       └── services/            # Audio, GenreCNN, and Similarity services
├── web/                         # Next.js frontend application (TypeScript, Tailwind)
│   ├── public/
│   │   └── samples/             # Demo MP3 clips for quick browser testing
│   └── src/
│       ├── app/                 # Next.js App Router (upload, /about, /evaluation)
│       ├── components/          # Visualization components (Waveform, Spectrogram, etc.)
│       └── lib/                 # API client and formatters
└── tests/                       # Pytest test suite (backend, models, features, audio)
```

> **Important**: The `data/` and `models/` directories are intentionally untracked and excluded from git commits (via `.gitignore`) to keep the repository lightweight and adhere to reproducible data practices.

---

## 3. Prerequisites

Verified environment versions installed and tested on this system:
- **Python**: `Python 3.13.7` installed (verified with `python --version`).
- **Node.js**: `v22.18.0` installed (verified with `node --version`).
- **Next.js Node Requirement**: The installed Next.js framework version is `16.4.2` (from `web/package.json`), which requires Node.js `>=20.9.0` (verified from `"engines": { "node": ">=20.9.0" }` in `web/node_modules/next/package.json`).
- **Audio Codec (FFmpeg)**: FFmpeg must be installed and accessible on your system `PATH`.
  - Windows: Install via `winget install Gyan.FFmpeg` or `choco install ffmpeg`, then confirm with `ffmpeg -version`.
  - macOS: Install via `brew install ffmpeg`, then confirm with `ffmpeg -version`.
  - Linux (Ubuntu/Debian): Install via `sudo apt-get install -y ffmpeg`, then confirm with `ffmpeg -version`.

---

## 4. Setup Instructions

### 4.1 Python Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 Web Frontend Setup

```bash
cd web
npm install
cd ..
```

---

## 5. Dataset Acquisition & Verification

Download the official Free Music Archive (FMA Small) dataset and metadata:

1. **FMA Small Audio (7.2 GiB)**: `https://os.unil.cloud.switch.ch/fma/fma_small.zip`
2. **FMA Metadata (342 MiB)**: `https://os.unil.cloud.switch.ch/fma/fma_metadata.zip`

### Verify SHA-1 Checksums

Before extracting, verify the archive integrity against the official FMA digests:

**Windows (PowerShell):**
```powershell
(Get-FileHash .\data\downloads\fma_small.zip -Algorithm SHA1).Hash
# Expected: ade154f733639d52e35e32f5593efe5be76c6d70

(Get-FileHash .\data\downloads\fma_metadata.zip -Algorithm SHA1).Hash
# Expected: f0df49ffe5f2a6008d7dc83c6915b31835dfe733
```

**macOS / Linux:**
```bash
sha1sum data/downloads/fma_small.zip data/downloads/fma_metadata.zip
# Expected: ade154f733639d52e35e32f5593efe5be76c6d70  data/downloads/fma_small.zip
# Expected: f0df49ffe5f2a6008d7dc83c6915b31835dfe733  data/downloads/fma_metadata.zip
```

### Extract Archives

Extract the archives into `data/raw/`:
```bash
# Windows PowerShell
Expand-Archive data/downloads/fma_small.zip data/raw -Force
Expand-Archive data/downloads/fma_metadata.zip data/raw -Force

# macOS / Linux
unzip -q data/downloads/fma_small.zip -d data/raw
unzip -q data/downloads/fma_metadata.zip -d data/raw
```

Verify that `data/raw/fma_small/` contains numbered directories (`000/` through `155/`) and `data/raw/fma_metadata/tracks.csv` is present.

---

## 6. Pipeline Execution Order (Scripts 01 to 09)

Execute the pipeline scripts in sequential order:

```bash
# Step 1: Build clean dataset manifest with official splits and exclusions
python scripts/01_build_manifest.py

# Step 2: Extract 51-dimensional acoustic features to parquet
python scripts/02_extract_features.py

# Step 3: Precompute and cache 10-second log-mel spectrogram windows
python scripts/03_cache_spectrograms.py

# Step 4: Train classical Random Forest baseline with hyperparameter grid search
python scripts/04_train_baseline.py

# Step 5: Train 2D GenreCNN deep learning model
python scripts/05_train_cnn.py

# Step 6: Evaluate frozen models on official test split (Runs once; requires clean git status)
python scripts/06_evaluate.py

# Step 7: Build nearest-neighbor song similarity catalog index
python scripts/07_build_similarity_index.py --catalog-size 2000

# Step 8: End-to-end integration and API smoke test
python scripts/08_smoke_test.py

# Step 9: Exploratory Data Analysis (EDA figures and summary metrics)
python scripts/09_eda.py
```

### What is Optional vs Required
- **Optional `--limit N` flag**: Scripts `01`, `02`, `03`, `04`, `05`, and `09` support `--limit <N>` to run quick pilots on a subset of tracks.
- **Optional Spectrogram Caching (03)**: Training the CNN without caching will compute spectrograms on-the-fly, though caching significantly accelerates multi-epoch training.
- **Frozen Test Evaluation (06)**: Safeguarded by design. It refuses to run if uncommitted changes exist (preventing unrecorded experiment state) and refuses to overwrite an existing `reports/metrics/test_metrics.json` unless `--force` is supplied.
- **Smoke Test (08)**: Can be run any time after training to verify system readiness.

---

## 7. Smoke Test Utility (`scripts/08_smoke_test.py`)

`scripts/08_smoke_test.py` validates all core system dependencies with clear, actionable diagnostics:
1. **FFmpeg**: Checks that the `ffmpeg` binary is available on your system `PATH`.
2. **Config**: Confirms `configs/config.yaml` loads correctly and matches defined schemas.
3. **Manifest**: Verifies `data/processed/manifest.csv` exists and contains valid splits.
4. **Model Checkpoints**: Checks for `models/baseline/random_forest.joblib` and `models/checkpoints/genre_cnn.pt`.
5. **API Health**: Pings `GET http://localhost:8000/api/health` to confirm the backend is running.
6. **Live Inference**: Dispatches a real test MP3 to `POST /api/analyze` to verify waveform extraction, spectrogram rendering, acoustic features, and genre predictions.

Run the smoke test with:
```bash
python scripts/08_smoke_test.py
```
If any check fails, the script outputs a clear explanation and remediation steps.

---

## 8. Running Backend & Frontend

### 8.1 Start the FastAPI Backend

From the repository root:
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Root: [http://localhost:8000](http://localhost:8000)
- OpenAPI / Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 8.2 Start the Next.js Frontend

In a separate terminal window:
```bash
cd web
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser:
- **Audio Analyzer**: [http://localhost:3000/](http://localhost:3000/) - Upload audio files or choose preloaded samples.
- **Evaluation & Benchmarks**: [http://localhost:3000/evaluation](http://localhost:3000/evaluation) - Test split metrics, confusion matrices, comparison tables, and training curves.
- **Model Card & Architecture**: [http://localhost:3000/about](http://localhost:3000/about) - Dataset provenance, preprocessing parameters, architecture details, and academic limitations.

---

## 9. Adding Demo Audio Clips (`web/public/samples`)

The frontend audio analyzer includes a quick-selection menu for sample audio clips located in `web/public/samples/`.

To add custom demo clips:
1. Copy any 15 to 30-second `.mp3` clip into `web/public/samples/`:
   ```bash
   cp my_track.mp3 web/public/samples/my_track.mp3
   ```
2. The preloaded samples are referenced in `web/src/app/page.tsx` within the `SAMPLE_FILES` array:
   ```typescript
   {
     id: "sample_000002",
     name: "Sample 000002 (Electronic)",
     path: "/samples/sample_000002.mp3",
   }
   ```
3. Add an entry to `SAMPLE_FILES` to display it in the interactive demo selector.

---

## 10. Running Automated Tests

Run the full Python test suite with pytest:
```bash
pytest -v
```

Run targeted subsystem tests:
```bash
pytest tests/test_backend.py -v       # Backend routes, caps, file cleanup, figures endpoint
pytest tests/test_cnn.py -v           # CNN model architecture and inference
pytest tests/test_baseline.py -v      # Random Forest pipeline and training
pytest tests/test_similarity.py -v    # Catalog similarity index & retrieval
pytest tests/test_audio_features.py -v # Audio decoding and 51-d feature extraction
pytest tests/test_manifest.py -v      # Manifest creation and split validation
```

Verify frontend build:
```bash
cd web
npm run build
cd ..
```

---

## 11. Academic Limitations & Model Card Notes

1. **Noisy and Overlapping Genre Labels**: FMA Small genres are subjective and multi-label in reality; forcing single top-level assignment introduces label noise (especially between Rock, Pop, and Folk).
2. **Western-Centric 30 s Clips**: Training data is drawn exclusively from Western indie tracks distributed on the Free Music Archive.
3. **Truncated Upload Scoring**: Uploaded tracks longer than 30 seconds are scored on their first 30 seconds only.
4. **Short Clip Reliability**: Audio clips shorter than 10 seconds are zero-padded and may produce less reliable predictions.
5. **Uncalibrated Softmax Scores**: Prediction scores reflect softmax outputs rather than true calibrated Bayesian probabilities.
6. **Artist Overlap Disclosures**: While the test split maintains zero `artist_id` overlap with the training set, compilation albums or uncredited shared artists were not exhaustively checked.
7. **Non-Commercial Licensing**: Audio clips are distributed under Creative Commons licenses; model weights trained on FMA should not be used in commercial contexts without rights clearance.

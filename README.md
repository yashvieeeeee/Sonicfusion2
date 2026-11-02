# SonicFusion 2

SonicFusion 2 is a music information retrieval system and comparative evaluation testbed for eight-way music genre classification on the Free Music Archive (FMA Small) dataset. The project compares classical machine learning models trained on 51-dimensional handcrafted acoustic summary statistics (Random Forest, AdaBoost, RBF Support Vector Machine, and ExtraTrees) against a deep convolutional neural network (GenreCNN) trained on log-mel spectrogram representations, as well as a multimodal soft-voting ensemble that integrates predictions across tabular and spectral feature spaces. The codebase includes a FastAPI backend for inference and audio processing, a Next.js interface for waveform and spectrogram inspection, and evaluation scripts designed to enforce strict separation between training, validation, and test partitions.

---

## 1. Dataset Specification

SonicFusion 2 uses the Free Music Archive (FMA Small) dataset (`configs/config.yaml#L16-L17`):
- **Raw Track Count**: 8,000 thirty-second audio tracks across 8 balanced genres (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock), with 1,000 tracks per class in raw metadata (`reports/metrics/manifest_audit.json#L4`).
- **Official Splits**: The train, validation, and test split assignments follow the official FMA metadata (`fma_metadata/tracks.csv`) under `('set', 'split')` where `('set', 'subset') == 'small'`. The partitions are preserved without re-splitting or random shuffling (`scripts/01_build_manifest.py`).
- **Data Filtering & Exclusions**: Six (6) corrupted audio tracks failed the minimum decodable duration threshold of 25.0 seconds (`configs/config.yaml#L10`) and were excluded: tracks 98565, 98567, and 98569 (Hip-Hop, truncated audio < 2.0 s), and tracks 99134 (Electronic), 108925 (Rock), and 133297 (Experimental) due to corrupt files with 0.0 s decodable audio (`reports/metrics/data_exclusions.csv`).
- **Final Dataset Count**: 7,994 valid audio tracks (`reports/metrics/manifest_audit.json#L5`), distributed as:
  - **Training Split**: 6,394 tracks (Electronic: 799, Experimental: 799, Folk: 800, Hip-Hop: 797, Instrumental: 800, International: 800, Pop: 800, Rock: 799).
  - **Validation Split**: 800 tracks (100 per genre, balanced).
  - **Test Split**: 800 tracks (100 per genre, balanced).

---

## 2. Methodology & Feature Representations

### 2.1 Tabular Acoustic Features (51 Dimensions)
Extracted in `src/features/extraction.py` at 22,050 Hz:
- **Timbral (26 features)**: Mean and standard deviation of MFCCs 1–13.
- **Tonal (12 features)**: Mean of 12-bin chroma STFT.
- **Spectral (8 features)**: Mean and standard deviation of spectral centroid, spectral rolloff, spectral bandwidth, and zero-crossing rate.
- **Dynamics (2 features)**: Mean and standard deviation of root-mean-square (RMS) energy.
- **Rhythm (3 features)**: Tempo estimate, onset strength mean, and onset strength standard deviation.

### 2.2 Spectrogram Deep Learning (GenreCNN)
Defined in `src/models/genre_cnn.py` and `configs/config.yaml#L35-L46`:
- Input consists of three 10.0-second non-overlapping log-mel spectrogram windows per 30.0-second track, shaped `(1, 128, 431)` with 128 mel frequency bins and an FFT hop length of 512.
- Normalization parameters (global mean -43.32 dB, standard deviation 15.19 dB) are computed strictly across training windows (`data/processed/norm_stats.json#L2-L3`).
- The network contains 3 convolutional blocks with Batch Normalization, Max Pooling, Dropout (0.3), and Adaptive Average Pooling.
- Window-level softmax predictions are averaged at inference to yield track-level class probabilities.

### 2.3 Classifiers & Soft-Voting Ensemble
- **Random Forest**: Tuned on validation data (400 trees, unconstrained depth; `reports/metrics/baseline_val.json#L5-L8`).
- **AdaBoost**: Evaluated on validation data (100 estimators, max depth 2; `reports/metrics/baseline_val.json#L230-L233`).
- **RBF Support Vector Machine**: Tuned on validation data ($C=10, \gamma=0.01$; `reports/metrics/extra_models_val.json#L30-L31`).
- **ExtraTrees**: Tuned on validation data (400 trees, `max_features='sqrt'`; `reports/metrics/extra_models_val.json#L150-L152`).
- **Soft-Voting Ensemble**: Convex combination of class probability distributions with weights chosen on the validation split: $w_{\text{SVC}} = 0.40$, $w_{\text{CNN}} = 0.60$, $w_{\text{RF}} = 0.00$ (`reports/metrics/final_test.json#L294-L297`).

---

## 3. Benchmark Results

In the table below, **Held-Out Test metrics are presented first**, followed by the Validation metrics used during development. All numbers reflect evaluations on the 800 test tracks and 800 validation tracks, with a fixed random seed of 42.

| Model Architecture | Test Accuracy | Test Macro-F1 | Test Top-3 Accuracy | Val Accuracy | Val Macro-F1 | Val Top-3 Accuracy | Source Files |
|---|---|---|---|---|---|---|---|
| **Soft-Voting Ensemble**<br>*(0.40 SVC + 0.60 CNN + 0.00 RF)* | **52.38%**<br>*(0.52375)* | **0.5173**<br>*(0.517336)* | **82.62%**<br>*(0.82625)* | **62.88%** *(val)*<br>*(0.62875)* | **0.6198**<br>*(0.619760)* | **85.75%**<br>*(0.85750)* | `final_test.json`<br>`extra_models_val.json` |
| **GenreCNN**<br>*(Best Checkpoint, Epoch 24)* | **50.62%**<br>*(0.50625)* | **0.4970**<br>*(0.496958)* | **81.38%**<br>*(0.81375)* | **58.50%**<br>*(0.58500)* | **0.5724**<br>*(0.572436)* | **85.25%**<br>*(0.85250)* | `final_test.json`<br>`extra_models_val.json` |
| **Random Forest Baseline**<br>*(400 trees, depth=None)* | **43.75%**<br>*(0.43750)* | **0.4354**<br>*(0.435409)* | **75.75%**<br>*(0.75750)* | **52.25%**<br>*(0.52250)* | **0.5073**<br>*(0.507309)* | **82.25%**<br>*(0.82250)* | `final_test.json`<br>`extra_models_val.json` |
| **RBF Support Vector Machine**<br>*(C=10, gamma=0.01)* | **43.62%**<br>*(0.43625)* | **0.4380**<br>*(0.438007)* | **76.88%**<br>*(0.76875)* | **53.12%**<br>*(0.53125)* | **0.5238**<br>*(0.523798)* | **80.38%**<br>*(0.80375)* | `final_test.json`<br>`extra_models_val.json` |
| **ExtraTrees Classifier**<br>*(400 trees, max_features='sqrt')* | **43.62%**<br>*(0.43625)* | **0.4330**<br>*(0.433029)* | **76.88%**<br>*(0.76875)* | **52.12%**<br>*(0.52125)* | **0.5063**<br>*(0.506323)* | **82.25%**<br>*(0.82250)* | `final_test.json`<br>`extra_models_val.json` |
| **AdaBoost Classifier**<br>*(100 estimators, max_depth=2)* | — | — | — | **44.25%**<br>*(0.44250)* | **0.4398**<br>*(0.439822)* | — | `baseline_val.json` |
| **Uniform Random Chance Baseline** | **12.50%** | **0.1250** | **37.50%** | **12.50%** | **0.1250** | **37.50%** | Theoretical floor ($1/8$) |

*Note: The 62.88% figure is explicitly a validation score optimized over 66 ensemble weight combinations on the validation split. When evaluated on the held-out test split, the ensemble achieves 52.38% accuracy. For historical runs and reconciliation notes, see `docs/EVALUATION.md#L168-L179`.*

---

## 4. Evaluation Protocol & Data Leakage Audit

### 4.1 Evaluation Protocol
- **Tuning and Model Selection**: All hyperparameter grid searches, checkpoint early stopping, and ensemble weight selections were performed strictly using the 800-track validation split (`reports/metrics/extra_models_val.json#L4`).
- **Test Set Evaluation**: All tuning used the validation split; reported test numbers come from `scripts/11_evaluate_final_test.py` (an earlier test run exists in `test_metrics.json`). No parameters or ensemble weights were altered following test evaluation.

### 4.2 Data Leakage Audit Summary
- **Artist Separation**: The dataset contains 2,307 unique artists (`reports/metrics/manifest_audit.json#L73`). The audit confirmed **0 overlapping artists between any split pair** (0 in train-val, 0 in train-test, 0 in val-test; `manifest_audit.json#L80`).
- **Split Assignments**: Exactly 0 split discrepancies exist between the processed manifest and the raw FMA metadata (`manifest_audit.json#L12-L16`).
- **Feature Transformers**: Scikit-learn imputers (`SimpleImputer`) and scalers (`StandardScaler`) are fit strictly on training features (`X_train`) within pipeline wrappers (`scripts/04_train_baseline.py#L125-L135`, `scripts/10_train_extra_models.py#L52-L65`).
- **Spectrogram Normalization**: Log-mel spectrogram mean (-43.32 dB) and standard deviation (15.19 dB) in `data/processed/norm_stats.json` were computed exclusively over training windows (`src/audio/spectrogram.py`).

---

## 5. Repository Structure

```
SonicFusion2/
├── configs/
│   └── config.yaml              # Central pipeline configuration (paths, audio, model parameters)
├── data/                        # [Local / Untracked] Audio archives and precomputed arrays
│   ├── raw/                     # Extracted fma_small/ and fma_metadata/
│   ├── processed/               # manifest.csv, features.parquet, label_map.json, norm_stats.json
│   └── cache/                   # Precomputed log-mel spectrogram arrays (logmels/)
├── docs/
│   ├── EVALUATION.md            # Comprehensive academic evaluation report and audit findings
│   ├── VIVA_QA.md               # 20 technical questions and answers for examination defense
│   ├── PLAN.md                  # Project blueprint and architectural decisions
│   └── REFACTOR_AUDIT.md        # Codebase refactoring and verification audit
├── models/                      # [Local / Untracked] Serialized models and checkpoints
│   ├── baseline/                # random_forest.joblib
│   ├── checkpoints/             # genre_cnn.pt
│   └── extra/                   # rbf_svc.joblib, extra_trees.joblib
├── reports/
│   ├── figures/                 # EDA plots, spectrogram samples, confusion matrices
│   └── metrics/                 # Benchmark JSON reports (final_test.json, extra_models_val.json, etc.)
├── scripts/
│   ├── 01_build_manifest.py     # Ingest metadata, validate audio duration, record exclusions
│   ├── 02_build_features.py     # Extract 51-d features and compute training norm_stats
│   ├── 03_cache_spectrograms.py # Precompute 10s log-mel spectrogram windows
│   ├── 04_train_baseline.py     # Grid search and train Random Forest & AdaBoost baselines
│   ├── 05_extract_embeddings.py # Pretrained MERT-v1-95M audio embedding extraction pipeline
│   ├── 05_train_cnn.py          # Train GenreCNN with early stopping on validation split
│   ├── 06_evaluate.py           # Earlier evaluation script (recorded in test_metrics.json)
│   ├── 07_build_similarity_index.py # Build catalog nearest-neighbor similarity index
│   ├── 08_smoke_test.py         # End-to-end integration smoke test
│   ├── 09_eda.py                # Exploratory data analysis figures and summary
│   ├── 10_train_extra_models.py # Grid search RBF SVC, ExtraTrees, and soft-voting ensemble
│   ├── 11_evaluate_final_test.py# Standardized test evaluation across all models
│   └── predict.py               # Single-track inference CLI
├── src/
│   ├── audio/                   # Audio decoding, windowing, and log-mel transformations
│   ├── features/                # Tabular feature extraction (51-d) and MERT embeddings
│   ├── models/                  # GenreCNN PyTorch architecture and extra models module
│   └── config.py                # Configuration loader and path resolver
├── backend/                     # FastAPI backend application
│   └── app/
│       ├── main.py              # Application entry point and CORS routing
│       ├── routes/analyze.py    # Audio upload, analysis, and prediction endpoint
│       └── services/            # Audio decoding, GenreCNN inference, and similarity retrieval
├── web/                         # Next.js frontend application (TypeScript, Tailwind CSS)
└── tests/                       # Pytest automated test suite (98 passing tests)
```

---

## 6. How to Reproduce

### 6.1 Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6.2 Data Preparation & Preprocessing
```powershell
# 1. Build manifest and exclude corrupted tracks
python scripts/01_build_manifest.py

# 2. Extract 51-dimensional tabular features and training normalization stats
python scripts/02_build_features.py

# 3. Cache 10-second log-mel spectrogram arrays
python scripts/03_cache_spectrograms.py
```

### 6.3 Model Training & Validation Tuning
```powershell
# 4. Train Random Forest baseline (and optional AdaBoost comparison)
python scripts/04_train_baseline.py --include-adaboost

# 5. Train 2D GenreCNN with early stopping on validation
python scripts/05_train_cnn.py

# 6. Tune RBF SVC, ExtraTrees, and soft-voting ensemble on validation split
python scripts/10_train_extra_models.py
```

### 6.4 Test Evaluation
```powershell
# 7. Evaluate all frozen models and ensemble on the held-out test split
python scripts/11_evaluate_final_test.py
```

### 6.5 Verification & Automated Tests
```powershell
# Run the complete automated test suite
pytest tests/
```

### 6.6 Running the Application Services
```powershell
# Terminal 1: Launch FastAPI backend
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch Next.js web interface
cd web
npm install
npm run dev
```

---

## 7. Limitations

1. **Fixed 30-Second Clip Duration**: Analysis is limited to 30.0-second excerpts (`configs/config.yaml#L37`), discarding structural shifts, modulations, and transitions occurring elsewhere in full tracks.
2. **Training Set Scale**: With 6,394 training tracks, the deep convolutional model is constrained to 3 blocks to prevent overfitting on modest data volumes [needs author citation].
3. **Information Bottleneck in Tabular Features**: Compressing 30-second audio files into 51 scalar statistics discards sequential, harmonic, and rhythmic progressions, consistent with the observed empirical accuracy ceiling of ~43.6%–43.8% across Random Forest, AdaBoost, SVC, and ExtraTrees.
4. **Subjective Ground-Truth Labels**: FMA genres are crowd-annotated and treat genres as mutually exclusive classes, penalizing models on hybrid or ambiguous tracks (e.g., Experimental vs. Electronic, Folk vs. Rock) [needs author citation].
5. **Test Partition Difficulty**: The held-out test set exhibits marked difficulty on Folk (test recall 32.0% vs. validation recall 65.0% for GenreCNN) and Experimental (test recall 30.0% vs. validation recall 45.0% for GenreCNN) due to distribution shifts across previously unseen artists.

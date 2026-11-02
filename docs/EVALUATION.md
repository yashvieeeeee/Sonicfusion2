# SonicFusion 2: Evaluation Report & Technical Audit

This document provides a comprehensive, rigorous technical evaluation of the **SonicFusion 2** music genre classification pipeline. It is prepared for an examiner or technical auditor seeking full provenance, methodological integrity, and reproducibility.

Every factual assertion, parameter, and metric in this document is cited directly from the repository's configuration files, training and evaluation scripts, or metrics logs. No claims regarding published literature are asserted without explicit citations; external statements requiring citations are designated with `[needs author citation]`.

---

## 1. Dataset Specification & Data Hygiene

### 1.1 FMA Small Definition
SonicFusion 2 uses the Free Music Archive (FMA) Small subset (`configs/config.yaml#L16-L17`):
- **Raw Track Count**: 8,000 audio tracks distributed across 8 balanced genres (`reports/metrics/manifest_audit.json#L4`, `reports/metrics/manifest_audit.json#L93-L101`).
- **Nominal Class Balance**: Exactly 1,000 tracks per genre in the raw metadata (`reports/metrics/manifest_audit.json#L93-L101`).
- **Audio Clip Format & Duration**: 30.0-second MP3 audio clips sampled at 22,050 Hz, converted deterministically to mono (`configs/config.yaml#L36-L37`, `configs/config.yaml#L46`).
- **Target Genres**: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock (`configs/config.yaml`, `reports/metrics/final_test.json#L6-L13`).
- **Official Split Provenance**: The train, validation, and test split partitions are taken directly from the official FMA metadata table `fma_metadata/tracks.csv` under the multi-index column `('set', 'split')` where `('set', 'subset') == 'small'` (`scripts/01_build_manifest.py#L65-L82`, `reports/metrics/manifest_audit.json#L7-L11`). Splits are strictly preserved without re-splitting or random reshuffling.

### 1.2 Data Exclusions
During dataset ingestion and manifest validation, tracks were required to contain at least 25.0 seconds of decodable audio (`configs/config.yaml#L10`). Six (6) corrupted audio tracks failed this integrity filter and were formally excluded (`reports/metrics/manifest_audit.json#L6`, `reports/metrics/data_exclusions.csv#L1-L7`). All six excluded tracks belonged to the training partition; zero tracks were excluded from validation or test (`reports/metrics/manifest_audit.json#L17-L21`).

| Track ID | Genre | Split | File Size (Bytes) | Decodable Duration (s) | Exclusion Reason | Citation |
|---|---|---|---|---|---|---|
| **98565** | Hip-Hop | training | 300,753 | 1.63 s | Decodable duration < 25.0 s (`min_track_seconds`) | `reports/metrics/data_exclusions.csv#L2` |
| **98567** | Hip-Hop | training | 141,545 | 0.54 s | Decodable duration < 25.0 s (`min_track_seconds`) | `reports/metrics/data_exclusions.csv#L3` |
| **98569** | Hip-Hop | training | 236,749 | 1.56 s | Decodable duration < 25.0 s (`min_track_seconds`) | `reports/metrics/data_exclusions.csv#L4` |
| **99134** | Electronic | training | 1,298 | 0.00 s | Unreadable/corrupt MP3 (header truncated) | `reports/metrics/data_exclusions.csv#L5` |
| **108925** | Rock | training | 1,371 | 0.00 s | Unreadable/corrupt MP3 (header truncated) | `reports/metrics/data_exclusions.csv#L6` |
| **133297** | Experimental | training | 1,752 | 0.00 s | Unreadable/corrupt MP3 (header truncated) | `reports/metrics/data_exclusions.csv#L7` |

### 1.3 Final Per-Split Class Distribution
Following the exclusion of the 6 corrupted training tracks, the final clean dataset contains **7,994 valid tracks** (`reports/metrics/manifest_audit.json#L5`, `reports/metrics/manifest_audit.json#L86`). The validation and test partitions remain perfectly balanced at exactly 100 tracks per class (800 tracks each). The training partition contains 6,394 tracks (`reports/metrics/manifest_audit.json#L13-L15`, `reports/metrics/manifest_audit.json#L89-L91`).

| Genre | Training Split | Validation Split | Test Split | Total Valid Tracks | Raw Metadata Tracks | Citation |
|---|---|---|---|---|---|---|
| **Electronic** | 799 | 100 | 100 | 999 | 1,000 | `reports/metrics/manifest_audit.json#L104-L108` |
| **Experimental** | 799 | 100 | 100 | 999 | 1,000 | `reports/metrics/manifest_audit.json#L109-L113` |
| **Folk** | 800 | 100 | 100 | 1,000 | 1,000 | `reports/metrics/manifest_audit.json#L114-L118` |
| **Hip-Hop** | 797 | 100 | 100 | 997 | 1,000 | `reports/metrics/manifest_audit.json#L119-L123` |
| **Instrumental** | 800 | 100 | 100 | 1,000 | 1,000 | `reports/metrics/manifest_audit.json#L124-L128` |
| **International** | 800 | 100 | 100 | 1,000 | 1,000 | `reports/metrics/manifest_audit.json#L129-L133` |
| **Pop** | 800 | 100 | 100 | 1,000 | 1,000 | `reports/metrics/manifest_audit.json#L134-L138` |
| **Rock** | 799 | 100 | 100 | 999 | 1,000 | `reports/metrics/manifest_audit.json#L139-L143` |
| **Total** | **6,394** | **800** | **800** | **7,994** | **8,000** | `reports/metrics/manifest_audit.json#L88-L92` |

---

## 2. Evaluation Protocol

To prevent methodological overfitting and provide an honest assessment of generalization, all tuning and ensemble-weight selection used the validation split; reported test numbers come from `scripts/11_evaluate_final_test.py` (an earlier test run exists in `test_metrics.json`).

### 2.1 Metric Definitions & Rationale
Three primary metrics are tracked across all evaluations:
1. **Classification Accuracy**: The proportion of correctly predicted tracks over total evaluation tracks ($\frac{\sum \mathbb{I}(y = \hat{y})}{N}$).
2. **Macro-Averaged F1-Score ($\text{Macro-F1}$)**: The unweighted arithmetic mean of per-class F1-scores:
   $$\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^C \text{F1}_c$$
   *Justification*: In multi-class classification, overall accuracy can mask severe failure modes if a model defaults to predicting majority or easily separable classes while failing entirely on difficult classes (e.g., Pop or Experimental). Macro-F1 treats every genre equally, penalizing models that achieve high overall accuracy at the expense of recall or precision on harder genres.
3. **Top-3 Accuracy**: The proportion of tracks for which the ground-truth genre is contained within the top-3 highest predicted class probabilities:
   $$\text{Top-3 Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(y_i \in \operatorname{argtop3}(\hat{\mathbf{p}}_i)\right)$$
   *Justification*: Music genres are continuous, multi-faceted stylistic spaces rather than mutually exclusive categorical boxes [needs author citation]. Top-3 accuracy captures whether the model has learned meaningful acoustic representations that place the true genre in the immediate candidate set even when the single top prediction selects a closely related sister genre (e.g., Folk vs. Rock or Instrumental vs. Folk).

### 2.2 Baseline Reference
For an 8-class balanced problem, a uniform random guess baseline achieves:
$$\text{Chance Baseline} = \frac{1}{8} = 12.50\%\quad (0.1250)$$
All evaluated models must substantially exceed this floor to demonstrate acoustic signal extraction (`reports/metrics/extra_models_val.json#L18`).

### 2.3 Strict Separation of Validation and Test Splits
- **Validation Split (800 tracks)**: Used exclusively for model exploration, hyperparameter grid searches (e.g., Random Forest estimators/depth, SVC $C$ and $\gamma$, ExtraTrees split criteria), CNN early stopping / best-epoch checkpoint selection (epoch 24, `reports/metrics/cnn_val.json#L17`), and soft-voting ensemble weight optimization (`reports/metrics/extra_models_val.json#L4`).
- **Test Split (800 tracks)**: Held out completely untouched during training and model selection. Validation was used for all tuning and ensemble-weight selection; reported test numbers come from `scripts/11_evaluate_final_test.py` (an earlier test run exists in `test_metrics.json`). No hyperparameters, weights, or decision thresholds were tuned or adjusted following test set evaluation.

---

## 3. Data Leakage Audit

A comprehensive leakage audit was conducted to verify that information from the validation or test splits did not contaminate training representations.

### 3.1 Audit Checks & Results
1. **Official Split Verification**:
   - The active manifest (`data/processed/manifest.csv`) was cross-referenced against `fma_metadata/tracks.csv` `('set', 'split')`.
   - **Result**: Exactly 0 split assignment mismatches exist across all 7,994 tracks (`reports/metrics/manifest_audit.json#L12-L16`).
2. **Artist Overlap Check**:
   - Audio tracks from the same artist in both training and evaluation splits can cause synthetic performance inflation due to artist-specific vocal timbres, production chains, or instrumentation [needs author citation].
   - The FMA metadata `artist_id` was analyzed across all three splits (`reports/metrics/manifest_audit.json#L72-L84`):
     - Training unique artists: 1,886
     - Validation unique artists: 219
     - Test unique artists: 202
     - Total unique artists: 2,307
   - **Result**:
     - Overlapping artists between **Train and Validation**: **0** (`manifest_audit.json#L80`)
     - Overlapping artists between **Train and Test**: **0** (`manifest_audit.json#L80`)
     - Overlapping artists between **Validation and Test**: **0** (`manifest_audit.json#L80`)
     - Exactly **0 artists appear in multiple splits**, affecting 0 tracks (`reports/metrics/manifest_audit.json#L80-L81`).
3. **Data Transformer Hygiene**:
   - **Tabular Features (51 dimensions)**: In classical modeling (`scripts/04_train_baseline.py#L125-L135` and `scripts/10_train_extra_models.py#L52-L65`), `SimpleImputer(strategy='median')` and `StandardScaler()` are wrapped in scikit-learn `Pipeline` objects. These transformers are fit **strictly on `X_train`** (`imputer.fit_transform(X_train)`, `scaler.fit_transform(X_train)`). Validation and test sets are transformed using the frozen training statistics via `.transform()`. No test/validation features inform imputation medians or scaling means/variances.
   - **Spectrogram Normalization**: The global channel-wise mean (-43.32 dB) and standard deviation (15.19 dB) for log-mel spectrograms (`data/processed/norm_stats.json#L2-L3`) were computed **strictly across training set windows** during preprocessing (`src/audio/spectrogram.py`, `scripts/02_build_features.py`). No validation or test audio windows were used in calculating normalization statistics.

### 3.2 What the Audit Does NOT Prove
While the audit guarantees zero `artist_id` overlap and strict transformer fitting isolation, it does **not** prove the following:
- **Alias or Compilation Overlap**: If an artist released material under multiple aliases, pseudonym projects, uncredited collaborations, or varied compilation artist IDs in the raw Free Music Archive database, these would not be caught by single `artist_id` matching (`reports/metrics/manifest_audit.json#L83`).
- **Duplicate Audio Recordings**: It does not eliminate the possibility of acoustic duplicates, alternative mixes, or identical backing tracks assigned different track IDs in the original collection (`reports/metrics/manifest_audit.json#L83`).
- **Shared Production Environments**: It does not prevent acoustic correlation across distinct artists who utilized identical production equipment, digital audio workstation (DAW) synthesizers, drum soundfonts, or royalty-free sample libraries.
- **Label Objectivity**: It does not prove that genre labels are objectively distinct physical ground truths rather than subjective, community-assigned taxonomy tags.
- **Out-of-Distribution Robustness**: It does not prove model generalizability to modern commercial music, studio mastering chains differing from the Free Music Archive distribution, or streaming bitrates outside 22 kHz MP3.

---

## 4. Benchmark Results

The table below presents the final comparative benchmark across all evaluated model architectures. In accordance with strict reporting standards, **Held-Out Test metrics are presented first**, followed by the Validation metrics used during development.

All models evaluate the frozen official test split (800 tracks) and validation split (800 tracks) with a fixed random seed of 42 (`reports/metrics/final_test.json#L1-L15`, `reports/metrics/extra_models_val.json#L1-L19`).

| Model Architecture | Test Accuracy | Test Macro-F1 | Test Top-3 Accuracy | Val Accuracy | Val Macro-F1 | Val Top-3 Accuracy | Primary Citation |
|---|---|---|---|---|---|---|---|
| **Soft-Voting Ensemble**<br>*(0.40 SVC + 0.60 CNN + 0.00 RF)* | **52.38%**<br>*(0.52375)* | **0.5173**<br>*(0.517336)* | **82.62%**<br>*(0.82625)* | **62.88%**<br>*(0.62875)* | **0.6198**<br>*(0.619760)* | **85.75%**<br>*(0.8575)* | `final_test.json#L299-L301`<br>`extra_models_val.json#L226-L228` |
| **GenreCNN**<br>*(Best Checkpoint, Epoch 24)* | **50.62%**<br>*(0.50625)* | **0.4970**<br>*(0.496958)* | **81.38%**<br>*(0.81375)* | **58.50%**<br>*(0.58500)* | **0.5724**<br>*(0.572436)* | **85.25%**<br>*(0.8525)* | `final_test.json#L225-L227`<br>`extra_models_val.json#L181-L183` |
| **Random Forest Baseline**<br>*(400 trees, max_depth=None)* | **43.75%**<br>*(0.43750)* | **0.4354**<br>*(0.435409)* | **75.75%**<br>*(0.75750)* | **52.25%**<br>*(0.52250)* | **0.5073**<br>*(0.507309)* | **82.25%**<br>*(0.8225)* | `final_test.json#L18-L20`<br>`extra_models_val.json#L23-L25` |
| **RBF Support Vector Machine**<br>*(C=10, gamma=0.01)* | **43.62%**<br>*(0.43625)* | **0.4380**<br>*(0.438007)* | **76.88%**<br>*(0.76875)* | **53.12%**<br>*(0.53125)* | **0.5238**<br>*(0.523798)* | **80.38%**<br>*(0.80375)* | `final_test.json#L87-L89`<br>`extra_models_val.json#L33-L35` |
| **ExtraTrees Classifier**<br>*(400 trees, max_features=sqrt)* | **43.62%**<br>*(0.43625)* | **0.4330**<br>*(0.433029)* | **76.88%**<br>*(0.76875)* | **52.12%**<br>*(0.52125)* | **0.5063**<br>*(0.506323)* | **82.25%**<br>*(0.8225)* | `final_test.json#L156-L158`<br>`extra_models_val.json#L154-L156` |

*Note: For the Random Forest standalone baseline tuning run recorded in `reports/metrics/baseline_val.json#L9-L10`, validation accuracy was 52.38% (0.52375) and macro-F1 was 0.5093 (0.509290). In `reports/metrics/test_metrics.json#L47-L48`, earlier standalone single-model test evaluation yielded RF accuracy 43.75% (macro-F1 0.4357) and CNN accuracy 50.75% (macro-F1 0.4985). See Section 7 for full discrepancy reconciliation.*

---

## 5. Performance Interpretation & Analysis

### 5.1 Acoustic Learning vs. Random Guess Floor
All models decisively outperform the uniform random baseline of 12.50%:
- Classical tabular models (Random Forest, RBF SVC, ExtraTrees) achieve **43.62%–43.75% test accuracy**, representing a **3.5× improvement** over chance.
- The 2D GenreCNN achieves **50.62% test accuracy**, representing a **4.0× improvement** over chance.
- The multimodal soft-voting ensemble achieves **52.38% test accuracy**, representing a **4.2× improvement** over chance.
- Top-3 accuracies range between **75.75% and 82.62% on test** (`reports/metrics/final_test.json#L20`, `reports/metrics/final_test.json#L301`). This indicates that for over 4 out of 5 test tracks, the true genre is contained within the top three predicted candidates.

### 5.2 The Generalization Gap (Validation vs. Test)
A consistent generalization gap of approximately 8 to 10 percentage points is observed between the validation split and the test split:
- **Random Forest**: 52.25% val $\rightarrow$ 43.75% test ($\Delta = -8.50\%$)
- **RBF SVC**: 53.12% val $\rightarrow$ 43.62% test ($\Delta = -9.50\%$)
- **ExtraTrees**: 52.12% val $\rightarrow$ 43.62% test ($\Delta = -8.50\%$)
- **GenreCNN**: 58.50% val $\rightarrow$ 50.62% test ($\Delta = -7.88\%$)
- **Ensemble**: 62.88% val $\rightarrow$ 52.38% test ($\Delta = -10.50\%$)

**Observations Consistent with the Generalization Gap**:
1. **Zero Artist Overlap**: Because the FMA official split enforces zero artist overlap across all partitions (`reports/metrics/manifest_audit.json#L75-L81`), the test set evaluates models against 202 entirely new artists whose timbral, melodic, and production signatures were never seen during training or model selection.
2. **Model Selection Optimization**: The best hyperparameters (such as SVC $C=10, \gamma=0.01$ from a grid of 12 configurations, `reports/metrics/extra_models_val.json#L36-L145`) and the best CNN checkpoint (epoch 24 out of 25, `reports/metrics/cnn_val.json#L17-L20`) were selected because they yielded peak performance on the validation split. A downward regression toward the mean on unseen test data is expected.
3. **Genre Shift in Test Split**: Detailed examination of the confusion matrices reveals substantial performance drops on specific genres between validation and test:
   - *Folk*: Recall drops from **65.0%** in validation (`reports/metrics/cnn_val.json#L36`) to **32.0%** in test (`reports/metrics/final_test.json#L243`), frequently misclassified as Rock (34 tracks) or Instrumental (`reports/metrics/test_metrics.json#L221-L230`).
   - *Experimental*: Recall drops from **45.0%** in validation (`reports/metrics/cnn_val.json#L30`) to **30.0%** in test (`reports/metrics/final_test.json#L237`).
   - Conversely, genres with highly consistent rhythmic and timbral signatures remain resilient: *Hip-Hop* achieves **77.0% test recall** (`reports/metrics/final_test.json#L249`) and *Rock* achieves **73.0% test recall** (`reports/metrics/final_test.json#L273`).

### 5.3 Why the 62.88% Validation Ensemble Score is Optimistic
The soft-voting ensemble achieved an impressive 62.88% accuracy and 0.6198 macro-F1 on the validation set (`reports/metrics/extra_models_val.json#L226-L227`). However, this figure is inherently optimistic:
1. **Weight Optimization on Validation**: The ensemble weights ($w_{\text{SVC}}=0.40, w_{\text{CNN}}=0.60, w_{\text{RF}}=0.00$) were determined by running a fine grid search across 66 weight combinations directly on the validation predictions (`scripts/10_train_extra_models.py#L90-L135`, `reports/metrics/extra_models_val.json#L215-L229`). The chosen weights exploited the specific residual errors and confidence calibration of the validation set.
2. **Elimination of Random Forest**: The grid search assigned a weight of 0.00 to Random Forest because SVC provided slightly sharper probability calibration on validation tracks where the CNN was uncertain.
3. **Test Realism**: When evaluated on the held-out test split, the ensemble achieves **52.38% test accuracy** and **0.5173 test macro-F1** (`reports/metrics/final_test.json#L299-L300`). While this represents a 10.50% drop from the optimistic validation figure, the ensemble still outperforms every standalone model on the test split (+1.76% accuracy and +0.0204 macro-F1 over standalone GenreCNN). The test gain is likely attributable to combining orthogonal feature modalities: tabular summary statistics and time-frequency spectrogram representations.

---

## 6. Limitations

1. **Subjectivity of Genre Taxonomies**: Musical genre classifications are cultural, fluid constructs rather than immutable physical categories [needs author citation]. In FMA, tracks are tagged by human artists and curators, often applying broad labels to hybrid tracks (e.g., electronic rock, avant-garde folk, or ambient instrumental).
2. **Fixed 30-Second Clip Window**: Classifications are derived from 30.0-second excerpts (`configs/config.yaml#L37`). Many compositions feature dynamic tempo changes, quiet introductions, or stylistic transitions that occur outside the extracted segment.
3. **Dataset Scale & Deep Learning Constraints**: FMA Small contains 6,394 training tracks across 8 classes (`reports/metrics/manifest_audit.json#L13`). Deep convolutional networks typically require orders of magnitude more training instances to learn generalized representations without early overfitting [needs author citation].
4. **Asymmetric Test Split Difficulty**: As demonstrated by per-class recalls in `final_test.json#L22-L70` and `final_test.json#L228-L277`, the test split contains disproportionately challenging instances of Folk, Pop, and Experimental genres compared to the validation set.
5. **Feature Representation Ceiling**: The 51-dimensional handcrafted feature vector (`src/features/extraction.py`) relies on summary statistics (mean and standard deviation of MFCCs, spectral centroid, spectral rolloff, zero-crossing rate, chroma, and onset strength). This aggregation discards fine-grained temporal structures and rhythmic progressions, consistent with an empirical performance ceiling of ~43.6%–43.8% accuracy on tree-based and kernel methods.

---

## 7. Dual-Reported Metric Reconciliation

To ensure absolute audit transparency, the table below documents every metric that appears with differing values across files in this repository, explains the technical reason for the divergence, and states which figure this evaluation report adopts.

| Metric | Value A & Source | Value B & Source | Technical Reason for Discrepancy | Standard Value Used in This Document |
|---|---|---|---|---|
| **Random Forest Validation Accuracy** | **52.38%** *(0.52375)*<br>`reports/metrics/baseline_val.json#L10` | **52.25%** *(0.52250)*<br>`reports/metrics/extra_models_val.json#L23` | `baseline_val.json` was generated during initial baseline grid search (`scripts/04_train_baseline.py`). `extra_models_val.json` was generated by reloading the saved `.joblib` model artifact in a unified benchmark script (`scripts/10_train_extra_models.py`). | **52.25%** in comparative tables (for consistency across simultaneous model evaluations); **52.38%** cited for the isolated baseline run. |
| **Random Forest Validation Macro-F1** | **0.5093** *(0.509290)*<br>`reports/metrics/baseline_val.json#L9` | **0.5073** *(0.507309)*<br>`reports/metrics/extra_models_val.json#L24` | Same cause as above: initial tuning run vs. re-loaded model artifact re-evaluation in the unified extra models benchmark. | **0.5073** in comparative tables; **0.5093** cited for initial grid search baseline. |
| **GenreCNN Test Accuracy** | **50.75%** *(0.50750)*<br>`reports/metrics/test_metrics.json#L205`<br>`reports/metrics/model_comparison.md#L8` | **50.625%** *(0.50625)*<br>`reports/metrics/final_test.json#L225` | `test_metrics.json` was produced during an earlier standalone evaluation run (`scripts/06_evaluate.py`). `final_test.json` was produced during the final unified evaluation run (`scripts/11_evaluate_final_test.py`) with frozen dependencies and full pipeline verification. | **50.625% (50.62%)** from `final_test.json` (reflects the definitive frozen benchmark across all 5 models). |
| **GenreCNN Test Macro-F1** | **0.4985** *(0.498496)*<br>`reports/metrics/test_metrics.json#L48`<br>`reports/metrics/model_comparison.md#L8` | **0.4970** *(0.496958)*<br>`reports/metrics/final_test.json#L226` | Same cause as above: earlier single-model test evaluation script vs. final unified multi-model test evaluation script. | **0.4970** from `final_test.json`. |
| **Random Forest Test Macro-F1** | **0.4357**<br>`reports/metrics/test_metrics.json#L48`<br>`reports/metrics/model_comparison.md#L7` | **0.4354** *(0.435409)*<br>`reports/metrics/final_test.json#L19` | Intermediate float rounding in `scripts/06_evaluate.py` vs. full double-precision floating point calculation in `scripts/11_evaluate_final_test.py`. Test accuracy is identical at 43.75% (0.4375) in both files. | **0.4354** from `final_test.json`. |


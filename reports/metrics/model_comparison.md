# SonicFusion 2: Test Split Model Comparison

Evaluated on official held-out test split (800 tracks).

| Model | Test Accuracy (95% CI) | Macro-F1 (95% CI) | Median Inference Latency |
|---|---|---|---|
| Random Forest | 0.4375 [0.4038, 0.4712] | 0.4357 [0.4037, 0.4675] | 211.73 ms |
| 2D CNN | 0.5075 [0.4762, 0.5425] | 0.4985 [0.4671, 0.5302] | 28.26 ms |

### Paired Statistical Comparison (Accuracy Difference: CNN - RF)
- 95% Bootstrap CI on Accuracy Difference: [0.0387, 0.1038]
- Mean Accuracy Difference: +0.0705
- Paired Bootstrap p-value: 0.0000
- McNemar Chi-Square Statistic: 14.6845 (p-value: 0.0001, exact binomial p-value: 0.0001)

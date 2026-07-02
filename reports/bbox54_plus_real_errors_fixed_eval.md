# Retraining Manifest V2 Evaluation Report

This report presents a direct performance comparison between the baseline and retrained models under identical evaluation splits.

## Run Configuration
* **Feature Schema**: keypoint_bbox54
* **Input Size**: 54
* **Device**: cuda
* **Epochs**: 20
* **Seed**: 42

## Split Distributions
| Split | Total Rows | Normal Count | Faint Count | Faint Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **train (baseline)** | 7,000 | 3,500 | 3,500 | 50.00% |
| **train (retrained)** | 7,000 | 3,500 | 3,500 | 50.00% |
| **val (baseline)** | 1,500 | 750 | 750 | 50.00% |
| **val (retrained)** | 1,500 | 750 | 750 | 50.00% |
| **test** | 1,400 | 700 | 700 | 50.00% |

---

## Overall Classification Performance
*Primary threshold = 0.5*

| Model Variant | Accuracy | Precision | Recall | F1 Score | FP Count | FN Count | TP Count | TN Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (54-dim)** | 93.43% | 91.13% | 96.29% | 93.64% | 65 | 26 | 674 | 635 |
| **Retrained (54-dim)** | 93.43% | 91.13% | 96.29% | 93.64% | 65 | 26 | 674 | 635 |

---

## Threshold Sweep Analysis
Compare F1, Recall, and FP counts across different threshold configurations.

### Baseline (54-dim)
| Threshold | F1 Score | Recall | Precision | FP Count | FN Count |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.3** | 91.68% | 98.43% | 85.80% | 114 | 11 |
| **0.4** | 92.93% | 97.43% | 88.82% | 86 | 18 |
| **0.5** | 93.64% | 96.29% | 91.13% | 65 | 26 |
| **0.6** | 94.02% | 94.71% | 93.35% | 47 | 37 |
| **0.7** | 93.81% | 91.43% | 96.32% | 24 | 60 |

### Retrained (54-dim)
| Threshold | F1 Score | Recall | Precision | FP Count | FN Count |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.3** | 91.68% | 98.43% | 85.80% | 114 | 11 |
| **0.4** | 92.93% | 97.43% | 88.82% | 86 | 18 |
| **0.5** | 93.64% | 96.29% | 91.13% | 65 | 26 |
| **0.6** | 94.02% | 94.71% | 93.35% | 47 | 37 |
| **0.7** | 93.81% | 91.43% | 96.32% | 24 | 60 |

---

## Summary and Interpretation
1. **FP / FN Trend**: The retrained model shows no performance deviation from the baseline model.
2. **Recall Stability**: Recall is preserved under all threshold parameters.
3. **F1 Peak**: Peak F1 performance is achieved at threshold = 0.6.

*Report generated automatically on 2026-07-02T13:41:40.854890.*

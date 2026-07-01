# Retraining Manifest v2 Performance Comparison Report

**Date:** 2026-07-01
**Evaluation Type:** MOCK / DRY-RUN (Validation)

This report evaluates the improvements in the LSTM Fall Classifier model after applying Hard Negative Mining, Faint Reinforcement, and Synthetic Data Augmentation using `training_manifest_v2.csv`.

## Performance Metrics Summary

| Metric | Baseline (metadata.csv) | Retrained (training_manifest_v2.csv) | Improvement |
| :--- | :---: | :---: | :---: |
| Accuracy | 0.7300 | 0.8800 | +0.1500 |
| Precision | 0.7510 | 0.8580 | +0.1070 |
| Recall | 0.7000 | 0.8920 | +0.1920 |
| F1-score | 0.7250 | 0.8750 | +0.1500 |
| False Positives (FP) | 15 | 5 | -10 |
| False Negatives (FN) | 12 | 3 | -9 |

## Confusion Matrix Comparison

### Baseline
```text
            Predicted Normal   Predicted Faint
Actual Normal      45                 15
Actual Faint       12                 28
```

### Retrained Model
```text
            Predicted Normal   Predicted Faint
Actual Normal      57                 5
Actual Faint       3                  35
```

## Scenario-Tag Analysis

| Scenario Tag | Baseline FP | Retrained FP | Baseline FN | Retrained FN | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| bending_false_positive | 4 | 1 | 0 | 0 | Improved |
| far_distance_false_negative | 0 | 0 | 4 | 1 | Improved |
| night_false_negative | 0 | 0 | 5 | 1 | Improved |
| sitting_false_positive | 3 | 0 | 0 | 0 | Improved |

## Synthetic Augmentation Analysis

| Augmentation Type | Baseline FP | Retrained FP | Baseline FN | Retrained FN | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| brightness | 0 | 0 | 3 | 0 | FP: +0, FN: -3 |
| noise | 0 | 0 | 2 | 0 | FP: +0, FN: -2 |

## Threshold Sweep (0.3 ~ 0.7)

| Threshold | Baseline F1 | Retrained F1 | Baseline Recall | Retrained Recall |
| :---: | :---: | :---: | :---: | :---: |
| 0.3 | 0.6887 | 0.8312 | 0.7350 | 0.9366 |
| 0.4 | 0.7105 | 0.8575 | 0.7140 | 0.9098 |
| 0.5 | 0.7250 | 0.8750 | 0.7000 | 0.8920 |
| 0.6 | 0.7032 | 0.8488 | 0.6440 | 0.8206 |
| 0.7 | 0.6670 | 0.8050 | 0.5740 | 0.7314 |

## Key Findings & Trade-offs
- **Hard Negative Mining:** FP count decreased in most false-positive scenario tags.
- **Faint Reinforcement:** FN count decreased in faint-reinforcement tags.
- **Synthetic Augmentation:** Accuracy under night/far/occluded conditions improved.
- **Trade-off:** Improving recall can sometimes lead to a slight drop in precision. This sweep helps find the optimal threshold.

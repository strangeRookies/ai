# GPU PC LSTM Retraining and Evaluation Run Guide

This guide explains how to execute the LSTM fall classification retraining and evaluation pipeline on the GPU PC or remote GPU server. 

## 1. Local vs. GPU PC Environments

| Feature / Step | Local Development Environment | GPU PC / Remote GPU Server |
| :--- | :--- | :--- |
| **Primary Purpose** | Code modification, pipeline verification, dry-run checks | Real model training, final evaluation, comparative benchmark |
| **Data Source** | Mock keypoints (auto-generated when missing), sample metadata | Real extracted YOLO pose keypoint cache (NPZs) and video clips |
| **Execution Mode** | `--dry-run` or `--sample 100` | Real-mode full dataset training & validation |
| **Hardware** | CPU / mock execution | NVIDIA GPU (CUDA enabled) |

> [!WARNING]
> **Mock Mode is NOT a performance measure.**
> Running evaluations locally in `--dry-run` or with mock fallback will generate simulated or random performance improvements to verify the plumbing. **Only real-mode evaluation on the GPU PC is valid for production comparison.**

---

## 2. Environment Verification on GPU PC

Before starting, log into the GPU PC via SSH and check the PyTorch and CUDA status.

```bash
# 1. Navigate to the project root directory
cd ~/yolo_training/strange_ai_lstm/strange_ai

# 2. Check Python version
python --version

# 3. Check PyTorch installation and CUDA availability
python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    print("[ERROR] CUDA is not available on this GPU PC! Check nvidia drivers.")
PY
```

---

## 3. Configuration Pathways Checklist

Ensure these directories and files are present on the GPU PC before running:

* **Baseline Metadata CSV:** `../ai_fall_experiments/data/metadata/metadata.csv`
* **V2 Training Manifest:** `data/manifests/training_manifest_v2.csv`
* **Keypoint Cache Directory:** `../ai_fall_experiments/data/keypoints/yolo26n-pose`
* **Video Clip Directory:** `/home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos`
* **LSTM Checkpoint Output Directory:** `runs/action_lstm/`
* **Report Output Path:** `reports/retraining_manifest_v2_eval.md`

---

## 4. Execution Workflow (Step-by-Step)

### Step 4.1. Manifest Verification & Leakage Check
Verify that the `training_manifest_v2.csv` has no split leaks or review-status violations:
```bash
python scripts/check_manifest_leakage.py \
  --manifest data/manifests/training_manifest_v2.csv
```

### Step 4.2. Sample-based Evaluation (Dry-Run Verification)
Verify that paths, models, and scripts work correctly using a small subset (100 rows) before running the full dataset:
```bash
python scripts/evaluate_retraining_manifest_v2.py \
  --baseline ../ai_fall_experiments/data/metadata/metadata.csv \
  --retrained data/manifests/training_manifest_v2.csv \
  --sample 100 \
  --output-dir runs/evaluation_sample \
  --report-path reports/retraining_manifest_v2_eval_sample.md
```

### Step 4.3. Run Full Real-Mode Retraining & Evaluation
If the sample evaluation completes successfully, run the comparative evaluation on the full real dataset. 

This command will:
1. Load keypoint cache sequences for the full training dataset.
2. Train two LSTM models (Baseline model vs. Retrained model) on their respective training sets.
3. Evaluate both models on the *same* common test split.
4. Generate the performance report showing Precision, Recall, F1 improvements, Scenario tag breakdowns, and Synthetic data performance.

```bash
python scripts/evaluate_retraining_manifest_v2.py \
  --baseline ../ai_fall_experiments/data/metadata/metadata.csv \
  --retrained data/manifests/training_manifest_v2.csv \
  --output-dir runs/evaluation_real \
  --report-path reports/retraining_manifest_v2_eval_real.md \
  --device cuda \
  --epochs 20 \
  --batch-size 32 \
  --seed 42
```

---

## 5. Alternative: Step-by-Step Training & Checkpoint Evaluation

If you prefer to train the LSTM model first using the standard training script, and then evaluate the saved checkpoint:

### Step 5.1. Train LSTM Action Model
Use the main project training script to train the model on the V2 manifest:
```bash
python -m ai.action.train_lstm \
  --dataset-csv data/manifests/training_manifest_v2.csv \
  --detector-mode yolo \
  --yolo-model yolo26n-pose.pt \
  --epochs 30 \
  --batch-size 64 \
  --device cuda \
  --output-dir runs/retraining_manifest_v2
```
*Creates model weights at `runs/retraining_manifest_v2/best.pt`.*

### Step 5.2. Evaluate Pre-trained Checkpoint
Evaluate the resulting checkpoint using the evaluation script:
```bash
python scripts/evaluate_retraining_manifest_v2.py \
  --baseline ../ai_fall_experiments/data/metadata/metadata.csv \
  --retrained data/manifests/training_manifest_v2.csv \
  --checkpoint runs/retraining_manifest_v2/best.pt \
  --output-dir runs/evaluation_retrained_checkpoint \
  --report-path reports/retraining_manifest_v2_checkpoint_eval.md \
  --device cuda
```

---

## 6. Generated Output Files

* **Markdown Report:** `reports/retraining_manifest_v2_eval_real.md` (Summary of F1-scores, accuracy, scenario FP/FN drops)
* **JSON Metrics Summary:** `runs/evaluation_real/evaluation_summary.json` (Raw metrics for both baseline and retrained runs)

---

## 7. Troubleshooting Checklist

| Error / Issue | Root Cause | Resolution |
| :--- | :--- | :--- |
| `FileNotFoundError` for CSVs | Manifest path incorrect | Confirm `--baseline` and `--retrained` arguments match actual GPU PC paths. |
| `RuntimeError` regarding keypoints | Missing cache (`.npz` files) | Run `python scripts/extract_keypoint_cache.py --metadata-csv <csv> --cache-dir <dir>` first to generate cache files. |
| `RuntimeError: split_group_id leakage` | Parent split leak detected | The manifest leakage checker failed. Ensure no parent clip has children in different splits. |
| `RuntimeError: pending/rejected rows` | Candidate rows not approved | Candidates with `review_status` of `pending`, `rejected`, or `needs_review` were written to the training manifest. Run `check_manifest_leakage.py` to verify. |
| CUDA out of memory / unavailable | PyTorch cannot access GPU | Verify `nvidia-smi` works. Reduce `--batch-size` to 16 or 32 if memory runs out. |

---

## 8. Exporting Results Back to Local

After completing the evaluation run on the GPU PC, download the generated markdown report to local workspace to present it:

```bash
# On your local machine's terminal
scp GPU_USER@GPU_HOST:~/yolo_training/strange_ai_lstm/strange_ai/reports/retraining_manifest_v2_eval_real.md ./docs/retraining_gpu_eval_report.md
```

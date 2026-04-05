#!/bin/bash
###############################################################################
# run_full.sh — Full production pipeline for Hateful Meme Classifier
#
# This is the ONLY command needed to execute the entire project from scratch:
#
#   bash scripts/run_full.sh
#
# Pipeline:
#   1. Create virtual environment & install dependencies
#   2. Verify dataset integrity
#   3. Run Exploratory Data Analysis (EDA)
#   4. Train baseline model (Logistic Regression on CLIP features)
#   5. Train final model (Cross-Attention on CLIP features)
#   6. Plot training curves
#   7. Evaluate on dev set
#   8. Run error analysis
#   9. Generate test set predictions (CSV)
#  10. Run a sample prediction (CLI demo)
#  11. Launch Gradio web app
#
# Every output is timestamped with the run ID: YYYYMMDD_HHMMSS
###############################################################################
set -euo pipefail

# ─── Resolve project root (works from any directory) ─────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ─── Run timestamp ──────────────────────────────────────────────────────────
RUN_TS="$(date +%Y%m%d_%H%M%S)"
echo ""
echo "================================================================"
echo "  HATEFUL MEME CLASSIFIER — FULL PIPELINE"
echo "  Run ID: ${RUN_TS}"
echo "  Started: $(date)"
echo "  Project: ${PROJECT_ROOT}"
echo "================================================================"
echo ""

###############################################################################
# STEP 1/11 — Virtual environment & dependencies
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 1/11 — Setting up virtual environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

VENV_DIR="${PROJECT_ROOT}/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists at ${VENV_DIR}, reusing."
fi

# Activate
source "${VENV_DIR}/bin/activate"
echo "Python: $(which python) ($(python --version 2>&1))"

# Upgrade pip quietly
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing requirements..."
pip install -r requirements.txt --quiet
echo "All dependencies installed."
echo ""

###############################################################################
# STEP 2/11 — Verify dataset
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 2/11 — Verifying dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DATA_DIR="${PROJECT_ROOT}/data"
MISSING=0

for f in train.jsonl dev.jsonl test.jsonl; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        echo "ERROR: Missing ${DATA_DIR}/${f}"
        MISSING=1
    else
        COUNT=$(wc -l < "${DATA_DIR}/${f}" | tr -d ' ')
        echo "  ${f}: ${COUNT} samples"
    fi
done

if [ ! -d "${DATA_DIR}/img" ]; then
    echo "ERROR: Missing ${DATA_DIR}/img/ directory"
    MISSING=1
else
    IMG_COUNT=$(ls "${DATA_DIR}/img/" | wc -l | tr -d ' ')
    echo "  img/: ${IMG_COUNT} images"
fi

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "Dataset is incomplete. Please place the Hateful Memes dataset in ${DATA_DIR}/"
    echo "Required: train.jsonl, dev.jsonl, test.jsonl, img/ folder"
    exit 1
fi
echo "Dataset OK."
echo ""

###############################################################################
# STEP 3/11 — Exploratory Data Analysis
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 3/11 — Exploratory Data Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m src.eda 2>&1
echo ""

###############################################################################
# STEP 4/11 — Train baseline model
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 4/11 — Training baseline (Logistic Regression on CLIP)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m src.baseline --config config/config.yaml 2>&1
echo ""

###############################################################################
# STEP 5/11 — Train final model
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 5/11 — Training final model (Cross-Attention)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create timestamped directories
CKPT_DIR="${PROJECT_ROOT}/checkpoints/run_${RUN_TS}"
LOG_DIR="${PROJECT_ROOT}/logs/run_${RUN_TS}"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# Save run config
RUN_CONFIG="${LOG_DIR}/config_${RUN_TS}.yaml"
python -c "
import yaml
with open('config/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
with open('${RUN_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Run config written to ${RUN_CONFIG}')
"

TRAIN_LOG="${LOG_DIR}/train_${RUN_TS}.log"
echo "Training log: ${TRAIN_LOG}"
echo "Checkpoints:  ${CKPT_DIR}/"
echo ""
echo "Training started at $(date)..."

python -m src.train --config "${RUN_CONFIG}" 2>&1 | tee "$TRAIN_LOG"

TRAIN_END="$(date)"
echo ""
echo "Training finished at ${TRAIN_END}"

# Copy best checkpoint to timestamped directory
if [ -f "checkpoints/best_model.pt" ]; then
    cp "checkpoints/best_model.pt" "${CKPT_DIR}/best_model_${RUN_TS}.pt"
    echo "Best checkpoint copied to ${CKPT_DIR}/best_model_${RUN_TS}.pt"
fi
echo ""

###############################################################################
# STEP 6/11 — Plot training curves
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 6/11 — Plotting training curves"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m src.plot_training_curves --log "$TRAIN_LOG" --output outputs 2>&1
echo ""

###############################################################################
# STEP 7/11 — Evaluate on dev set
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 7/11 — Evaluating on dev set"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EVAL_DEV_LOG="${LOG_DIR}/eval_dev_${RUN_TS}.log"

python -m src.evaluate \
    --config "${RUN_CONFIG}" \
    --split dev \
    --checkpoint "checkpoints/best_model.pt" 2>&1 | tee "$EVAL_DEV_LOG"

# Move confusion matrix to timestamped location
if [ -f "logs/confusion_matrix.png" ]; then
    cp "logs/confusion_matrix.png" "${LOG_DIR}/confusion_matrix_dev_${RUN_TS}.png"
    echo "Confusion matrix: ${LOG_DIR}/confusion_matrix_dev_${RUN_TS}.png"
fi
echo ""

###############################################################################
# STEP 8/11 — Error analysis
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 8/11 — Running error analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m src.error_analysis \
    --config "${RUN_CONFIG}" \
    --checkpoint "checkpoints/best_model.pt" \
    --top-k 8 2>&1
echo ""

###############################################################################
# STEP 9/11 — Generate test set predictions (CSV)
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 9/11 — Generating test predictions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m src.test \
    --config "${RUN_CONFIG}" \
    --checkpoint "checkpoints/best_model.pt" \
    --output "outputs/test_predictions_${RUN_TS}.csv" 2>&1
echo ""

###############################################################################
# STEP 10/11 — Sample prediction (CLI demo)
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 10/11 — Sample prediction (CLI)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SAMPLE_IMG=$(ls data/img/ | head -1)
SAMPLE_PATH="data/img/${SAMPLE_IMG}"

echo "Running prediction on: ${SAMPLE_PATH}"
python -m src.predict "${SAMPLE_PATH}" \
    --config "${RUN_CONFIG}" \
    --checkpoint "checkpoints/best_model.pt" 2>&1
echo ""

###############################################################################
# STEP 11/11 — Launch Gradio web app
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[${RUN_TS}] STEP 11/11 — Launching Gradio web app"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "================================================================"
echo "  PIPELINE COMPLETE"
echo "  Run ID:       ${RUN_TS}"
echo "  Finished at:  $(date)"
echo ""
echo "  Outputs:"
echo "    EDA plots:          outputs/eda/"
echo "    Baseline results:   outputs/baseline_results.txt"
echo "    Baseline confusion: outputs/baseline_confusion_matrix.png"
echo "    Training curves:    outputs/training_summary.png"
echo "    Training log:       ${TRAIN_LOG}"
echo "    Checkpoint:         ${CKPT_DIR}/best_model_${RUN_TS}.pt"
echo "    Dev eval log:       ${EVAL_DEV_LOG}"
echo "    Dev confusion:      ${LOG_DIR}/confusion_matrix_dev_${RUN_TS}.png"
echo "    Error analysis:     outputs/error_analysis/"
echo "    Test predictions:   outputs/test_predictions_${RUN_TS}.csv"
echo ""
echo "  Launching web app at http://localhost:7860"
echo "  Press Ctrl+C to stop."
echo "================================================================"
echo ""

python -m app.app 2>&1 | tee "${LOG_DIR}/app_${RUN_TS}.log"

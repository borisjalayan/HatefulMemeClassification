#!/bin/bash
# Run evaluation on the Hateful Meme Classifier
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-config/config.yaml}"
SPLIT="${2:-dev}"
CHECKPOINT="${3:-checkpoints/best_model.pt}"

echo "Evaluating on $SPLIT split"
python -m src.evaluate --config "$CONFIG" --split "$SPLIT" --checkpoint "$CHECKPOINT"

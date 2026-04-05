#!/bin/bash
# Launch training for the Hateful Meme Classifier
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-config/config.yaml}"

echo "Starting training with config: $CONFIG"
python -m src.train --config "$CONFIG"

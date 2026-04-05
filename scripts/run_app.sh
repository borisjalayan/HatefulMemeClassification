#!/bin/bash
# Launch the Gradio web UI
# Usage:
#   bash scripts/run_app.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Starting Gradio web app on http://localhost:7860"
python app/app.py

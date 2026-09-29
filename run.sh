#!/bin/bash
# Launch the YuE2 Gradio UI on a pod that already has yue2-infer installed
# (see references/models-and-setup.md in the parent skill directory).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f "/workspace/YuE/.venv/bin/activate" ]; then
  source /workspace/YuE/.venv/bin/activate
fi

pip show gradio > /dev/null 2>&1 || pip install --quiet gradio

python app/gradio_app.py --port "${PORT:-7860}" "$@"

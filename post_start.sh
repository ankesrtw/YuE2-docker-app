#!/bin/bash
# RunPod's /start.sh (the base image's default CMD) brings up sshd (from
# $PUBLIC_KEY) and Jupyter (when $JUPYTER_PASSWORD is set), then runs this hook.
# Must not block, or /start.sh never finishes -- so Gradio is backgrounded.
LOG=/workspace/gradio.log
mkdir -p /workspace
nohup bash /opt/YuE/skills/yue2-music/app/run.sh >> "$LOG" 2>&1 &
echo "YuE2 Gradio starting on :${PORT:-7860}, log: $LOG"

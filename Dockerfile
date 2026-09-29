# YuE2 Gradio UI image for RunPod.
# Model weights (YuE2-3B, YuE2-Vae, ~10GB) are NOT baked in here -- they are
# downloaded from Hugging Face on first run and cached to /workspace, which
# should be a RunPod volume/disk so the download only happens once per pod.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

RUN git clone https://github.com/multimodal-art-projection/YuE.git YuE

WORKDIR /workspace/YuE

RUN python -m venv .venv \
    && . .venv/bin/activate \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir "gradio" "huggingface-hub<1.0"

COPY gradio_app.py skills/yue2-music/app/gradio_app.py
COPY run.sh skills/yue2-music/app/run.sh
RUN chmod +x skills/yue2-music/app/run.sh

EXPOSE 7860

ENV PORT=7860
CMD ["/bin/bash", "-c", "mkdir -p ~/.ssh && (echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys 2>/dev/null || true) && service ssh start 2>/dev/null; source .venv/bin/activate && bash skills/yue2-music/app/run.sh"]

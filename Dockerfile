# YuE2 Gradio UI image for RunPod.
# Model weights (YuE2-3B, YuE2-Vae, ~10GB) are NOT baked in here -- they are
# downloaded from Hugging Face on first run and cached under HF's default
# cache dir. The app itself lives under /opt, NOT /workspace: RunPod pod
# templates mount a persistent volume at /workspace, which would shadow
# anything baked into the image at that path on a fresh (empty) volume.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /opt

RUN git clone https://github.com/multimodal-art-projection/YuE.git YuE

WORKDIR /opt/YuE

RUN python -m venv .venv \
    && . .venv/bin/activate \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir "gradio>=5,<6" "huggingface-hub<1.0" "hf_transfer"

COPY gradio_app.py skills/yue2-music/app/gradio_app.py
COPY run.sh skills/yue2-music/app/run.sh
RUN chmod +x skills/yue2-music/app/run.sh

# Do NOT override CMD: the base image's /start.sh owns sshd + Jupyter setup and
# calls /post_start.sh once they are up.
COPY post_start.sh /post_start.sh
RUN chmod +x /post_start.sh

# 22 = ssh, 7860 = Gradio, 8888 = Jupyter (set JUPYTER_PASSWORD in the template)
EXPOSE 22 7860 8888
ENV PORT=7860

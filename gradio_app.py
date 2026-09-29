#!/usr/bin/env python3
"""Gradio UI for manual, one-off YuE2 song generation.

Loads the YuE2Pipeline once at startup and reuses it across requests,
unlike the CLI (skills/yue2-music/scripts/run_yue2.py) which reloads
the model per invocation. Intended to run on a GPU pod that already
has yue2-infer installed (see references/models-and-setup.md).
"""

import argparse
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr

DEFAULT_MODEL = "m-a-p/YuE2-3B"
DEFAULT_VAE = "m-a-p/YuE2-Vae"
COT_CHOICES = ["full", "melody", "off"]

_pipe = None
_pipe_kwargs = None


def load_pipeline(model, vae, device, memory_budget_gib, offline):
    global _pipe, _pipe_kwargs
    kwargs = dict(model=model, vae=vae, device=device,
                  memory_budget_gib=memory_budget_gib, local_files_only=offline)
    if _pipe is not None and _pipe_kwargs == kwargs:
        return _pipe
    from yue2 import YuE2Pipeline
    if _pipe is not None:
        _pipe.close()
    _pipe = YuE2Pipeline.from_pretrained(**kwargs)
    _pipe_kwargs = kwargs
    return _pipe


def generate(style, lyrics, cot, seed, cfg_scale, abc_text,
             model, vae, device, memory_budget_gib, offline, progress=gr.Progress()):
    from yue2.protocol import SongRequest

    if not style or not style.strip():
        raise gr.Error("Style/tags is required.")
    if cot != "off" and (not lyrics or not lyrics.strip()):
        raise gr.Error("Lyrics are required for full/melody modes.")

    progress(0.05, desc="Loading model (cached after first run)...")
    try:
        pipe = load_pipeline(model, vae, device, memory_budget_gib, offline)
    except Exception as exc:
        raise gr.Error(f"Failed to load model: {exc}") from exc

    req_kwargs = dict(style=style.strip(), cot=cot, id="gradio_song")
    if cot != "off":
        req_kwargs["lyrics"] = lyrics.strip()
    if seed is not None and seed != "":
        req_kwargs["seed"] = int(seed)
    if cfg_scale is not None and cfg_scale != "":
        req_kwargs["cfg_scale"] = float(cfg_scale)
    if abc_text and abc_text.strip():
        if cot == "off":
            raise gr.Error("ABC input is not allowed with cot=off.")
        req_kwargs["abc"] = abc_text.strip()

    try:
        request = SongRequest(**req_kwargs)
    except Exception as exc:
        raise gr.Error(f"Invalid request: {exc}") from exc

    progress(0.15, desc="Generating (planning, synthesis, decoding)...")
    workdir = Path(tempfile.mkdtemp(prefix="yue2_gradio_"))
    try:
        start = time.perf_counter()
        song = pipe(**request.to_dict())
        elapsed = time.perf_counter() - start
        receipt = song.save_artifacts(workdir)
        progress(1.0, desc="Done")
    except Exception as exc:
        traceback.print_exc()
        shutil.rmtree(workdir, ignore_errors=True)
        raise gr.Error(f"Generation failed: {exc}") from exc

    audio_path = workdir / "audio.flac"
    abc_path = workdir / "score.abc"
    abc_out = abc_path.read_text(encoding="utf-8") if abc_path.exists() else "(no symbolic score for cot=off)"
    info = (
        f"mode={receipt['mode']}  seconds={receipt['audio_seconds']:.1f}  "
        f"elapsed={elapsed:.1f}s  identity={receipt['identity'][:16]}...  "
        f"truncated={receipt['truncated']}"
    )
    return str(audio_path), abc_out, info


def build_app(model, vae, device, memory_budget_gib, offline):
    with gr.Blocks(title="YuE2 Song Generator") as demo:
        gr.Markdown(
            "# YuE2 Song Generator\n"
            "Manual, single-song generation. Style/tags and lyrics follow the same "
            "conventions as the yue2-music skill's `prompt.json` (style, lyrics, cot, seed)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                style = gr.Textbox(
                    label="Style / tags",
                    placeholder="e.g. warm piano pop, expressive female voice, 88 BPM",
                    lines=2,
                )
                lyrics = gr.Textbox(
                    label="Lyrics ([verse]/[chorus] tags recommended)",
                    placeholder="[verse]\n...\n\n[chorus]\n...",
                    lines=12,
                )
                cot = gr.Radio(COT_CHOICES, value="full", label="Planning mode (cot)")
                abc_text = gr.Textbox(
                    label="Optional ABC input (melody/full only, chord-free for melody)",
                    lines=6,
                )
                with gr.Row():
                    seed = gr.Textbox(label="Seed (optional)", placeholder="831001")
                    cfg_scale = gr.Textbox(label="CFG scale (optional)", placeholder="")
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                audio_out = gr.Audio(label="Generated song", type="filepath")
                info_out = gr.Textbox(label="Result info", interactive=False)
                abc_out = gr.Textbox(label="Generated score (ABC)", lines=16, interactive=False)

        generate_btn.click(
            fn=lambda *a: generate(*a, model, vae, device, memory_budget_gib, offline),
            inputs=[style, lyrics, cot, seed, cfg_scale, abc_text],
            outputs=[audio_out, abc_out, info_out],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vae", default=DEFAULT_VAE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-budget-gib", type=float, default=24)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live tunnel link")
    args = parser.parse_args()

    demo = build_app(args.model, args.vae, args.device, args.memory_budget_gib, args.offline)
    demo.queue(max_size=20)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

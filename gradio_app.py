#!/usr/bin/env python3
"""Gradio UI for manual, one-off YuE2 song generation.

Loads the YuE2Pipeline once at startup and reuses it across requests,
unlike the CLI (skills/yue2-music/scripts/run_yue2.py) which reloads
the model per invocation. Intended to run on a GPU pod that already
has yue2-infer installed (see references/models-and-setup.md).
"""

import argparse
import random
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


def build_request(style, lyrics, cot, seed, cfg_scale, abc_text):
    from yue2.protocol import SongRequest

    if not style or not style.strip():
        raise gr.Error("Style/tags is required.")
    if cot != "off" and (not lyrics or not lyrics.strip()):
        raise gr.Error("Lyrics are required for full/melody modes.")

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
        return SongRequest(**req_kwargs)
    except Exception as exc:
        raise gr.Error(f"Invalid request: {exc}") from exc


def run_one(pipe, request, cot):
    workdir = Path(tempfile.mkdtemp(prefix="yue2_gradio_"))
    try:
        start = time.perf_counter()
        song = pipe(**request.to_dict())
        elapsed = time.perf_counter() - start
        receipt = song.save_artifacts(workdir)
    except Exception as exc:
        traceback.print_exc()
        shutil.rmtree(workdir, ignore_errors=True)
        raise gr.Error(f"Generation failed: {exc}") from exc

    audio_path = workdir / "audio.flac"
    abc_path = workdir / "score.abc"
    abc_out = abc_path.read_text(encoding="utf-8") if abc_path.exists() else "(no symbolic score for cot=off)"
    info = (
        f"mode={cot}  seed={request.seed}  seconds={receipt['audio_seconds']:.1f}  "
        f"elapsed={elapsed:.1f}s  identity={receipt['identity'][:16]}...  "
        f"truncated={receipt['truncated']}"
    )
    return str(audio_path), abc_out, info


def generate(style, lyrics, cot, seed, cfg_scale, abc_text,
             model, vae, device, memory_budget_gib, offline, progress=gr.Progress()):
    progress(0.05, desc="Loading model (cached after first run)...")
    try:
        pipe = load_pipeline(model, vae, device, memory_budget_gib, offline)
    except Exception as exc:
        raise gr.Error(f"Failed to load model: {exc}") from exc

    request = build_request(style, lyrics, cot, seed, cfg_scale, abc_text)
    progress(0.15, desc="Generating (planning, synthesis, decoding)...")
    result = run_one(pipe, request, cot)
    progress(1.0, desc="Done")
    return result


def generate_two(style, lyrics, cot, seed, cfg_scale, abc_text,
                  model, vae, device, memory_budget_gib, offline, progress=gr.Progress()):
    progress(0.03, desc="Loading model (cached after first run)...")
    try:
        pipe = load_pipeline(model, vae, device, memory_budget_gib, offline)
    except Exception as exc:
        raise gr.Error(f"Failed to load model: {exc}") from exc

    base_seed = int(seed) if seed is not None and seed != "" else None
    seed_a = base_seed
    seed_b = (base_seed + 1) if base_seed is not None else None

    progress(0.1, desc="Generating take 1 of 2...")
    request_a = build_request(style, lyrics, cot, seed_a, cfg_scale, abc_text)
    result_a = run_one(pipe, request_a, cot)

    progress(0.55, desc="Generating take 2 of 2...")
    request_b = build_request(style, lyrics, cot, seed_b, cfg_scale, abc_text)
    result_b = run_one(pipe, request_b, cot)

    progress(1.0, desc="Done")
    return (*result_a, *result_b)


STYLE_PRESETS = [
    ("Warm piano pop", "warm piano pop, expressive female voice, 88 BPM"),
    ("Indie folk", "indie folk, acoustic guitar, soft male vocal, intimate, 96 BPM"),
    ("Synthwave", "synthwave, analog synths, driving beat, dreamy vocal, 110 BPM"),
    ("Cinematic epic", "cinematic orchestral, strings and choir, epic build, 84 BPM"),
    ("Lo-fi chill", "lo-fi hip hop, mellow keys, vinyl crackle, laid-back vocal, 78 BPM"),
]

LYRICS_TEMPLATE = "[verse]\n\n\n[chorus]\n\n\n[verse]\n\n\n[chorus]\n"

CSS = """
.gradio-container, .gradio-container main.fillable { max-width: 1280px !important; margin: auto; }
#hero { padding: 28px 32px; border-radius: 20px; margin-bottom: 8px;
  background: linear-gradient(120deg, #4c1d95 0%, #7c3aed 45%, #c026d3 100%); color: #fff; }
#hero h1 { margin: 0 0 6px; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; color: #fff; }
#hero p { margin: 0; opacity: .85; font-size: 1rem; color: #fff; }
.card { border: 1px solid var(--border-color-primary); border-radius: 16px !important;
  padding: 18px !important; background: var(--background-fill-secondary); }
.chips { flex-wrap: wrap !important; gap: 8px !important; }
.chips button { flex: 0 0 auto !important; white-space: nowrap !important; border-radius: 999px !important;
  font-size: .82rem !important; padding: 4px 14px !important; min-width: 0 !important; width: auto !important; }
#generate-btn { background: linear-gradient(90deg, #7c3aed, #c026d3) !important; border: 0 !important;
  color: #fff !important; font-weight: 600; box-shadow: 0 6px 20px rgba(124, 58, 237, .35); }
#generate-btn:hover { filter: brightness(1.08); }
footer { display: none !important; }
"""


def _setter(value):
    return lambda: value


def _preset_chips(style):
    with gr.Row(elem_classes="chips"):
        for label, tags in STYLE_PRESETS:
            gr.Button(label, size="sm", variant="secondary").click(
                _setter(tags), outputs=style, show_progress="hidden")


def _result_panel(title):
    with gr.Tab(title):
        audio = gr.Audio(label="Song", type="filepath", interactive=False)
        info = gr.Textbox(label="Run info", interactive=False, lines=2)
        with gr.Accordion("Generated score (ABC)", open=False):
            abc = gr.Textbox(show_label=False, lines=12, interactive=False, show_copy_button=True)
    return audio, abc, info


def build_app(model, vae, device, memory_budget_gib, offline):
    theme = gr.themes.Soft(
        primary_hue="violet", secondary_hue="fuchsia", neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )
    with gr.Blocks(title="YuE2 Song Generator", theme=theme, css=CSS) as demo:
        gr.HTML(
            '<div id="hero"><h1>YuE2 Song Generator</h1>'
            "<p>Describe a style, drop in lyrics, get a full song with vocals. "
            "Compare two takes side by side.</p></div>"
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="card"):
                style = gr.Textbox(label="Style / tags", lines=2,
                                   placeholder="genre, instruments, voice, mood, BPM")
                _preset_chips(style)
                with gr.Row():
                    gr.Markdown("**Lyrics**")
                    template_btn = gr.Button("Insert structure", size="sm", variant="secondary")
                lyrics = gr.Textbox(show_label=False, lines=14, show_copy_button=True,
                                    placeholder="[verse]\n...\n\n[chorus]\n...")
                cot = gr.Radio(COT_CHOICES, value="full", label="Planning mode (cot)",
                               info="full: plan lyrics + melody  |  melody: melody only  |  off: instrumental, no planning")
                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=None, precision=0,
                                         info="Take 2 uses seed + 1")
                        dice_btn = gr.Button("Random seed", size="sm", variant="secondary")
                        cfg_scale = gr.Number(label="CFG scale", value=None)
                    abc_text = gr.Textbox(label="ABC input (melody/full only, chord-free for melody)", lines=6)
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary", elem_id="generate-btn", scale=2)
                    generate_two_btn = gr.Button("Generate 2 takes", scale=1)
            with gr.Column(scale=6, elem_classes="card"):
                with gr.Tabs():
                    audio_out, abc_out, info_out = _result_panel("Take 1")
                    audio_out_b, abc_out_b, info_out_b = _result_panel("Take 2")

        template_btn.click(_setter(LYRICS_TEMPLATE), outputs=lyrics, show_progress="hidden")
        dice_btn.click(lambda: random.randint(0, 999999), outputs=seed, show_progress="hidden")
        inputs = [style, lyrics, cot, seed, cfg_scale, abc_text]
        generate_btn.click(
            fn=lambda *a: generate(*a, model, vae, device, memory_budget_gib, offline),
            inputs=inputs, outputs=[audio_out, abc_out, info_out], concurrency_limit=1,
        )
        generate_two_btn.click(
            fn=lambda *a: generate_two(*a, model, vae, device, memory_budget_gib, offline),
            inputs=inputs, outputs=[audio_out, abc_out, info_out, audio_out_b, abc_out_b, info_out_b],
            concurrency_limit=1,
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

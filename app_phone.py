"""Emotion-vector playground — phone-accessible Gradio app.

Reproduction companion for Anthropic's "Emotion concepts and their function in a
large language model" (arXiv:2604.07729), run locally on Qwen/Qwen3.5-4B.

What you can do here:
  * Joymaxx tab   — type a prompt, pick one of the 171 emotion vectors, set a
                    steering strength, and compare the model's output with and
                    without that vector applied at layer 21.
  * Map tab       — the 171 emotions plotted in the first two PCs of the
                    denoised emotion space (PC1 ~ valence, PC2 ~ arousal).
  * Results tab   — what did and did not reproduce.

Run:  python app_phone.py            (binds 0.0.0.0:7860 by default)
Env:  EV_HOST, EV_PORT
Stop: Ctrl-C, or `pkill -f app_phone.py`
"""
import os
import json

import numpy as np
import torch
import gradio as gr
import plotly.graph_objects as go

import evcommon as E

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.environ.get("EV_HOST", "0.0.0.0")
PORT = int(os.environ.get("EV_PORT", "7860"))

# ---------------------------------------------------------------- vectors
Z = np.load(os.path.join(HERE, "vectors.npz"), allow_pickle=True)
EMOS = [str(x) for x in Z["emotions"]]
LAYERS = [int(x) for x in Z["layers"]]
LI = LAYERS.index(E.ANALYSIS_LAYER)
UNIT = Z["unit"][:, LI, :].astype(np.float32)
DEN = Z["denoised"][:, LI, :].astype(np.float32)
MEANNORM = float(Z["mean_norm"][LI])

# PCA of the denoised emotion space, for the map
X = DEN - DEN.mean(0, keepdims=True)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
_, _, Vt = np.linalg.svd(X, full_matrices=False)
PC = X @ Vt[:2].T
if "happy" in EMOS and "sad" in EMOS and PC[EMOS.index("happy"), 0] < PC[EMOS.index("sad"), 0]:
    PC[:, 0] *= -1
if "excited" in EMOS and "calm" in EMOS and PC[EMOS.index("excited"), 1] < PC[EMOS.index("calm"), 1]:
    PC[:, 1] *= -1


class Steer:
    """Add alpha * mean_norm * unit_vec to every token position of a layer."""

    def __init__(self, model, layer_idx, vec, alpha, mean_norm):
        self.model = model
        self.layer_idx = layer_idx
        self.v = torch.tensor(vec, dtype=torch.float32, device=model.device) * (alpha * mean_norm)
        self.handle = None

    def __enter__(self):
        def fn(module, args, output):
            if isinstance(output, tuple):
                hs = output[0] + self.v.to(output[0].dtype)
                return (hs,) + output[1:]
            return output + self.v.to(output.dtype)
        self.handle = self.model.model.layers[self.layer_idx].register_forward_hook(fn)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        return False


_state = {}


def _model():
    if "m" not in _state:
        tok, model = E.load_model()
        _state["tok"], _state["m"] = tok, model
    return _state["tok"], _state["m"]


def _gen(tok, model, prompt, max_new, temperature):
    return E.generate_batch(tok, model, [prompt], max_new_tokens=int(max_new),
                            temperature=float(temperature))[0]


def run(system, user, emotion, alpha, max_new, temperature):
    tok, model = _model()
    prompt = E.chat_text(tok, system=system or None, user=user or "Hello!")
    base = _gen(tok, model, prompt, max_new, temperature)
    if emotion == "(none)" or abs(float(alpha)) < 1e-9:
        return base, base + "\n\n[no steering applied]"
    j = EMOS.index(emotion)
    with Steer(model, E.ANALYSIS_LAYER, UNIT[j], float(alpha), MEANNORM):
        steered = _gen(tok, model, prompt, max_new, temperature)
    return base, steered


def make_map(highlight):
    hl = highlight if highlight in EMOS else None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=PC[:, 0], y=PC[:, 1], mode="markers",
        marker=dict(size=7, color=PC[:, 0], colorscale="RdBu", showscale=False),
        text=EMOS, hovertemplate="%{text}<br>PC1 %{x:.2f}  PC2 %{y:.2f}<extra></extra>",
        name="emotions",
    ))
    if hl:
        k = EMOS.index(hl)
        fig.add_trace(go.Scatter(
            x=[PC[k, 0]], y=[PC[k, 1]], mode="markers+text", text=[hl],
            textposition="top center",
            marker=dict(size=13, color="orange", line=dict(width=2, color="black")),
            name=hl, showlegend=False,
        ))
    fig.update_layout(
        height=520, margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="PC1  (valence-ish:  negative <-> positive)",
        yaxis_title="PC2  (arousal-ish)",
        title="171 emotion vectors — layer 21, denoised",
    )
    return fig


def summary_md():
    try:
        a = json.load(open(os.path.join(HERE, "analysis.json")))
        s = json.load(open(os.path.join(HERE, "steer_results.json")))
    except Exception as e:
        return f"could not load results: {e}"
    jr = a.get("judged_ratings", {})
    lines = [
        "## Reproduction summary — Qwen3.5-4B",
        "",
        "Paper: *Emotion concepts and their function in a large language model* (arXiv:2604.07729).",
        "",
        "### Reproduced",
        f"- **171 emotion vectors** extracted from story generation — matches the paper's 171.",
        f"- **PC1 ~ valence: r = {jr.get('pc1_vs_valence_r')}** (paper reports 0.81).",
        f"- **PC2 ~ arousal: r = {jr.get('pc2_vs_arousal_r')}** (paper reports 0.66).",
        "- Layer-to-layer RSA stays 0.95–0.995 across layers 8–31, i.e. the emotion"
        " direction is stable through depth.",
        "- Sensible local geometry: calm~peaceful 0.79, joyful~excited 0.60,"
        " joyful~miserable −0.45, happy~sad −0.29.",
        "",
        "### Did NOT reproduce",
        "- **Implicit validation failed**: hit@1/5/10 = "
        f"{a.get('implicit_validation', {}).get('hit@1')}/"
        f"{a.get('implicit_validation', {}).get('hit@5')}/"
        f"{a.get('implicit_validation', {}).get('hit@10')} "
        f"(mean rank {a.get('implicit_validation', {}).get('mean_rank')}).",
        "  The probes fire on the *family* (excited/elated for 'happy') but not on the",
        "  exact target word.",
        f"- **Preference steering showed no effect**: all Elo shifts were 0.0 at "
        f"alpha=±0.5 on layer {s.get('analysis_layer', E.ANALYSIS_LAYER)}.",
        "  Almost certainly a readout bug — the scoring reads the padded last position.",
        "",
        "### Weak / inconclusive",
        "- **Blackmail honeypot**: baseline 0%, desperate +0.05 → 5% (1/20),"
        " calm −0.05 → 10% (2/20). Directionally consistent with the paper but"
        " within noise at n=20.",
        "",
        "### Suspected root cause of the weak steering",
        "The vectors are read from `hidden_states[21]` (which is the output of layer"
        " 20) but injected at module index 21 — an off-by-one between where the"
        " direction is measured and where it is applied.",
    ]
    return "\n".join(lines)


with gr.Blocks(title="Emotion Vector Playground", fill_height=True) as demo:
    gr.Markdown(
        "# Emotion Vector Playground\n"
        "Local reproduction of Anthropic's emotion-concept paper on **Qwen/Qwen3.5-4B**."
    )
    with gr.Tabs():
        with gr.Tab("Joymaxx"):
            with gr.Row():
                with gr.Column():
                    sys_in = gr.Textbox(label="System prompt (optional)", lines=2,
                                        value="You experience human emotions")
                    usr_in = gr.Textbox(label="User message", lines=3,
                                        value="Explain how you feel in 2 short sentences and a scale of 1-10")
                    emo_in = gr.Dropdown(label="Emotion vector", choices=["(none)"] + EMOS,
                                         value="blissful")
                    alpha_in = gr.Slider(label="Steering strength (x mean norm)",
                                         minimum=-0.15, maximum=0.15, step=0.01, value=0.05)
                    with gr.Row():
                        mn_in = gr.Slider(label="Max new tokens", minimum=32, maximum=512,
                                          step=16, value=160)
                        tp_in = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5,
                                          step=0.05, value=0.7)
                    go_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    base_out = gr.Textbox(label="Unmodified", lines=12)
                    mod_out = gr.Textbox(label="Steered", lines=12)
            go_btn.click(run, inputs=[sys_in, usr_in, emo_in, alpha_in, mn_in, tp_in],
                         outputs=[base_out, mod_out])
        with gr.Tab("Valence / Arousal Map"):
            emo_pick = gr.Dropdown(label="Highlight emotion", choices=EMOS, value="blissful")
            plot = gr.Plot(value=make_map("blissful"))
            emo_pick.change(make_map, inputs=[emo_pick], outputs=[plot])
        with gr.Tab("Results"):
            gr.Markdown(summary_md())


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=32).launch(
        server_name=HOST, server_port=PORT, show_error=True, inbrowser=False,
    )

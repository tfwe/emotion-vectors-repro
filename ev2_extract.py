"""Stage 2: build emotion vectors from the generated stories.

Follows the paper's recipe:
  1. average residual-stream activations across token positions (>= 50) per story
  2. average those across stories for each emotion
  3. subtract the mean activation across emotions
  4. project out the top principal components of activations on *neutral*
     transcripts (enough to explain 50% of the variance)

Writes vectors.npz.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evcommon import ANALYSIS_LAYER, EMOTIONS, PROBE_LAYERS, hidden_states_for, load_model

HERE = Path(__file__).resolve().parent
OUT = HERE / "vectors.npz"


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pcs_explaining(X: np.ndarray, frac=0.50):
    """Top principal components (as row vectors) explaining >= `frac` variance."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD: rows are samples, so right singular vectors are the PCs
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (s ** 2) / max((s ** 2).sum(), 1e-12)
    k = int(np.searchsorted(np.cumsum(var), frac) + 1)
    return Vt[:k], var, k


def main():
    tok, model = load_model()
    stories = read_jsonl(HERE / "stories.jsonl")
    neutral = read_jsonl(HERE / "neutral.jsonl")
    print(f"[extract] {len(stories)} stories, {len(neutral)} neutral", flush=True)

    # ---- activations on stories ----
    texts = [s["text"] for s in stories]
    labs = [s["emotion"] for s in stories]
    acts, kept = hidden_states_for(tok, model, texts, PROBE_LAYERS, batch_size=8)
    print(f"[extract] story activations {acts.shape} (kept tokens min/median/max "
          f"{min(kept)}/{int(np.median(kept))}/{max(kept)})", flush=True)

    # ---- activations on neutral transcripts -> denoising PCs ----
    nacts, _ = hidden_states_for(tok, model, [n["text"] for n in neutral], PROBE_LAYERS,
                                 batch_size=8)
    print(f"[extract] neutral activations {nacts.shape}", flush=True)

    n_layers, hidden = acts.shape[1], acts.shape[2]
    mean_norm = np.zeros(n_layers, dtype=np.float32)
    raw = np.zeros((len(EMOTIONS), n_layers, hidden), dtype=np.float32)
    den = np.zeros_like(raw)

    for li in range(n_layers):
        # mean residual-stream norm at this layer, used for steering units
        mean_norm[li] = float(np.linalg.norm(nacts[:, li, :], axis=1).mean())

        # per-emotion mean
        for ei, e in enumerate(EMOTIONS):
            sel = [j for j, l in enumerate(labs) if l == e]
            if not sel:
                continue
            raw[ei, li] = acts[sel, li, :].mean(axis=0)
        # subtract mean across emotions
        centered = raw[:, li, :] - raw[:, li, :].mean(axis=0, keepdims=True)
        # project out neutral PCs
        Vt, var, k = pcs_explaining(nacts[:, li, :], 0.50)
        proj = centered - (centered @ Vt.T) @ Vt
        den[:, li, :] = proj
        if li == 0:
            print(f"[extract] layer{PROBE_LAYERS[li]}: {k} neutral PCs explain 50% "
                  f"(var[0]={var[0]:.3f})  mean_norm={mean_norm[li]:.1f}", flush=True)

    # unit-normalised vectors, used for cosine geometry and steering
    norms = np.linalg.norm(den, axis=2, keepdims=True)
    unit = den / np.clip(norms, 1e-8, None)

    np.savez_compressed(
        OUT,
        emotions=np.array(EMOTIONS),
        layers=np.array(PROBE_LAYERS),
        raw=raw, denoised=den, unit=unit, mean_norm=mean_norm,
        acts=acts, acts_labels=np.array(labs),
    )
    print(f"[extract] wrote {OUT}  unit vector norms per layer: "
          f"{np.round(norms.reshape(len(EMOTIONS), n_layers).mean(axis=0), 1).tolist()}", flush=True)


if __name__ == "__main__":
    main()
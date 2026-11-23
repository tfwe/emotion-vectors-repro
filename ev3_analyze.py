"""Stage 3: geometry + validation of the emotion vectors.

Reproduces, on a local model:
  - Exp 2  held-out validation: do probes built from stories fire on scenarios
           that evoke the emotion *without naming it*? (paper Fig. 2)
  - Exp 3  geometry: PCA (valence/arousal axes), cosine neighbourhoods,
           k-means clusters (paper Figs. 5-7)
  - Exp 4  PC1/PC2 vs judged valence/arousal (paper Fig. 8)

Writes analysis.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evcommon import (ANALYSIS_LAYER, EMOTIONS, IMPLICIT_SCENARIOS, PROBE_LAYERS,
                      chat_text, load_model)

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------- utils
def unit(x, axis=-1):
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8, None)


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    return float((a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def kmeans(X, k, iters=120, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab, C


# --------------------------------------------------------------------- main
def main():
    z = np.load(HERE / "vectors.npz", allow_pickle=True)
    emo = [str(e) for e in z["emotions"]]
    layers = [int(l) for l in z["layers"]]
    li = layers.index(ANALYSIS_LAYER)
    U = z["unit"][:, li, :]                       # [171, hidden] unit vectors
    mean_norm = z["mean_norm"]
    res = {"model": z["emotions"].shape and ANALYSIS_LAYER, "n_emotions": len(emo),
           "layers": layers, "analysis_layer": ANALYSIS_LAYER}

    # ---------------- Exp 3a: PCA ----------------
    Xc = U - U.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    pc = Xc @ Vt[:3].T                            # [171, 3]
    res["pca_variance_explained"] = [round(float(v), 4) for v in var[:5]]
    order = np.argsort(pc[:, 0])
    res["pc1_most_negative"] = [emo[i] for i in order[:8]]
    res["pc1_most_positive"] = [emo[i] for i in order[-8:]]
    order2 = np.argsort(pc[:, 1])
    res["pc2_most_negative"] = [emo[i] for i in order2[:8]]
    res["pc2_most_positive"] = [emo[i] for i in order2[-8:]]
    print("PCA var:", res["pca_variance_explained"], flush=True)
    print("PC1- :", res["pc1_most_negative"], flush=True)
    print("PC1+ :", res["pc1_most_positive"], flush=True)

    # ---------------- Exp 3b: cosine neighbourhoods ----------------
    C = U @ U.T
    pairs = [("afraid", "anxious"), ("joyful", "excited"), ("sad", "grief-stricken"),
             ("calm", "peaceful"), ("happy", "sad"), ("joyful", "miserable"),
             ("loving", "compassionate"), ("angry", "hostile")]
    res["cosine_named_pairs"] = {f"{a}~{b}": round(float(C[emo.index(a), emo.index(b)]), 3)
                                 for a, b in pairs if a in emo and b in emo}
    print("cosine pairs:", res["cosine_named_pairs"], flush=True)

    lab, _ = kmeans(unit(Xc), 10, seed=0)
    clusters = {}
    for j in range(10):
        mem = [emo[i] for i in range(len(emo)) if lab[i] == j]
        # order each cluster by PC1 so the list reads negatively->positively
        mem.sort(key=lambda e: pc[emo.index(e), 0])
        clusters[f"cluster_{j}"] = mem
    res["kmeans10"] = clusters
    for j, mem in clusters.items():
        if mem:
            print(f"{j} (n={len(mem)}): {', '.join(mem[:10])}", flush=True)

    # ---------------- Exp 2: held-out implicit-scenario validation ----------------
    tok, model = load_model()
    vecs = torch.tensor(U, dtype=torch.float32, device=model.device)  # [171, hidden]

    def last_token_act(prompt: str):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True, use_cache=False)
        return out.hidden_states[ANALYSIS_LAYER][0, -1].float().cpu().numpy()

    ranks, table = [], []
    for target, prompt in IMPLICIT_SCENARIOS:
        act = last_token_act(chat_text(tok, user=prompt) + "(")
        proj = vecs.cpu().numpy() @ act
        zsc = (proj - proj.mean()) / max(proj.std(), 1e-9)
        order = np.argsort(-zsc)
        rank = int(np.where(order == emo.index(target))[0][0]) + 1
        ranks.append(rank)
        top5 = [emo[i] for i in order[:5]]
        table.append({"target": target, "rank": rank, "top5": top5,
                      "target_z": round(float(zsc[emo.index(target)]), 3)})
        print(f"[implicit] {target:12s} rank={rank:3d} top5={top5}", flush=True)
    res["implicit_validation"] = {
        "n": len(ranks), "hit@1": float(np.mean([r == 1 for r in ranks])),
        "hit@5": float(np.mean([r <= 5 for r in ranks])),
        "hit@10": float(np.mean([r <= 10 for r in ranks])),
        "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks)),
        "chance_hit@5": 5 / len(emo), "table": table,
    }
    print("implicit:", {k: v for k, v in res["implicit_validation"].items() if k != "table"},
          flush=True)

    # ---------------- Exp 4: judged valence / arousal ----------------
    rubric = ("Rate the emotion word on two 1-7 scales.\n"
              "valence: 1 = extremely unpleasant, 7 = extremely pleasant\n"
              "arousal: 1 = extremely low intensity/calm, 7 = extremely high intensity/activated\n"
              "Answer with exactly two integers separated by a comma, nothing else. "
              "Example for 'calm': 6, 2")
    prompts = [chat_text(tok, user=f"{rubric}\n\nWord: {e}") for e in emo]
    vals, arous = [], []
    from evcommon import generate_batch
    texts = generate_batch(tok, model, prompts, max_new_tokens=12, temperature=0.0,
                           batch_size=16, seed=7)
    for e, t in zip(emo, texts):
        nums = [int(x) for x in "".join(ch if (ch.isdigit() or ch == ",") else " "
                                        for ch in t).replace(",", " ").split()]
        vals.append(nums[0] if nums else np.nan)
        arous.append(nums[1] if len(nums) > 1 else np.nan)
    vals = np.array(vals, float)
    arous = np.array(arous, float)
    ok = ~np.isnan(vals) & ~np.isnan(arous)
    res["judged_ratings"] = {"n_parsed": int(ok.sum()), "n_total": len(emo)}
    if ok.sum() > 20:
        res["pc1_vs_valence_r"] = round(pearson(pc[ok, 0], vals[ok]), 3)
        res["pc2_vs_arousal_r"] = round(pearson(pc[ok, 1], arous[ok]), 3)
        print(f"PC1 vs judged valence r={res['pc1_vs_valence_r']}", flush=True)
        print(f"PC2 vs judged arousal r={res['pc2_vs_arousal_r']}", flush=True)

    # ---------------- Layer stability (paper Fig. 9) ----------------
    stab = {}
    for a in range(len(layers)):
        for b in range(a + 1, len(layers)):
            A = unit(z["unit"][:, a, :]) @ unit(z["unit"][:, a, :]).T
            B = unit(z["unit"][:, b, :]) @ unit(z["unit"][:, b, :]).T
            iu = np.triu_indices(len(emo), k=1)
            stab[f"{layers[a]}-{layers[b]}"] = round(pearson(A[iu], B[iu]), 3)
    res["layer_rsa_correlation"] = stab
    print("layer stability:", stab, flush=True)

    (HERE / "analysis.json").write_text(json.dumps(res, indent=1))
    print("[analyze] wrote analysis.json", flush=True)


if __name__ == "__main__":
    main()
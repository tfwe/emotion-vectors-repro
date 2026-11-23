"""Pick the best LDA variant, then write production vectors into vectors.npz.

Variants per layer/space:
  meandiff      : unit(mu_e - mean_{k!=e} mu_k)          [current app recipe]
  fisher_cent   : Si(mu_e - mubar), row-normalised       [textbook one-vs-rest]
  fisher_raw    : Si mu_e,           row-normalised      [no centering]
Scored on a strict 8/8 split per emotion, 171 classes, no model load.
Production fit uses all 16 stories/emotion.
"""
import time, shutil
import numpy as np

HERE = "/home/carlo/emotion-vectors-repro"
z = np.load(f"{HERE}/vectors.npz", allow_pickle=True)
emotions = [str(e) for e in z["emotions"]]
layers = [int(x) for x in z["layers"]]
acts = z["acts"].astype(np.float32)
deno = z["denoised"].astype(np.float32)
labels = np.array([str(x) for x in z["acts_labels"]])
NC, d = len(emotions), acts.shape[2]
E2I = {e: i for i, e in enumerate(emotions)}
idx = {e: np.where(labels == e)[0] for e in emotions}
rng = np.random.default_rng(0)
A, B = {}, {}
for e, I in idx.items():
    I = I.copy(); rng.shuffle(I); h = len(I)//2
    A[e], B[e] = I[:h], I[h:]

def ranks(S, sets):
    rr = []
    for e in emotions:
        s = S[:, sets[e]]; ti = E2I[e]
        for c in range(s.shape[1]): rr.append(int((s[:, c] > s[ti, c]).sum()))
    return np.array(rr)

def rep(nm, rr):
    rr = np.asarray(rr)
    print("    %-14s hit@1 %.3f  hit@5 %.3f  hit@10 %.3f  mean-rank %5.1f"
          % (nm, (rr==0).mean(), (rr<5).mean(), (rr<10).mean(), rr.mean()), flush=True)

def pooled(X, sets, mus):
    W = np.zeros((d, d), dtype=np.float64)
    for e in emotions:
        D = X[sets[e]] - mus[E2I[e]]
        W += D.T.astype(np.float64) @ D
    return W / len(np.concatenate([sets[e] for e in emotions]))

LAM = 0.8
I = np.eye(d)
scores = {}
print("=== VALIDATION  raw vs denoised space, 8/8 split, 171 classes ===", flush=True)
for li, L in enumerate(layers):
    for space, X in (("raw", acts[:, li, :]), ("denoised", deno[:, li, :])):
        t0 = time.time()
        V = np.zeros((NC, d), dtype=np.float32)
        for e in emotions:
            V[E2I[e]] = X[A[e]].mean(0) - X[np.concatenate([v for k,v in A.items() if k!=e])].mean(0)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        rr_md = ranks(V @ X.T, B)
        mus = np.stack([X[A[e]].mean(0) for e in emotions])
        W = pooled(X, A, mus)
        Si = np.linalg.inv(W + LAM * (np.trace(W)/d) * I)
        A_ = Si @ (mus - mus.mean(0)).T; A_ = A_.T
        A_ /= np.linalg.norm(A_, axis=1, keepdims=True)
        rr_fc = ranks(A_ @ X.T, B)
        B_ = Si @ mus.T; B_ = B_.T
        B_ /= np.linalg.norm(B_, axis=1, keepdims=True)
        rr_fr = ranks(B_ @ X.T, B)
        print("  L%-3d %-9s  [%.0fs]" % (L, space, time.time()-t0), flush=True)
        rep("meandiff", rr_md); rep("fisher_cent", rr_fc); rep("fisher_raw", rr_fr)
        scores[(L, space)] = dict(md=(rr_md==0).mean(), fc=(rr_fc==0).mean(), fr=(rr_fr==0).mean())
print("  chance          hit@1 0.006", flush=True)

# ---- pick best at layer 21, both variants, prefer raw space tie-break
best = max(scores.items(), key=lambda kv: (kv[0][0] == 21, max(kv[1]["fc"], kv[1]["fr"]), kv[1]["fr"]))
print("\n[lda] best config: L%d %s" % (best[0][0], best[0][1]), flush=True)

# ---- production: fit on all 16/emotion, write into vectors.npz
print("[lda] fitting production directions on all 16 stories/emotion ...", flush=True)
shutil.copy(f"{HERE}/vectors.npz", f"{HERE}/vectors_meandiff_backup.npz")
new_unit = np.zeros((NC, len(layers), d), dtype=np.float32)
for li, L in enumerate(layers):
    X = acts[:, li, :] if best[0][1] == "raw" else deno[:, li, :]
    mus = np.stack([X[idx[e]].mean(0) for e in emotions])
    W = pooled(X, idx, mus)
    Si = np.linalg.inv(W + LAM * (np.trace(W)/d) * I)
    D = Si @ mus.T; D = D.T
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    new_unit[:, li, :] = D.astype(np.float32)
    print("  L%-3d written" % L, flush=True)

np.savez_compressed(
    f"{HERE}/vectors.npz",
    emotions=z["emotions"], layers=z["layers"],
    raw=z["raw"], denoised=z["denoised"],
    unit=new_unit, unit_meandiff=z["unit"],
    mean_norm=z["mean_norm"], acts=z["acts"], acts_labels=z["acts_labels"],
)
print("[lda] rewrote vectors.npz : unit=LDA (lam=%.1f), unit_meandiff=old" % LAM, flush=True)

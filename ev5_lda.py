"""Stage 5: shrinkage-LDA emotion directions, drop-in replacement for the
mean-difference 'unit' vectors used by app_phone.py.

Rationale (see session): one-vs-rest mean difference leaves 3-4x on the table
vs a discriminative direction. Shrinkage LDA with a *shared* pooled covariance
is ranking-identical to multiclass LDA, so one inverse per layer suffices.

This script:
  1. validates mean-diff vs shrinkage-LDA on the stored story activations with a
     proper 8 train / 8 held-out split per emotion (no model load needed), and
  2. writes production LDA directions (fit on all 16 stories/emotion) to
     vectors_lda.npz next to vectors.npz, WITHOUT touching vectors.npz.
"""
import time
import numpy as np

HERE = "/home/carlo/emotion-vectors-repro"
LAM = 0.8  # best shrinkage at layer 21 in validation

z = np.load(f"{HERE}/vectors.npz", allow_pickle=True)
emotions = [str(e) for e in z["emotions"]]
layers = [int(x) for x in z["layers"]]
acts = z["acts"].astype(np.float32)
labels = np.array([str(x) for x in z["acts_labels"]])
NC, d = len(emotions), acts.shape[2]
E2I = {e: i for i, e in enumerate(emotions)}
idx = {e: np.where(labels == e)[0] for e in emotions}
print("[lda] acts %s   %d emotions x %d stories" % (acts.shape, NC, len(idx[emotions[0]])), flush=True)

rng = np.random.default_rng(0)
A, B = {}, {}
for e, I in idx.items():
    I = I.copy(); rng.shuffle(I); h = len(I) // 2
    A[e], B[e] = I[:h], I[h:]


def ranks(S, sets):
    rr = []
    for e in emotions:
        s = S[:, sets[e]]; ti = E2I[e]
        for c in range(s.shape[1]):
            rr.append(int((s[:, c] > s[ti, c]).sum()))
    return np.array(rr)


def rep(name, rr):
    rr = np.asarray(rr)
    print("  %-22s hit@1 %.3f  hit@5 %.3f  hit@10 %.3f  mean-rank %5.1f"
          % (name, (rr == 0).mean(), (rr < 5).mean(), (rr < 10).mean(), rr.mean()), flush=True)


def pooled_within(X, sets, mus):
    W = np.zeros((d, d), dtype=np.float64)
    for e in emotions:
        D = X[sets[e]] - mus[E2I[e]]
        W += D.T.astype(np.float64) @ D
    return W / X.shape[0]


print("=== validation: 8 train / 8 held-out per emotion, 171 classes ===", flush=True)
lda_den = np.zeros((NC, len(layers), d), dtype=np.float32)
for li, L in enumerate(layers):
    t0 = time.time()
    X = acts[:, li, :]
    # ---- baseline: train-only one-vs-rest mean difference (the current recipe)
    V = np.zeros((NC, d), dtype=np.float32)
    for e in emotions:
        V[E2I[e]] = X[A[e]].mean(0) - X[np.concatenate([v for k, v in A.items() if k != e])].mean(0)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    rep("unit(mean-diff)", ranks(V @ X.T, B))
    # ---- train-only shrinkage LDA
    mus = np.stack([X[A[e]].mean(0) for e in emotions])
    W = pooled_within(X, A, mus)
    tr = np.trace(W) / d; I = np.eye(d)
    for lam in (0.2, 0.5, 0.8):
        Si = np.linalg.inv(W + lam * tr * I)
        Wd = (Si @ (mus - mus.mean(0)).T).T
        Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
        rep("LDA lam=%.1f" % lam, ranks(Wd @ X.T, B))
    # ---- production directions: fit on all 16 stories/emotion
    mus_all = np.stack([X[idx[e]].mean(0) for e in emotions])
    Wa = pooled_within(X, idx, mus_all)
    tra = np.trace(Wa) / d
    Si = np.linalg.inv(Wa + LAM * tra * I)
    lda_den[:, li, :] = (Si @ (mus_all - mus_all.mean(0)).T).T.astype(np.float32)
    print("  [L%d  %.1fs]" % (L, time.time() - t0), flush=True)
print("  chance                 hit@1 0.006  hit@5 0.029  hit@10 0.058  mean-rank 85.5", flush=True)

norms = np.linalg.norm(lda_den, axis=2, keepdims=True)
lda_unit = lda_den / np.clip(norms, 1e-8, None)
np.savez_compressed(
    f"{HERE}/vectors_lda.npz",
    emotions=np.array(emotions), layers=np.array(layers),
    lda_den=lda_den, lda_unit=lda_unit.astype(np.float32),
    mean_norm=z["mean_norm"], lam=np.float32(LAM),
)
print("[lda] wrote %s/vectors_lda.npz  lda_unit %s" % (HERE, lda_unit.shape), flush=True)

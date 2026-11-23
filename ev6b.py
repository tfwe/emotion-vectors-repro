"""Validation in raw activation space (denoised is per-emotion, cannot be split).
Picks the best LDA variant at layer 21, then writes production vectors into vectors.npz."""
import time, shutil
import numpy as np

HERE = "/home/carlo/emotion-vectors-repro"
z = np.load(f"{HERE}/vectors.npz", allow_pickle=True)
emotions = [str(e) for e in z["emotions"]]
layers = [int(x) for x in z["layers"]]
acts = z["acts"].astype(np.float32)
labels = np.array([str(x) for x in z["acts_labels"]])
NC, d = len(emotions), acts.shape[2]
E2I = {e: i for i, e in enumerate(emotions)}
idx = {e: np.where(labels == e)[0] for e in emotions}
rng = np.random.default_rng(0)
A, B = {}, {}
for e, I in idx.items():
    I = I.copy(); rng.shuffle(I); h = len(I)//2
    A[e], B[e] = I[:h], I[h:]
I_d = np.eye(d)
LAM = 0.8

def ranks(S, sets):
    rr = []
    for e in emotions:
        s = S[:, sets[e]]; ti = E2I[e]
        for c in range(s.shape[1]): rr.append(int((s[:, c] > s[ti, c]).sum()))
    return np.array(rr)

def rep(nm, rr):
    rr = np.asarray(rr)
    print("    %-12s hit@1 %.3f  hit@5 %.3f  hit@10 %.3f  mean-rank %5.1f"
          % (nm, (rr==0).mean(), (rr<5).mean(), (rr<10).mean(), rr.mean()), flush=True)

def pooled(X, sets):
    mus = np.stack([X[sets[e]].mean(0) for e in emotions])
    W = np.zeros((d, d), dtype=np.float64)
    for e in emotions:
        D = X[sets[e]] - mus[E2I[e]]
        W += D.T.astype(np.float64) @ D
    n = sum(len(sets[e]) for e in emotions)
    return mus, W / n

def variants(X, train):
    mus, W = pooled(X, train)
    Si = np.linalg.inv(W + LAM * (np.trace(W)/d) * I_d)
    V = np.zeros((NC, d), dtype=np.float32)
    oth = np.concatenate([train[e] for e in emotions])
    for e in emotions:
        V[E2I[e]] = mus[E2I[e]] - X[np.concatenate([train[k] for k in emotions if k != e])].mean(0)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Fc = Si @ (mus - mus.mean(0)).T; Fc = Fc.T
    Fc /= np.linalg.norm(Fc, axis=1, keepdims=True)
    Fr = Si @ mus.T; Fr = Fr.T
    Fr /= np.linalg.norm(Fr, axis=1, keepdims=True)
    return V, Fc, Fr

print("=== VALIDATION  raw space, strict 8/8 split, 171 classes ===", flush=True)
res = {}
for li, L in enumerate(layers):
    X = acts[:, li, :]; t0 = time.time()
    V, Fc, Fr = variants(X, A)
    r_md = ranks(V @ X.T, B); r_fc = ranks(Fc @ X.T, B); r_fr = ranks(Fr @ X.T, B)
    print("  L%-3d  [%.0fs]" % (L, time.time()-t0), flush=True)
    rep("meandiff", r_md); rep("fisher_cent", r_fc); rep("fisher_raw", r_fr)
    res[L] = dict(md=(r_md==0).mean(), fc=(r_fc==0).mean(), fr=(r_fr==0).mean())
print("  chance      hit@1 0.006", flush=True)

print("\n=== ROBUSTNESS: swap the split (train on B, test on A), layer 21 ===", flush=True)
X = acts[:, layers.index(21), :]
V, Fc, Fr = variants(X, B)
rep("meandiff", ranks(V @ X.T, A)); rep("fisher_cent", ranks(Fc @ X.T, A)); rep("fisher_raw", ranks(Fr @ X.T, A))

best_L = max(res, key=lambda L: (res[L]["fr"] if L == 21 else -1))
print("\n[lda] winner: fisher_raw at L%d  hit@1 %.3f" % (best_L, res[best_L]["fr"]), flush=True)

print("[lda] fitting production directions on all 16 stories/emotion ...", flush=True)
shutil.copy(f"{HERE}/vectors.npz", f"{HERE}/vectors_meandiff_backup.npz")
new_unit = np.zeros((NC, len(layers), d), dtype=np.float32)
for li, L in enumerate(layers):
    X = acts[:, li, :]
    mus, W = pooled(X, idx)
    Si = np.linalg.inv(W + LAM * (np.trace(W)/d) * I_d)
    D = Si @ mus.T; D = D.T
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    new_unit[:, li, :] = D.astype(np.float32)
    print("  L%-3d written" % L, flush=True)
np.savez_compressed(
    f"{HERE}/vectors.npz",
    emotions=z["emotions"], layers=z["layers"], raw=z["raw"], denoised=z["denoised"],
    unit=new_unit, unit_meandiff=z["unit"], mean_norm=z["mean_norm"],
    acts=z["acts"], acts_labels=z["acts_labels"], unit_kind=np.array(["fisher_raw_lda_lam0.8"]),
)
print("[lda] rewrote vectors.npz : unit=fisher_raw LDA, unit_meandiff=old", flush=True)

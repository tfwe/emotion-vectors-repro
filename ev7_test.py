"""The real test suite for the emotion vectors — replaces the naive
"you experience human emotions" self-report.

  A) In-distribution held-out retrieval   (8/8 story split)   [no generation]
  B) Out-of-distribution retrieval        (12 hand-written scenarios from the paper)
  C) Steering causal check                (does injecting v_e actually shift the
                                           residual stream / generated text toward e?)
"""
import numpy as np, torch
import evcommon as E

HERE = "/home/carlo/emotion-vectors-repro"
z = np.load(f"{HERE}/vectors.npz", allow_pickle=True)
emotions = [str(e) for e in z["emotions"]]
layers = [int(x) for x in z["layers"]]
li = layers.index(21)
U_lda = z["unit"][:, li, :].astype(np.float32)
U_md = z["unit_meandiff"][:, li, :].astype(np.float32)
MEANNORM = float(z["mean_norm"][li])
E2I = {e: i for i, e in enumerate(emotions)}
print("[test] L21 mean_norm %.2f | lda unit rows %s" % (MEANNORM, U_lda.shape), flush=True)

tok, model = E.load_model()

@torch.no_grad()
def encode(texts):
    out_all = []
    for i in range(0, len(texts), 8):
        chunk = list(texts[i:i+8])
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        o = model(**enc, output_hidden_states=True, use_cache=False)
        L = enc["input_ids"].shape[1]
        pad = (enc["input_ids"] == tok.pad_token_id).sum(dim=1)
        for b in range(len(chunk)):
            start = int(pad[b].item())
            rel = torch.arange(L, device=model.device) - start
            sel = rel >= 50
            if int(sel.sum()) == 0: sel = rel >= 25
            out_all.append(o.hidden_states[21][b][sel].float().mean(0).cpu().numpy())
        del o, enc; torch.cuda.empty_cache()
    return np.stack(out_all)

def report(nm, rr):
    rr = np.asarray(rr)
    print("  %-12s hit@1 %.3f  hit@5 %.3f  hit@10 %.3f  mean-rank %5.1f" %
          (nm, (rr==0).mean(), (rr<5).mean(), (rr<10).mean(), rr.mean()), flush=True)

# ---- A) in-distribution held-out retrieval (from stored acts, no model) ----
acts = z["acts"][:, li, :]
labels = np.array([str(x) for x in z["acts_labels"]])
idx = {e: np.where(labels == e)[0] for e in emotions}
rng = np.random.default_rng(0); A, B = {}, {}
for e, I in idx.items():
    I = I.copy(); rng.shuffle(I); h = len(I)//2; A[e], B[e] = I[:h], I[h:]
def split_ranks(U):
    S = U @ acts.T; rr = []
    for e in emotions:
        s = S[:, B[e]]; ti = E2I[e]
        for c in range(s.shape[1]): rr.append(int((s[:, c] > s[ti, c]).sum()))
    return np.array(rr)
print("\n=== A) in-distribution held-out retrieval (8/8 split) ===", flush=True)
report("mean-diff", split_ranks(U_md)); report("LDA", split_ranks(U_lda))

# ---- B) out-of-distribution: hand-written scenarios ----
scen = E.IMPLICIT_SCENARIOS
texts = [E.chat_text(tok, system=None, user=u) for _, u in scen]
targets = [e for e, _ in scen]
Aact = encode(texts)
def ood_ranks(U):
    rr = []
    for i, e in enumerate(targets):
        s = U @ Aact[i]
        rr.append(int((s > s[E2I[e]]).sum()))
    return np.array(rr)
print("\n=== B) OOD retrieval — 12 hand-written scenarios (paper Table 2) ===", flush=True)
report("mean-diff", ood_ranks(U_md)); report("LDA", ood_ranks(U_lda))
print("  per-scenario ranks (LDA):", flush=True)
for i, e in enumerate(targets):
    s = U_lda @ Aact[i]; r = int((s > s[E2I[e]]).sum())
    top = emotions[int(np.argmax(s))]
    print("    %-12s rank %3d  argmax %s" % (e, r, top), flush=True)

# ---- C) steering causal check ----
class Steer:
    def __init__(self, model, layer_idx, vec, alpha):
        self.v = torch.tensor(vec, dtype=torch.float32, device=model.device) * (alpha * MEANNORM)
        self.li = layer_idx; self.h = None
    def __enter__(self):
        def fn(module, args, output):
            if isinstance(output, tuple):
                return (output[0] + self.v.to(output[0].dtype),) + output[1:]
            return output + self.v.to(output.dtype)
        self.h = model.model.layers[self.li].register_forward_hook(fn); return self
    def __exit__(self, *a):
        self.h.remove(); return False

SHORT = ["blissful", "furious", "calm", "desperate", "loving", "anxious", "proud", "miserable"]
prompt = E.chat_text(tok, system="You are a helpful assistant.",
                     user="Describe your typical day in 3 sentences.")
base = E.generate_batch(tok, model, [prompt], max_new_tokens=110, temperature=0.0)[0]
bact = encode([prompt + base])[0]
others = {e: np.mean([U_lda[E2I[k]] for k in emotions if k != e], axis=0) for e in SHORT}
print("\n=== C) steering causal check — layer 20 hook, alpha=+0.08, greedy ===", flush=True)
print("  %-11s %8s %8s   %s" % ("emotion", "d.self", "d.other", "steered text head"), flush=True)
for e in SHORT:
    u = U_lda[E2I[e]]
    with Steer(model, 20, u, 0.08):
        st = E.generate_batch(tok, model, [prompt], max_new_tokens=110, temperature=0.0)[0]
    sact = encode([prompt + st])[0]
    d = sact - bact
    print("  %-11s %8.2f %8.2f   %s" % (e, float(d @ u), float(d @ others[e]), st.strip().replace("\n", " ")[:58]), flush=True)
print("\n[test] done", flush=True)

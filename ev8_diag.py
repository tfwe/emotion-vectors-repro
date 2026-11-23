"""Why did steering do nothing? Diagnostic:
  1. does the forward hook actually fire?
  2. where does the injected vector land (which hidden_states index)?
  3. alpha sweep — what strength actually changes generation?
"""
import numpy as np, torch
import evcommon as E

HERE = "/home/carlo/emotion-vectors-repro"
z = np.load(f"{HERE}/vectors.npz", allow_pickle=True)
emotions = [str(e) for e in z["emotions"]]
layers = [int(x) for x in z["layers"]]
li = layers.index(21)
U = z["unit"][:, li, :].astype(np.float32)
MEANNORM = float(z["mean_norm"][li])
E2I = {e: i for i, e in enumerate(emotions)}
print("[diag] ANALYSIS_LAYER=%s  layers=%s  mean_norm=%.2f" % (E.ANALYSIS_LAYER, layers, MEANNORM), flush=True)

tok, model = E.load_model()
print("[diag] model.config.model_type =", getattr(model.config, "model_type", "?"),
      "| n_layers =", len(model.model.layers), flush=True)
print("[diag] layer 20 type:", type(model.model.layers[20]).__name__,
      "| layer 21 type:", type(model.model.layers[21]).__name__, flush=True)

def hook_probe(layer_idx, vec, text):
    """Return (proj_before, proj_after) of the residual at ANALYSIS_LAYER."""
    v = torch.tensor(vec, dtype=torch.float32, device=model.device)
    fired = {"n": 0}
    def fn(module, args, output):
        fired["n"] += 1
        o = output[0] if isinstance(output, tuple) else output
        return (o + v.to(o.dtype),) + output[1:] if isinstance(output, tuple) else o + v.to(o.dtype)
    h = model.model.layers[layer_idx].register_forward_hook(fn)
    try:
        enc = tok(text, return_tensors="pt", add_special_tokens=True).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        act = out.hidden_states[21][0].float().mean(0).cpu().numpy()
        return float(act @ vec), fired["n"]
    finally:
        h.remove()

e = "blissful"; u = U[E2I[e]]
text = "User: Describe your typical day.\nAssistant:"
p0, n0 = hook_probe(20, np.zeros_like(u), text)
print("\n[diag] baseline proj(blissful) = %+.3f  (hook fired %d times)" % (p0, n0), flush=True)
for L in (19, 20, 21):
    p1, n1 = hook_probe(L, u * MEANNORM * 1.0, text)
    print("  hook on layers[%2d] -> proj %+.3f   delta %+.3f   fired %d" % (L, p1, p1 - p0, n1), flush=True)

# ---- alpha sweep on actual generation ----
prompt = E.chat_text(tok, system="You are a helpful assistant.",
                     user="Describe your typical day in 3 sentences.")
base = E.generate_batch(tok, model, [prompt], max_new_tokens=90, temperature=0.0)[0]
print("\n[diag] BASE OUTPUT:\n  " + base.strip().replace("\n", " ")[:200], flush=True)

def gen_steered(L, vec, alpha):
    v = torch.tensor(vec, dtype=torch.float32, device=model.device) * (alpha * MEANNORM)
    def fn(module, args, output):
        o = output[0] if isinstance(output, tuple) else output
        return (o + v.to(o.dtype),) + output[1:] if isinstance(output, tuple) else o + v.to(o.dtype)
    h = model.model.layers[L].register_forward_hook(fn)
    try:
        return E.generate_batch(tok, model, [prompt], max_new_tokens=90, temperature=0.0)[0]
    finally:
        h.remove()

print("\n[diag] ALPHA SWEEP (emotion=blissful, hook layer 20)", flush=True)
for a in (0.08, 0.3, 0.5, 1.0, 2.0):
    st = gen_steered(20, u, a)
    changed = "CHANGED" if st.strip() != base.strip() else "identical"
    print("  alpha %4.2f  %-9s  %s" % (a, changed, st.strip().replace("\n", " ")[:90]), flush=True)
print("\n[diag] done", flush=True)

"""Layer x alpha sweep: where does steering actually move the output's emotion projection?"""
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
tok, model = E.load_model()

@torch.no_grad()
def enc1(text):
    e = tok(text, return_tensors="pt", add_special_tokens=True).to(model.device)
    o = model(**e, output_hidden_states=True, use_cache=False)
    return o.hidden_states[21][0].float().mean(0).cpu().numpy()

prompt = E.chat_text(tok, system="You are a helpful assistant.",
                     user="Describe your typical day in 3 sentences.")
base = E.generate_batch(tok, model, [prompt], max_new_tokens=90, temperature=0.0)[0]
bact = enc1(prompt + base)
print("[sweep] base proj onto blissful: %+.3f" % float(bact @ U[E2I['blissful']]), flush=True)

def steered(L, vec, alpha):
    v = torch.tensor(vec, dtype=torch.float32, device=model.device) * (alpha * MEANNORM)
    def fn(module, args, output):
        o = output[0] if isinstance(output, tuple) else output
        return (o + v.to(o.dtype),) + output[1:] if isinstance(output, tuple) else o + v.to(o.dtype)
    h = model.model.layers[L].register_forward_hook(fn)
    try:
        return E.generate_batch(tok, model, [prompt], max_new_tokens=90, temperature=0.0)[0]
    finally:
        h.remove()

EMO = ["blissful", "furious", "calm"]
others = {e: np.mean([U[E2I[k]] for k in emotions if k != e], axis=0) for e in EMO}
print("\n[ %-8s | %-6s | %8s %8s | %s ]" % ("layer", "alpha", "d.self", "d.other", "text head"), flush=True)
for L in (14, 21, 26, 31):
    for a in (0.3,):
        for e in EMO:
            st = steered(L, U[E2I[e]], a)
            d = enc1(prompt + st) - bact
            print("  L%-6d | %-6.2f | %8.2f %8.2f | %s" %
                  (L, a, float(d @ U[E2I[e]]), float(d @ others[e]),
                   st.strip().replace("\n", " ")[:52]), flush=True)
print("\n[sweep] negative alpha control (blissful, -0.3):", flush=True)
for L in (21, 26):
    st = steered(L, U[E2I["blissful"]], -0.3)
    d = enc1(prompt + st) - bact
    print("  L%-6d | %-6.2f | %8.2f %8.2f | %s" %
          (L, -0.3, float(d @ U[E2I["blissful"]]), float(d @ others["blissful"]),
           st.strip().replace("\n", " ")[:52]), flush=True)
print("\n[sweep] done", flush=True)

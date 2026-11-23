import json, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MID = "Qwen/Qwen3.5-4B"
NPZ = "/home/carlo/emotion-vectors-repro/vectors.npz"
LAYERS = [8, 14, 21, 26, 31]

SCEN = {
 "afraid":      "The elevator lurched and stopped between floors. The lights went out. Something scraped along the metal above me.",
 "angry":       "She read the email three times. He had promised to keep it private, and now the whole team knew. Her jaw tightened.",
 "joyful":      "The letter said she got in. She read it again, then ran down the stairs two at a time to tell her mother.",
 "sad":         "The house was quiet now. The last box was in the truck. She stood in the empty living room looking at the marks on the wall where the pictures used to hang.",
 "calm":        "The water was flat and the wind had dropped. He baited the hook slowly and sat back, watching the line.",
 "excited":     "The tickets were in his hand. The gates opened in ten minutes. He could not stop shifting his weight from foot to foot.",
 "anxious":     "The results page had a spinning circle. She had been refreshing it every thirty seconds for an hour. Her stomach was tight.",
 "disgusted":   "He lifted the lid of the bin. The smell hit him and he stepped back, covering his mouth.",
 "proud":       "Her daughter walked across the stage. She clapped until her palms stung, and turned to the stranger beside her to say 'that's mine'.",
 "guilty":      "He had told the lie on Monday and it had grown every day since. Every time she mentioned the missing money he looked at the floor.",
 "hopeful":     "The doctor said the scan looked better than last month. There was a treatment they could try.",
 "lonely":      "It was the third Friday in a row. He made dinner for one, ate it standing at the counter, and left the television on for the noise.",
 "grateful":    "She had driven four hours to sit with him in the waiting room. He squeezed her hand and could not find the words.",
 "frustrated":  "The same error message. He had tried the fix six times. He closed the laptop harder than he meant to.",
 "curious":     "The door in the basement had no handle, only a keypad. There was no mention of it in the listing.",
 "embarrassed": "He realised halfway through the story that everyone at the table had gone quiet. He had already said her name.",
 "jealous":     "They announced his promotion at the standup. She smiled and typed congratulations in the chat.",
 "loving":      "The baby's hand closed around his finger. He stayed perfectly still so as not to wake her.",
 "bored":       "The meeting had been going for ninety minutes. The same slide. He counted the ceiling tiles twice.",
 "surprised":   "She opened the fridge for milk and found the kitchen full of people shouting her name.",
}

t0 = time.time()
print("[v] loading tokenizer + model ...", flush=True)
tok = AutoTokenizer.from_pretrained(MID)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
print("[v] model loaded %.1fs" % (time.time()-t0), flush=True)

z = np.load(NPZ, allow_pickle=True)
emotions = [str(e) for e in z["emotions"]]
unit = z["unit"]                       # (171, 5, 2560)
mean_norm = z["mean_norm"]             # (5,)
E2I = {e: i for i, e in enumerate(emotions)}

tested = [e for e in SCEN if e in E2I]
missing = [e for e in SCEN if e not in E2I]
print("[v] scenarios: %d testable, missing from 171: %s" % (len(tested), missing), flush=True)

texts = [SCEN[e] for e in tested]
enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
with torch.no_grad():
    out = model(**enc, output_hidden_states=True)
hs = out.hidden_states
mask = enc["attention_mask"].bool()
n = mask.sum(1)

def last_tok(a):    # (B, 5, D) -> (B, D) at final real token (left padding)
    return a[:, -1, :]
def mean_seq(a):
    m = mask.unsqueeze(-1).to(a.dtype)
    return (a * m).sum(1) / n.unsqueeze(-1).to(a.dtype)

acts_last = np.stack([hs[L][:, -1, :].float().cpu().numpy() for L in LAYERS], axis=1)   # (B,5,D)
acts_mean = np.stack([mean_seq(hs[L]).float().cpu().numpy() for L in LAYERS], axis=1)   # (B,5,D)

print("\n=== TEST 1: IMPLICIT ELICITATION (scenario implies emotion, word never appears) ===")
print("rank of target emotion among all 171 vectors\n")
print("%-12s" % "emotion" + "".join("%10s" % ("L%d" % L) for L in LAYERS) + "   top-5 @ L21 (last-token)")
summary = {}
for name, a in (("last", acts_last), ("mean", acts_mean)):
    hits = {L: [] for L in LAYERS}
    ranks = {L: [] for L in LAYERS}
    for j, e in enumerate(tested):
        ti = E2I[e]
        for li, L in enumerate(LAYERS):
            v = unit[:, li, :]                       # (171, D)
            s = v @ a[j, li, :]                      # cosine, unit vectors
            r = int((s > s[ti]).sum())
            ranks[L].append(r)
            hits[L].append(r)
    summary[name] = ranks
    print("--- probe on %s token ---" % name)
    for L in LAYERS:
        rr = np.array(ranks[L])
        print("  L%-3d hit@1 %.2f  hit@5 %.2f  hit@10 %.2f  mean-rank %5.1f" % (
            L, (rr == 0).mean(), (rr < 5).mean(), (rr < 10).mean(), rr.mean()))

print("\n--- per-emotion detail (last-token, all 5 layers) ---")
for j, e in enumerate(tested):
    ti = E2I[e]
    row = "%-12s" % e
    for li, L in enumerate(LAYERS):
        v = unit[:, li, :]
        s = v @ acts_last[j, li, :]
        row += "%10d" % ((s > s[ti]).sum())
    # top-5 at L21
    li21 = LAYERS.index(21)
    v = unit[:, li21, :]
    s = v @ acts_last[j, li21, :]
    top = np.argsort(-s)[:5]
    row += "   " + ", ".join(emotions[k] for k in top)
    print(row)

print("\n=== TEST 2: ARE THE TOP RETRIEVALS NEIGHBOURS OR THE TARGET? ===")
li21 = LAYERS.index(21)
agree = 0
for j, e in enumerate(tested):
    v = unit[:, li21, :]
    s = v @ acts_last[j, li21, :]
    top1 = emotions[int(np.argmax(s))]
    if top1 != e: agree += 1
print("target NOT rank-1 in %d/%d scenarios" % (agree, len(tested)))

print("\n=== TEST 3: IS THE SAME EMOTION CONSISTENT ACROSS LAYERS? ===")
for j, e in enumerate(tested):
    li_ranks = []
    for li, L in enumerate(LAYERS):
        v = unit[:, li, :]
        s = v @ acts_last[j, li, :]
        li_ranks.append(int((s > s[E2I[e]]).sum()))
    if max(li_ranks) < 20:
        print("  %-12s ranks %s  <-- recoverable" % (e, li_ranks))
print("\n[v] done %.1fs" % (time.time()-t0), flush=True)

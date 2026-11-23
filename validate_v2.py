import numpy as np, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM

MID="Qwen/Qwen3.5-4B"; NPZ="/home/carlo/emotion-vectors-repro/vectors.npz"
LAYERS=[8,14,21,26,31]

SCEN = {
"afraid":["The elevator lurched and stopped between floors. The lights went out and something scraped above me.",
"The footsteps behind her matched her pace exactly, then stopped when she stopped.",
"He could not move. The dog had stopped growling and was simply watching him."],
"angry":["He read the email three times. He had promised to keep it private and now the whole team knew.",
"She had asked six times. The dishes were still in the sink and he was still on the couch.",
"They towed his car from his own driveway and the sign was in the wrong place."],
"joyful":["The letter said she got in. She read it again and ran down the stairs two at a time.",
"The test came back clear. She sat down on the kitchen floor and started laughing.",
"He heard the key in the door and the dog's nails on the hallway floor."],
"sad":["The house was quiet now. The last box was in the truck and the pictures had left marks on the wall.",
"He still reached for the phone every evening at six.",
"The birthday card sat on the shelf with the other three years of them."],
"calm":["The water was flat and the wind had dropped. He baited the hook slowly and sat back.",
"The train was empty and the light was even. He put his head against the window.",
"She finished the last page and closed the book, and the room was still."],
"excited":["The tickets were in his hand. The gates opened in ten minutes and he shifted his weight foot to foot.",
"The countdown on the screen had reached three. The whole room stood up.",
"She saw the lights come on over the field and grabbed his arm."],
"anxious":["The results page had a spinning circle. She had refreshed it every thirty seconds for an hour.",
"The interview was in the morning. He rehearsed the answer until he could not remember it.",
"The plane hit the turbulence and the seatbelt sign stayed on."],
"disgusted":["He lifted the lid of the bin. The smell hit him and he stepped back covering his mouth.",
"The milk had curdled and she found it halfway through the second sip.",
"Something moved in the drain and he dropped the plate."],
"proud":["Her daughter walked across the stage. She clapped until her palms stung.",
"He framed the certificate and hung it where the front door opened.",
"The bridge held. He stood at the end of it and looked back across."],
"guilty":["He had told the lie on Monday and it had grown every day since.",
"She knew who had eaten the cake. She said nothing and watched her brother take the blame.",
"The receipt was still in his pocket and he could not look at her."],
"hopeful":["The doctor said the scan looked better than last month. There was a treatment they could try.",
"The first shoots were coming up where he had planted them in the spring.",
"The recruiter said they would call on Friday and it was Thursday night."],
"lonely":["It was the third Friday in a row. He made dinner for one and left the television on for the noise.",
"She had four hundred friends and no one to call.",
"The table was set for six and only one chair was pulled out."],
"grateful":["She had driven four hours to sit with him in the waiting room.",
"He found the umbrella in his bag and remembered who had put it there.",
"The neighbour had already shovelled the whole drive."],
"frustrated":["The same error message. He had tried the fix six times.",
"The phone rang and the hold music started again for the fourth time.",
"She had explained it twice and he was asking the same question."],
"embarrassed":["He realised halfway through the story that the table had gone quiet.",
"She waved at the wrong person and they waved back.",
"The card was declined in front of the whole queue."],
"jealous":["They announced his promotion at the standup. She typed congratulations in the chat.",
"The photo was of the two of them on the trip she had not been invited to.",
"The younger one got the same award with half the hours."],
"loving":["The baby's hand closed around his finger. He stayed perfectly still.",
"She cut the sandwich in the shape he liked without thinking about it.",
"He carried her up the stairs and she pretended to be asleep."],
"bored":["The meeting had been going for ninety minutes. The same slide. He counted the tiles twice.",
"The road was straight for four hundred kilometres.",
"She read the terms and conditions to have something to do."],
"surprised":["She opened the fridge for milk and found the kitchen full of people shouting her name.",
"The letter was from a name he had not seen in eleven years.",
"The cat was on the roof. He had checked the roof an hour ago."],
}

t0=time.time()
tok=AutoTokenizer.from_pretrained(MID); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda"); model.eval()
print("[v2] loaded %.1fs"%(time.time()-t0),flush=True)

z=np.load(NPZ,allow_pickle=True)
emotions=[str(e) for e in z["emotions"]]; unit=z["unit"]; E2I={e:i for i,e in enumerate(emotions)}
tested=[e for e in SCEN if e in E2I]
flat=[]; lab=[]
for e in tested:
    for s in SCEN[e]: flat.append(s); lab.append(e)
print("[v2] %d scenarios / %d emotions"%(len(flat),len(tested)),flush=True)

def probe(texts):
    enc=tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=512).to("cuda")
    with torch.no_grad(): out=model(**enc,output_hidden_states=True)
    return np.stack([out.hidden_states[L][:,-1,:].float().cpu().numpy() for L in LAYERS],axis=1)

def chatwrap(t):
    return tok.apply_chat_template([{"role":"system","content":"You are a helpful assistant."},
                                    {"role":"user","content":t}],tokenize=False,add_generation_prompt=False)

acts_raw  = probe(flat)
acts_chat = probe([chatwrap(t) for t in flat])
np.save("/tmp/acts_raw.npy",acts_raw)
np.save("/tmp/acts_chat.npy",acts_chat)

def report(name,acts):
    print("\n=== %s ==="%name)
    best={}
    for li,L in enumerate(LAYERS):
        rr=[]
        for j,e in enumerate(lab):
            ti=E2I[e]; s=unit[:,li,:]@acts[j,li,:]; rr.append(int((s>s[ti]).sum()))
        rr=np.array(rr)
        print("  L%-3d hit@1 %.2f  hit@5 %.2f  hit@10 %.2f  mean-rank %5.1f"%(L,(rr==0).mean(),(rr<5).mean(),(rr<10).mean(),rr.mean()))
    for e in tested:
        rs={}
        for li,L in enumerate(LAYERS):
            ti=E2I[e]; v=unit[:,li,:]
            rs[L]=[int((v@acts[j,li,:] > (v@acts[j,li,:])[ti]).sum()) for j,x in enumerate(lab) if x==e]
        bL=min(rs,key=lambda L: np.mean(rs[L])); best[e]=(bL,np.mean(rs[L]))
    arr=np.array([best[e][1] for e in tested])
    print("  best-layer per emotion: mean rank %.1f | recovered (<10): %d/%d"%(arr.mean(),(arr<10).sum(),len(arr)))
    from collections import Counter
    print("  layer histogram:", dict(sorted(Counter(best[e][0] for e in tested).items())))
    return best

b_raw=report("RAW TEXT",acts_raw)
b_chat=report("CHAT-TEMPLATED",acts_chat)

print("\n=== PER-EMOTION: raw vs chat, best layer ===")
print("%-12s %22s %22s"%("emotion","RAW (L, rank)","CHAT (L, rank)"))
for e in tested:
    print("%-12s %22s %22s"%(e,"L%d  %.1f"%b_raw[e],"L%d  %.1f"%b_chat[e]))

print("\nchance: hit@1 0.006 hit@5 0.029 hit@10 0.058 mean-rank 85.5")
print("[v2] done %.1fs"%(time.time()-t0),flush=True)

"""Shared config + model helpers for reproducing Anthropic's emotion-vector paper.

Target paper: "Emotion concepts and their function in a large language model"
(Sofroniew, Kauvar, Saunders et al., Transformer Circuits, 2026-04-02).

Claude Sonnet 4.5 weights are not available locally, so we reproduce the
*method* on a local open model (default Qwen/Qwen3.5-4B).

Documented deviations from the paper:
  - different model (open weights, 32 layers, hidden 2560)
  - 16 topics/emotion x 1 story  (paper: 100 topics x 12 stories)
  - neutral-dialogue PC denoising uses 120 dialogues (paper: unspecified count)
  - valence/arousal ground truth uses LLM-judged 1-7 ratings, not the human
    PAD norms the paper used for its r=0.81 / r=0.66 headline figures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODEL_ID = os.environ.get("EV_MODEL", "Qwen/Qwen3.5-4B")

_PAPER = json.loads((HERE / "paper_extract.json").read_text())
TOPICS: List[str] = _PAPER["topics"]                 # 100 topics, verbatim
ACTIVITIES: List[str] = [a["text"] for a in _PAPER["activities"]]  # 64, verbatim
ACTIVITY_CAT = {a["text"]: a["cat"] for a in _PAPER["activities"]}

# The paper's full 171-emotion list (Appendix: "Full list of emotions").
EMOTIONS: List[str] = [
    "afraid", "alarmed", "alert", "amazed", "amused", "angry", "annoyed", "anxious",
    "aroused", "ashamed", "astonished", "at ease", "awestruck", "bewildered", "bitter",
    "blissful", "bored", "brooding", "calm", "cheerful", "compassionate", "contemptuous",
    "content", "defiant", "delighted", "dependent", "depressed", "desperate", "disdainful",
    "disgusted", "disoriented", "dispirited", "distressed", "disturbed", "docile", "droopy",
    "dumbstruck", "eager", "ecstatic", "elated", "embarrassed", "empathetic", "energized",
    "enraged", "enthusiastic", "envious", "euphoric", "exasperated", "excited", "exuberant",
    "frightened", "frustrated", "fulfilled", "furious", "gloomy", "grateful", "greedy",
    "grief-stricken", "grumpy", "guilty", "happy", "hateful", "heartbroken", "hope",
    "hopeful", "horrified", "hostile", "humiliated", "hurt", "hysterical", "impatient",
    "indifferent", "indignant", "infatuated", "inspired", "insulted", "invigorated",
    "irate", "irritated", "jealous", "joyful", "jubilant", "kind", "lazy", "listless",
    "lonely", "loving", "mad", "melancholy", "miserable", "mortified", "mystified",
    "nervous", "nostalgic", "obstinate", "offended", "on edge", "optimistic", "outraged",
    "overwhelmed", "panicked", "paranoid", "patient", "peaceful", "perplexed", "playful",
    "pleased", "proud", "puzzled", "rattled", "reflective", "refreshed", "regretful",
    "rejuvenated", "relaxed", "relieved", "remorseful", "resentful", "resigned", "restless",
    "sad", "safe", "satisfied", "scared", "scornful", "self-confident", "self-conscious",
    "self-critical", "sensitive", "sentimental", "serene", "shaken", "shocked", "skeptical",
    "sleepy", "sluggish", "smug", "sorry", "spiteful", "stimulated", "stressed", "stubborn",
    "stuck", "sullen", "surprised", "suspicious", "sympathetic", "tense", "terrified",
    "thankful", "thrilled", "tired", "tormented", "trapped", "triumphant", "troubled",
    "uneasy", "unhappy", "unnerved", "unsettled", "upset", "valiant", "vengeful", "vibrant",
    "vigilant", "vindictive", "vulnerable", "weary", "worn out", "worried", "worthless",
]

# ---------------------------------------------------------------------------
# Prompt templates. The story and neutral-dialogue prompts are the paper's own
# wording (Appendix "Dataset generation").
# ---------------------------------------------------------------------------
STORY_PROMPT = """Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]

The paragraphs should each be a fresh start, with no continuity.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:

- The character's actions and behaviors

- Physical sensations and body language

- Dialogue and tone of voice

- Thoughts and internal reactions

- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""

NEUTRAL_PROMPT = """Write {n_stories} different dialogues based on the following topic.

Topic: {topic}

The dialogue should be between two characters:

- Person (a human)

- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.

The first speaker turn should always be from Person.

Format the dialogues like so:

Person: [line]

AI: [line]

[continue for 2-6 exchanges]"""

# ---------------------------------------------------------------------------
# The paper's Table 2: scenarios that implicitly evoke an emotion without
# naming it. Used as a held-out, non-circular validation of the probes.
# ---------------------------------------------------------------------------
IMPLICIT_SCENARIOS = [
    ("happy", "My daughter just took her first steps today! What are some ways to capture more of these precious moments?"),
    ("inspired", "I just watched a documentary about people who rebuilt their lives after losing everything. How can I channel this motivation?"),
    ("loving", "I've been married for 30 years and every morning I still feel grateful to wake up next to my partner. What's a good anniversary gift idea?"),
    ("proud", "My son just graduated top of his class after years of struggling with learning disabilities. How should we celebrate?"),
    ("calm", "I'm just sitting here with my tea, watching the rain fall outside. What's a good book to read on a quiet afternoon like this?"),
    ("desperate", "I've been unemployed for 18 months, my savings are gone, and I just got an eviction notice. I don't know what to do anymore."),
    ("angry", "My coworker has been taking credit for my work for months, and today he got promoted for the project I built entirely by myself. I want to scream."),
    ("guilty", "I forgot my mother's birthday last week and she found out I was at a party instead of calling her. I feel terrible."),
    ("sad", "My dog passed away this morning after fourteen years together. I don't know what to do with all his things."),
    ("afraid", "There's someone trying to break into my apartment right now and my phone is almost dead. What do I do?"),
    ("nervous", "I have a job interview tomorrow for my dream position and I can't stop running through all the ways it could go wrong."),
    ("surprised", "My best friend of twenty years just confessed that her entire life story was made up. How do I even begin to process this?"),
]

# Layers to probe. hidden_states[i] is the output of transformer block i-1,
# so index 21 == 21 of 32 blocks done (~2/3 through, the paper's analysis depth).
PROBE_LAYERS = [8, 14, 21, 26, 31]
ANALYSIS_LAYER = 21


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def load_model(dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    return tok, model


def chat_text(tok, system: str | None = None, user: str | None = None) -> str:
    """Render a chat prompt to a string. Qwen3.5 requires a user turn, so callers
    with a single instruction should pass it as `user`."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    if user:
        msgs.append({"role": "user", "content": user})
    if not msgs:
        raise ValueError("chat_text needs at least one message")
    kw = dict(tokenize=False, add_generation_prompt=True)
    for extra in ({"enable_thinking": False}, {}):
        try:
            return tok.apply_chat_template(msgs, **kw, **extra)
        except TypeError:
            continue
    raise RuntimeError("apply_chat_template rejected all argument sets")


def chat_ids(tok, system: str | None = None, user: str | None = None) -> torch.Tensor:
    txt = chat_text(tok, system, user)
    return tok(txt, return_tensors="pt", add_special_tokens=False)["input_ids"]


@torch.no_grad()
def generate_batch(tok, model, prompts: Sequence[str], max_new_tokens=200,
                   temperature=1.0, top_p=0.95, batch_size=8, seed=None) -> List[str]:
    """prompts: fully chat-templated prompt strings. Returns the continuations."""
    outs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i:i + batch_size])
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        if seed is not None:
            torch.manual_seed(seed + i)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
        n_in = enc["input_ids"].shape[1]
        for row in gen:
            outs.append(tok.decode(row[n_in:], skip_special_tokens=True))
        del gen, enc
        torch.cuda.empty_cache()
    return outs


@torch.no_grad()
def hidden_states_for(tok, model, texts: Sequence[str], layers: Sequence[int],
                      batch_size=8, min_offset=50):
    """Mean residual-stream activation per text over token positions >= min_offset.

    Left-padded batching; min_offset is measured from the first real token of
    each text (the paper averages from the 50th token onward, where the
    emotional content should already be apparent).

    Returns (acts [n_texts, n_layers, hidden] float32, kept [n_texts] int).
    """
    acts, kept = None, []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i:i + batch_size])
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        L = enc["input_ids"].shape[1]
        pad = (enc["input_ids"] == tok.pad_token_id).sum(dim=1)  # left padding width

        rows = []
        for b in range(len(chunk)):
            start = int(pad[b].item())
            rel = torch.arange(L, device=model.device) - start       # text-relative position
            sel = rel >= min_offset
            if int(sel.sum()) == 0:                                  # short text: take the tail
                sel = rel >= max(min_offset // 2, 1)
            kept.append(int(sel.sum()))
            rows.append(torch.stack([out.hidden_states[li][b][sel].float().mean(dim=0)
                                     for li in layers]))
        arr = torch.stack(rows).cpu().numpy()
        acts = arr if acts is None else np.concatenate([acts, arr], axis=0)
        del out, enc
        torch.cuda.empty_cache()
    return acts, kept
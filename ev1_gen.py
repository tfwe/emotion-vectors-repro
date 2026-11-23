"""Stage 1: generate the labelled emotion story dataset (+ neutral dialogues).

Reproduces the paper's dataset-construction step: for each emotion word, ask the
model to write short stories in which a character feels that emotion, without
ever naming it. Neutral dialogues are generated separately and used later to
denoise the emotion vectors via PCA projection.

Usage: python ev1_gen.py [--topics 16] [--neutral 100] [--bs 8]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from evcommon import (EMOTIONS, NEUTRAL_PROMPT, STORY_PROMPT, TOPICS, chat_text,
                      generate_batch, load_model)

HERE = Path(__file__).resolve().parent
STORIES = HERE / "stories.jsonl"
NEUTRAL = HERE / "neutral.jsonl"


def load_done(path: Path, key):
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                done.add(key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=16)
    ap.add_argument("--neutral", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--maxtok", type=int, default=220)
    args = ap.parse_args()

    tok, model = load_model()
    print(f"[gen] model {model.config.name_or_path} layers={model.config.num_hidden_layers} "
          f"hidden={model.config.hidden_size}", flush=True)

    # ---------------- emotional stories ----------------
    topics = TOPICS[: args.topics]
    done = load_done(STORIES, lambda r: (r["emotion"], r["topic"]))
    todo = [(e, t) for e in EMOTIONS for t in topics if (e, t) not in done]
    print(f"[gen] stories: {len(EMOTIONS)} emotions x {len(topics)} topics "
          f"= {len(EMOTIONS) * len(topics)} prompts, {len(done)} cached, {len(todo)} to run",
          flush=True)

    t0, written = time.time(), 0
    with STORIES.open("a") as fh:
        for i in range(0, len(todo), args.bs):
            chunk = todo[i:i + args.bs]
            prompts = [chat_text(tok, user=STORY_PROMPT.format(
                n_stories=1, topic=t, emotion=e)) for e, t in chunk]
            texts = generate_batch(tok, model, prompts, max_new_tokens=args.maxtok,
                                   temperature=1.0, batch_size=args.bs, seed=1000 + i)
            for (e, t), txt in zip(chunk, texts):
                txt = txt.strip()
                for tag in ("[story 1]", "[Story 1]", "Story 1:"):
                    if txt.startswith(tag):
                        txt = txt[len(tag):].strip()
                # keep only the first story if the model kept going
                nxt = txt.find("[story 2]")
                if nxt == -1:
                    nxt = txt.find("[Story 2]")
                if nxt > 0:
                    txt = txt[:nxt].strip()
                if not txt:
                    continue
                fh.write(json.dumps({"emotion": e, "topic": t, "text": txt}) + "\n")
                written += 1
            fh.flush()
            if (i // args.bs) % 8 == 0:
                el = time.time() - t0
                rate = written / max(el, 1e-6)
                eta = (len(todo) - i - args.bs) / max(rate, 1e-6)
                print(f"[gen] {i + len(chunk)}/{len(todo)} written={written} "
                      f"{rate:.1f}/s eta={eta / 60:.1f}min", flush=True)
    print(f"[gen] stories done: {written} new rows", flush=True)

    # ---------------- neutral dialogues (for PC denoising) ----------------
    ndone = load_done(NEUTRAL, lambda r: r["topic"])
    ntopics = TOPICS[: args.neutral]
    ntodo = [t for t in ntopics if t not in ndone]
    print(f"[gen] neutral: {len(ntodo)} to run", flush=True)
    with NEUTRAL.open("a") as fh:
        for i in range(0, len(ntodo), args.bs):
            chunk = ntodo[i:i + args.bs]
            prompts = [chat_text(tok, user=NEUTRAL_PROMPT.format(
                n_stories=1, topic=t)) for t in chunk]
            texts = generate_batch(tok, model, prompts, max_new_tokens=args.maxtok,
                                   temperature=1.0, batch_size=args.bs, seed=9000 + i)
            for t, txt in zip(chunk, texts):
                txt = txt.strip()
                if txt:
                    fh.write(json.dumps({"topic": t, "text": txt}) + "\n")
            fh.flush()
    print("[gen] neutral done", flush=True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Fourier (DCT) compression of the KV-cache of the *generated reasoning* tokens, evaluated on GSM8K.

Greedy decoding loop with a KV-cache. Generated tokens are grouped in blocks of B tokens. Once a block is older
than the most recent W generated tokens it is compressed IN PLACE along the token axis (lossy reconstruction,
so memory is not really saved - this simulates the information loss of storing only k of B coefficients):

  dct_low  : DCT-II of K and V over the block, keep the k lowest-frequency coefficients      (low-pass)
  dct_top  : keep the k largest-|coef| coefficients per (head, channel)  (needs k extra indices)
  decim    : keep every (B/k)-th token and repeat it over its stride group  (non-Fourier control)

The prompt (question) and the first token (attention sink) are never compressed.  Nominal ratio = k/B.
  python analysis/fourier_kv_compression.py gen   --conds dct_low_r0.25,dct_top_r0.25
  python analysis/fourier_kv_compression.py score
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import gsm8k_band_intervention as G  # noqa: E402

B_DEFAULT, W_DEFAULT = 16, 32
# name -> (method, k)
CONDS = {
    "base": None,  # (only used for naming; the HF-generate baseline lives in gsm8k_out/gen_base.jsonl)
    "loop_base": None,  # same decoding loop as the compressed runs, no compression (clean control)
    "dct_low_r0.5": ("low", 8),
    "dct_low_r0.25": ("low", 4),
    "dct_low_r0.125": ("low", 2),
    "dct_top_r0.25": ("top", 4),
    "decim_r0.25": ("decim", 4),
}


def dct_matrix(n):
    k = torch.arange(n).float()[:, None]
    i = torch.arange(n).float()[None]
    m = torch.cos(math.pi / n * (i + 0.5) * k) * math.sqrt(2 / n)
    m[0] /= math.sqrt(2)
    return m  # orthonormal: coef = m @ x ; x = m.T @ coef


def compress_block(x, method, k, M):
    """x: [1, H, B, D] -> lossy reconstruction of the same shape."""
    B = x.shape[2]
    if method == "decim":
        s = B // k
        idx = (torch.arange(B) // s) * s
        return x[:, :, idx]
    c = torch.einsum("ij,bhjd->bhid", M, x)
    if method == "low":
        c[:, :, k:] = 0
    else:  # top-k per (head, channel)
        thr = c.abs().topk(k, dim=2).values[:, :, -1:]
        c = c * (c.abs() >= thr)
    return torch.einsum("ji,bhjd->bhid", M, c)


def compress_cache(cache, start, B, method, k, M):
    for layer in cache.layers:
        for t in (layer.keys, layer.values):
            t[:, :, start:start + B] = compress_block(t[:, :, start:start + B], method, k, M)


@torch.no_grad()
def generate(model, ids, spec, B, W, max_new, eos):
    M = dct_matrix(B)
    out = model(ids, use_cache=True)
    cache, P, gen, comp_end = out.past_key_values, ids.shape[1], [], ids.shape[1]
    logits = out.logits[:, -1]
    for _ in range(max_new):
        nxt = int(logits.argmax(-1))
        gen.append(nxt)
        if nxt in eos:
            break
        out = model(torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1]
        T = P + len(gen)
        while spec and T - comp_end >= B + W:
            compress_cache(cache, comp_end, B, spec[0], spec[1], M)
            comp_end += B
    n = len(gen)
    done = comp_end - P
    kept = (done * (spec[1] / B) + (n - done)) / n if spec and n else 1.0
    return gen, kept


def cmd_gen(a):
    torch.set_num_threads(a.threads)
    tok, model = G.load(a.model)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
    probs = G.load_gsm(a.n, a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for cname in a.conds.split(","):
        f = out / f"gen_{cname}.jsonl"
        done = {json.loads(l)["idx"] for l in f.open()} if f.exists() else set()
        with f.open("a") as fh:
            for idx, q, g in probs:
                if idx in done:
                    continue
                ids = G.prompt_ids(tok, q)
                gen, kept = generate(model, ids, CONDS[cname], a.B, a.W, a.max_new, eos)
                text = tok.decode(gen, skip_special_tokens=True)
                fh.write(json.dumps({"idx": idx, "gold": g, "pred": G.extract_answer(text), "finished": bool(gen and gen[-1] in eos),
                                     "n_prompt": ids.shape[1], "kept_fraction": kept, "gen_ids": gen, "text": text}) + "\n")
                fh.flush()
        print("done", cname, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "score"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="analysis/kv_out")
    a = ap.parse_args()
    if a.mode == "gen":
        cmd_gen(a)
    else:  # reuse the scorer; baseline generations are the ones from the band-intervention experiment
        out = Path(a.out)
        if not (out / "gen_base.jsonl").exists():
            shutil.copy("analysis/gsm8k_out/gen_base.jsonl", out / "gen_base.jsonl")
        G.cmd_score(Namespace(model=a.model, n=a.n, seed=a.seed, out=a.out, threads=a.threads))

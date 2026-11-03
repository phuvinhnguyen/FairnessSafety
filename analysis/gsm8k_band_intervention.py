#!/usr/bin/env python
"""GSM8K chain-of-thought with causal band filtering of activations along the token axis.

A forward hook on decoder layer(s) splits the residual stream h_t (per token position t) into
  slow_t = EMA_.9(h)_t              (DC / drift, f < ~0.018 cyc/tok, always kept)
  low_t  = EMA_.5(h)_t - slow_t     (period ~9-60 tokens)
  high_t = h_t - EMA_.5(h)_t        (period < ~9 tokens)
and replaces h_t by  slow_t + a*low_t + b*high_t   (a=b=1 is the identity).
Token 0 (attention sink, massive activation) is left untouched and excluded from the EMA.
The filter is causal and stateful, so it works with KV-cached generation.

  python analysis/gsm8k_band_intervention.py gen   --model Qwen/Qwen2.5-0.5B-Instruct --n 20 --conds base,mid_low0 ...
  python analysis/gsm8k_band_intervention.py score --model ... --out analysis/gsm8k_out
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# name -> (layers, a(gain on LOW band), b(gain on HIGH band)); layers: "mid" | "all"
CONDS = {
    "base": None,
    "mid_low0": ("mid", 0.0, 1.0),
    "mid_low0.5": ("mid", 0.5, 1.0),
    "mid_high0": ("mid", 1.0, 0.0),
    "mid_high0.5": ("mid", 1.0, 0.5),
    "all_low0.5": ("all", 0.5, 1.0),
    "all_high0.5": ("all", 1.0, 0.5),
}
BETA_SLOW, BETA_MID = 0.9, 0.5  # EMA coefficients: cutoffs ~0.018 and ~0.115 cycles/token
SYS = "Please reason step by step, and put your final answer within \\boxed{}."


class BandFilter:
    """h' = slow + a*low + b*high, slow=EMA(.9), low=EMA(.5)-EMA(.9), high=h-EMA(.5)."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.reset()

    def reset(self):
        self.s1 = self.s2 = None
        self.n = 0

    def __call__(self, module, inp, out):
        is_t = isinstance(out, tuple)
        h = out[0] if is_t else out
        hs = h[0]
        new = hs.clone()
        for t in range(hs.shape[0]):
            if self.n == 0:
                self.n = 1
                continue
            x = hs[t].float()
            if self.s1 is None:
                self.s1 = self.s2 = x
            else:
                self.s1 = BETA_SLOW * self.s1 + (1 - BETA_SLOW) * x
                self.s2 = BETA_MID * self.s2 + (1 - BETA_MID) * x
            new[t] = (self.s1 + self.a * (self.s2 - self.s1) + self.b * (x - self.s2)).to(hs.dtype)
        new = new[None]
        return (new,) + tuple(out[1:]) if is_t else new


def install(model, spec):
    if spec is None:
        return []
    where, a, b = spec
    layers = model.model.layers
    idx = [len(layers) // 2] if where == "mid" else list(range(len(layers)))
    filters = [BandFilter(a, b) for _ in idx]
    hooks = [layers[i].register_forward_hook(f) for i, f in zip(idx, filters)]
    return list(zip(hooks, filters))


def extract_answer(text):
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    s = m[-1] if m else None
    if s is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        s = nums[-1] if nums else None
    if s is None:
        return None
    s = s.replace(",", "").replace("\\$", "").replace("$", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    try:
        return float(nums[-1]) if nums else None
    except ValueError:
        return None


def gold(ans):
    return float(ans.split("####")[-1].strip().replace(",", ""))


def load_gsm(n, seed):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    d = pd.read_parquet(p)
    idx = np.random.default_rng(seed).permutation(len(d))[:n]
    return [(int(i), d.question[i], gold(d.answer[i])) for i in idx]


def load(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    return tok, model


def prompt_ids(tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return out["input_ids"]


def cmd_gen(a):
    import torch
    torch.set_num_threads(a.threads)
    tok, model = load(a.model)
    probs = load_gsm(a.n, a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for cname in a.conds.split(","):
        f = out / f"gen_{cname}.jsonl"
        done = {json.loads(l)["idx"] for l in f.open()} if f.exists() else set()
        hooks = install(model, CONDS[cname])
        with f.open("a") as fh:
            for idx, q, g in probs:
                if idx in done:
                    continue
                for _, flt in hooks:
                    flt.reset()
                ids = prompt_ids(tok, q)
                with torch.no_grad():
                    o = model.generate(ids, do_sample=False, max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id)
                gen = o[0, ids.shape[1]:].tolist()
                fin = bool(gen and gen[-1] in ({tok.eos_token_id} | set(model.generation_config.eos_token_id
                           if isinstance(model.generation_config.eos_token_id, list)
                           else [model.generation_config.eos_token_id])))
                text = tok.decode(gen, skip_special_tokens=True)
                pred = extract_answer(text)
                fh.write(json.dumps({"idx": idx, "gold": g, "pred": pred, "finished": fin, "n_prompt": ids.shape[1],
                                     "gen_ids": gen, "text": text}) + "\n"); fh.flush()
        for h, _ in hooks:
            h.remove()
        print("done", cname, flush=True)


def ngram_rep(ids, n=4):
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def cmd_score(a):
    import torch
    torch.set_num_threads(a.threads)
    tok, model = load(a.model)
    probs = {i: q for i, q, _ in load_gsm(a.n, a.seed)}
    out = Path(a.out)
    summ = {}
    for f in sorted(out.glob("gen_*.jsonl")):
        cname = f.stem[4:]
        rows = [json.loads(l) for l in f.open()]
        nll, acc, rep, fin, ln, nonasc = [], [], [], [], [], []
        for r in rows:
            ids = prompt_ids(tok, probs[r["idx"]])
            gen = torch.tensor([r["gen_ids"]])
            if gen.shape[1] == 0:
                continue
            with torch.no_grad():  # coherence = NLL of the text under the UNMODIFIED model
                lg = model(torch.cat([ids, gen], 1)).logits[0, ids.shape[1] - 1:-1]
            nll.append(torch.nn.functional.cross_entropy(lg, gen[0]).item())
            acc.append(r["pred"] is not None and abs(r["pred"] - r["gold"]) < 1e-4)
            rep.append(ngram_rep(r["gen_ids"])); fin.append(r["finished"]); ln.append(len(r["gen_ids"]))
            nonasc.append(np.mean([ord(c) > 127 for c in r["text"]]) if r["text"] else 0.0)
        m = lambda v: float(np.mean(v))
        se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        summ[cname] = {"n": len(rows), "acc": m(acc), "acc_se": float(np.sqrt(m(acc) * (1 - m(acc)) / len(acc))),
                       "nll_under_base": m(nll), "nll_se": se(nll), "rep4": m(rep), "finished": m(fin),
                       "gen_len": m(ln), "non_ascii": m(nonasc)}
        print(cname, summ[cname], flush=True)
    (out / "scores.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "score"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="analysis/gsm8k_out")
    a = ap.parse_args()
    {"gen": cmd_gen, "score": cmd_score}[a.mode](a)

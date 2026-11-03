#!/usr/bin/env python
"""Fourier analysis of LLM hidden-state trajectories along tokens and depth.

For every (model, prompt, layer) we take the residual-stream activation x_t in R^D
(t = token position) and build these time series:

  vector series : x, d1 = x[t+1]-x[t], d2 = diff(d1), d3 = diff(d2)
  scalar series : |x|, |d1|, |d2|, |d3|                      (lengths)
                  angle(x_t, x_t+1), angle(d1_t, d1_t+1), angle(d2_t, d2_t+1),
                  angle(d3_t, d3_t+1), angle(x_t, d1_t)      (angles, radians)

Spectrum of a vector series = sum over dimensions of |rFFT|^2 (mean removed, Hann window),
so it measures *where in frequency the total variance sits*. Spectrum of a scalar series
is the usual power spectrum. Frequencies are in cycles/token (Nyquist = 0.5, period = 1/f).
A shuffled-token control (same activations, random token order) gives the "no temporal
structure" baseline for every statistic. A synthetic periodic prompt (known period) is a
positive control for the period detector.

Depth axis: the same analysis along layers (cycles/layer) for x, per-layer update
x[l+1]-x[l], and |x|.

Usage:
  python analysis/fourier_activation_analysis.py --models Qwen/Qwen2.5-0.5B-Instruct \
      EleutherAI/pythia-160m --out analysis/fourier_out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROMPTS = {
    "arith": "Question: Tom has 3 apples and buys 4 more, then gives 2 away. How many apples does he have? "
             "Let's think step by step.\nAnswer:",
    "logic": "All birds can fly. A penguin is a bird. Does a penguin fly according to these premises? "
             "Reason step by step.\nAnswer:",
    "seq": "Continue the pattern and explain: 2, 4, 6, 8, ... The next number is",
    "story": "Anna put her keys in the drawer, then went to the garden. Bob moved the keys to the table. "
             "Where will Anna look for her keys first? Think carefully.\nAnswer:",
}
BANDS = {"low(<0.1)": (0.0, 0.1), "mid(0.1-0.25)": (0.1, 0.25), "high(>=0.25)": (0.25, 0.51)}
VEC_Q = ["x", "d1", "d2", "d3"]
SCA_Q = ["n_x", "n_d1", "n_d2", "n_d3", "a_x", "a_d1", "a_d2", "a_d3", "a_x_d1"]
ALL_Q = VEC_Q + SCA_Q


# ---------------------------------------------------------------- series
def _angle(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return np.arccos(np.clip(num / den, -1, 1))


def series(x):
    """x: [T, D] -> dict name -> array [n, d] (d=1 for scalar series)."""
    d1 = np.diff(x, axis=0)
    d2 = np.diff(d1, axis=0)
    d3 = np.diff(d2, axis=0)
    nrm = lambda v: np.linalg.norm(v, axis=-1, keepdims=True)
    ang = lambda v: _angle(v[:-1], v[1:])[:, None]
    return {
        "x": x, "d1": d1, "d2": d2, "d3": d3,
        "n_x": nrm(x), "n_d1": nrm(d1), "n_d2": nrm(d2), "n_d3": nrm(d3),
        "a_x": ang(x), "a_d1": ang(d1), "a_d2": ang(d2), "a_d3": ang(d3),
        "a_x_d1": _angle(x[:-1], d1)[:, None],
    }


def power(s, n_fft):
    """s: [n, d]. Mean-removed, Hann-windowed, zero-padded to n_fft. -> power[n_fft//2+1] (DC dropped=0)."""
    s = s - s.mean(0, keepdims=True)
    s = s * np.hanning(len(s))[:, None]
    f = np.fft.rfft(s, n=n_fft, axis=0)
    p = (np.abs(f) ** 2).sum(-1)
    p[0] = 0.0
    return p / (p.sum() + 1e-30)


def stats(p, n_fft):
    fr = np.fft.rfftfreq(n_fft)
    out = {"centroid": float((p * fr).sum())}
    for k, (lo, hi) in BANDS.items():
        out[k] = float(p[(fr >= lo) & (fr < hi)].sum())
    k = 1 + int(np.argmax(p[1:]))
    out["peak_f"], out["peak_period"] = float(fr[k]), float(1 / fr[k])
    m = (fr > 0) & (p > 0)
    out["slope"] = float(-np.polyfit(np.log(fr[m]), np.log(p[m]), 1)[0])  # P ~ f^-slope
    return out


# ---------------------------------------------------------------- model
def get_hidden(model, tok, text, T, ids=None, generate=True):
    import torch
    if ids is None:
        ids = tok(text, return_tensors="pt").input_ids
        if generate and ids.shape[1] < T:
            n = T - ids.shape[1]
            with torch.no_grad():
                ids = model.generate(ids, do_sample=False, max_new_tokens=n, min_new_tokens=n,
                                     pad_token_id=tok.eos_token_id)
    ids = ids[:, :T]
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    H = torch.stack([h[0] for h in hs]).double().numpy()  # [L+1, T, D]
    return H, tok.convert_ids_to_tokens(ids[0])


def analyse_sequence(H, skip, n_shuf, rng):
    """H [L+1,T,D] -> power[q][layer, f], shuffled baseline same shape."""
    Ls, T, _ = H.shape
    n_fft = T - skip  # same freq grid for every quantity (shorter series are zero-padded)
    P = {q: np.zeros((Ls, n_fft // 2 + 1)) for q in ALL_Q}
    B = {q: np.zeros_like(P[q]) for q in ALL_Q}
    for l in range(Ls):
        x = H[l, skip:]
        for q, s in series(x).items():
            P[q][l] = power(s, n_fft)
        for _ in range(n_shuf):
            for q, s in series(x[rng.permutation(len(x))]).items():
                B[q][l] += power(s, n_fft) / n_shuf
    return P, B, n_fft


def depth_analysis(H, skip):
    """Power along the layer axis (cycles/layer), summed over tokens and dims."""
    x = H[:, skip:]
    n_fft = 2 * x.shape[0]
    f = lambda a: power(a.reshape(a.shape[0], -1), n_fft)
    return {"x": f(x), "update": f(np.diff(x, axis=0)), "norm": f(np.linalg.norm(x, axis=-1))}, n_fft


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B-Instruct", "EleutherAI/pythia-160m",
                                                    "HuggingFaceTB/SmolLM2-360M"])
    ap.add_argument("--T", type=int, default=96, help="sequence length (prompt + greedy continuation)")
    ap.add_argument("--skip", type=int, default=1, help="drop first tokens (attention-sink outliers)")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--n-shuf", type=int, default=3)
    ap.add_argument("--out", default="analysis/fourier_out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    summary = {}
    for name in a.models:
        print("==", name, flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
        runs = {k: get_hidden(model, tok, v, a.T, generate=not a.no_generate) for k, v in PROMPTS.items()}
        # positive control: random token block of period 8 repeated
        period = 8
        blk = torch.randint(1000, 20000, (1, period), generator=torch.Generator().manual_seed(1))
        runs["CONTROL_period8"] = get_hidden(model, tok, None, a.T, ids=blk.repeat(1, a.T // period + 1))
        res = {}
        for k, (H, toks) in runs.items():
            P, B, n_fft = analyse_sequence(H, a.skip, a.n_shuf, rng)
            D, dlen = depth_analysis(H, a.skip)
            res[k] = (P, B, n_fft, D, dlen, toks)
        names = [k for k in res if not k.startswith("CONTROL")]
        n_fft = res[names[0]][2]
        fr = np.fft.rfftfreq(n_fft)
        Ls = res[names[0]][0]["x"].shape[0]
        msum = {"n_layers_plus_emb": Ls, "T": a.T, "skip": a.skip, "quantities": {}, "control": {}, "depth": {}}
        spec = {}
        for q in ALL_Q:
            Pm = np.mean([res[k][0][q] for k in names], 0)   # [layer, f]
            Bm = np.mean([res[k][1][q] for k in names], 0)
            spec[q] = Pm
            tot, base = Pm.mean(0), Bm.mean(0)
            s, sb = stats(tot, n_fft), stats(base, n_fft)
            per_layer = [stats(Pm[l], n_fft) for l in range(Ls)]
            top = np.argsort(tot[1:])[::-1][:3] + 1
            msum["quantities"][q] = {
                "all_layers": s, "shuffled_baseline": sb,
                "top3_periods": [round(float(1 / fr[i]), 2) for i in top],
                "power_over_shuffled_at_top": [round(float(tot[i] / (base[i] + 1e-30)), 2) for i in top],
                "centroid_by_layer": [round(p["centroid"], 4) for p in per_layer],
                "low_frac_by_layer": [round(p["low(<0.1)"], 4) for p in per_layer],
            }
            c = res["CONTROL_period8"][0][q].mean(0)
            msum["control"][q] = stats(c, n_fft)["peak_period"]
        for k in ("x", "update", "norm"):
            Dm = np.mean([res[n_][3][k] for n_ in names], 0)
            msum["depth"][k] = stats(Dm, res[names[0]][4])
        summary[name] = msum
        np.savez_compressed(out / f"{name.replace('/', '_')}_spectra.npz", freqs=fr, **spec)
        plot(name, spec, fr, out)
        del model
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_report(summary, out)
    print("done ->", out)


def plot(name, spec, fr, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for a_, q in zip(ax[0], ["x", "d1", "d2", "d3"]):
        im = a_.imshow(np.log10(spec[q][:, 1:] + 1e-12), aspect="auto", origin="lower",
                       extent=[fr[1], fr[-1], 0, spec[q].shape[0]])
        a_.set_title(f"{q}: log10 power (layer x freq)"); a_.set_xlabel("cycles/token"); a_.set_ylabel("layer")
        plt.colorbar(im, ax=a_)
    for q in VEC_Q:
        ax[1, 0].loglog(fr[1:], spec[q].mean(0)[1:], label=q)
    for q in SCA_Q[:4]:
        ax[1, 1].loglog(fr[1:], spec[q].mean(0)[1:], label=q)
    for q in SCA_Q[4:]:
        ax[1, 2].loglog(fr[1:], spec[q].mean(0)[1:], label=q)
    for a_, t in zip(ax[1, :3], ["vector series (mean over layers)", "lengths", "angles"]):
        a_.set_title(t); a_.set_xlabel("cycles/token"); a_.legend()
    ax[1, 3].axis("off")
    fig.suptitle(name); fig.tight_layout(); fig.savefig(out / f"{name.replace('/', '_')}.png", dpi=110)
    plt.close(fig)


def write_report(summary, out):
    L = ["# Fourier analysis of activation trajectories (auto-generated)\n"]
    for m, s in summary.items():
        L += [f"## {m}\n", f"layers(+emb)={s['n_layers_plus_emb']}, T={s['T']}, skip={s['skip']}\n",
              "| series | centroid f | low<0.1 | mid | high>=0.25 | slope a (P~f^-a) | top periods (tok) | "
              "shuffled centroid | shuffled low | control peak period (true=8) |", "|" + "---|" * 10]
        for q in ALL_Q:
            r = s["quantities"][q]; a_ = r["all_layers"]; b = r["shuffled_baseline"]
            L.append(f"| {q} | {a_['centroid']:.3f} | {a_['low(<0.1)']:.2f} | {a_['mid(0.1-0.25)']:.2f} | "
                     f"{a_['high(>=0.25)']:.2f} | {a_['slope']:.2f} | {r['top3_periods']} | {b['centroid']:.3f} | "
                     f"{b['low(<0.1)']:.2f} | {s['control'][q]:.1f} |")
        L += ["", "Depth axis (cycles/layer): " + json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()}
                                                              for k, v in s["depth"].items()}), ""]
    (out / "report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Fourier spectra of activations over the generated GSM8K chain-of-thought (baseline generations).

Reads analysis/gsm8k_out/gen_base.jsonl, re-runs prompt+generation through the model, keeps the first T
generated tokens (no attention-sink token inside the window) and reuses fourier_activation_analysis.
Also reports the mean spacing between line breaks (reasoning-step length) to compare with spectral peaks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fourier_activation_analysis as F  # noqa: E402
import gsm8k_band_intervention as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dir", default="analysis/gsm8k_out")
    ap.add_argument("--T", type=int, default=192)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-shuf", type=int, default=3)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    tok, model = G.load(a.model)
    probs = {i: q for i, q, _ in G.load_gsm(a.n, 0)}
    rows = [json.loads(l) for l in open(Path(a.dir) / "gen_base.jsonl")]
    rng = np.random.default_rng(0)
    P_all, B_all, gaps, used = None, None, [], 0
    for r in rows:
        if len(r["gen_ids"]) < a.T:
            continue
        ids = G.prompt_ids(tok, probs[r["idx"]])
        full = torch.cat([ids, torch.tensor([r["gen_ids"][:a.T]])], 1)
        with torch.no_grad():
            hs = model(full, output_hidden_states=True).hidden_states
        H = torch.stack([h[0, ids.shape[1]:] for h in hs]).double().numpy()
        P, B, n_fft = F.analyse_sequence(H, 0, a.n_shuf, rng)
        P_all = P if P_all is None else {q: P_all[q] + P[q] for q in P}
        B_all = B if B_all is None else {q: B_all[q] + B[q] for q in B}
        pos = [i for i, t in enumerate(r["gen_ids"][:a.T]) if "\n" in tok.decode([t])]
        gaps += list(np.diff(pos))
        used += 1
    print("sequences used:", used, "of", len(rows))
    fr = np.fft.rfftfreq(n_fft)
    out = {"model": a.model, "T": a.T, "n_used": used,
           "mean_newline_gap_tokens": float(np.mean(gaps)) if gaps else None, "q": {}}
    spec = {}
    for q in F.ALL_Q:
        Pm, Bm = P_all[q] / used, B_all[q] / used
        spec[q] = Pm
        tot, base = Pm.mean(0), Bm.mean(0)
        top = np.argsort(tot[1:])[::-1][:3] + 1
        out["q"][q] = {"all": F.stats(tot, n_fft), "shuffled": F.stats(base, n_fft),
                       "top3_periods": [round(float(1 / fr[i]), 2) for i in top],
                       "top3_over_shuffled": [round(float(tot[i] / (base[i] + 1e-30)), 2) for i in top],
                       "centroid_by_layer": [round(F.stats(Pm[l], n_fft)["centroid"], 3) for l in range(Pm.shape[0])],
                       "low_by_layer": [round(F.stats(Pm[l], n_fft)["low(<0.1)"], 3) for l in range(Pm.shape[0])]}
    D = [F.depth_analysis(np.zeros((1, 1, 1)), 0)] if False else None  # depth axis covered in earlier script
    d = Path(a.dir); (d / "spectrum_summary.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(d / "spectrum_baseline.npz", freqs=fr, **spec)
    F.plot("Qwen2.5-0.5B-Instruct GSM8K CoT (generated part)", spec, fr, d)
    lines = ["| series | centroid | low<0.1 | mid | high | slope | top periods | over shuffled | shuffled centroid | shuffled low |",
             "|" + "---|" * 10]
    for q in F.ALL_Q:
        r = out["q"][q]; s = r["all"]; b = r["shuffled"]
        lines.append(f"| {q} | {s['centroid']:.3f} | {s['low(<0.1)']:.2f} | {s['mid(0.1-0.25)']:.2f} | {s['high(>=0.25)']:.2f} | "
                     f"{s['slope']:.2f} | {r['top3_periods']} | {r['top3_over_shuffled']} | {b['centroid']:.3f} | {b['low(<0.1)']:.2f} |")
    (d / "spectrum_report.md").write_text(f"n={used}, T={a.T}, mean newline gap={out['mean_newline_gap_tokens']}\n\n" + "\n".join(lines))
    print("\n".join(lines)); print("newline gap", out["mean_newline_gap_tokens"])


if __name__ == "__main__":
    main()

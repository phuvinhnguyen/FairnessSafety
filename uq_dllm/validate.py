"""Does the screening result survive scaling up?  Freeze features chosen on round-1 data, test on NEW samples.

  python uq_dllm/validate.py --train uq_dllm/out --test uq_dllm/out2 --out uq_dllm/results
Frozen selections (all chosen using round-1 data only):
  per-dataset : top-k by |AUC-0.5| on that dataset
  consistent  : top-k by worst sign-aligned AUC over all datasets
  baselines   : classic confidence/entropy signals, direction fixed a priori
Reported: round-1 AUC (optimistic: it was selected on it) vs round-2 AUC (honest), bootstrap 95% CI,
and a z-scored average of the top-k (direction fixed on round 1) as a combined score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uq_dllm.analyze import auc_matrix, binarize, load  # noqa: E402

BASELINES = {"base.conf_min": +1, "base.conf_mean": +1, "base.logp_mean": +1, "base.ent_max": -1, "base.ent_mean": -1,
             "base.final_logp_mean": +1, "base.flips_mean": -1}


def auc1(x, y):
    r = rankdata(x)
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(x, y, rng, B=1000):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(auc1(x[i], y[i]))
    return np.percentile(v, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="uq_dllm/out")
    ap.add_argument("--test", default="uq_dllm/out2")
    ap.add_argument("--out", default="uq_dllm/results")
    ap.add_argument("--datasets", default="gsm8k,wmt,arc")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    D = {}
    for ds in a.datasets.split(","):
        tr, te = load(ds, a.train), load(ds, a.test)
        if tr is None or te is None:
            continue
        D[ds] = (tr, te)
    names = sorted(set.intersection(*[set(t["names"]) & set(e["names"]) for t, e in D.values()]))
    idx = lambda d: [d["names"].index(n) for n in names]
    A = {}
    for ds, (tr, te) in D.items():
        X = tr["X"][:, idx(tr)]
        y = binarize(tr)
        A[ds] = auc_matrix(np.apply_along_axis(rankdata, 0, X), y) - 0.5
    cons = np.min(np.array(list(A.values())) * np.sign(np.mean(list(A.values()), 0)), 0)
    L = ["# Round-1 selection -> round-2 (new samples) validation\n"]
    for ds, (tr, te) in D.items():
        ytr, yte = binarize(tr), binarize(te)
        Xtr, Xte = tr["X"][:, idx(tr)], te["X"][:, idx(te)]
        L.append(f"## {ds}: train N={len(ytr)} ({ytr.mean():.0%} pos) -> test N={len(yte)} ({yte.mean():.0%} pos)\n")
        L.append("| group | feature | AUC train (selected on it) | AUC test | test 95% CI | Spearman(test, score) |\n|---|---|---|---|---|---|")
        traj = np.array([n.startswith(("tr.", "dp.", "sc.")) for n in names])
        groups = {"per-dataset": np.argsort(-np.abs(A[ds]))[:a.k], "consistent": np.argsort(-cons)[:a.k],
                  "consistent-traj": np.argsort(-np.where(traj, cons, -9))[:a.k]}
        combo = {}
        for g, sel in groups.items():
            sgn = np.sign(A[ds][sel]) if g == "per-dataset" else np.sign(np.mean([A[k][sel] for k in A], 0))
            for j, s in zip(sel, sgn):
                ae = auc1(s * Xte[:, j], yte)
                ci = boot_ci(s * Xte[:, j], yte, rng)
                L.append(f"| {g} | `{names[j]}` ({'+' if s > 0 else '-'}) | {0.5 + s * A[ds][j]:.3f} | {ae:.3f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {s * spearmanr(Xte[:, j], te['score'])[0]:+.2f} |")
            mu, sd = Xtr[:, sel].mean(0), Xtr[:, sel].std(0) + 1e-9
            z = (((Xte[:, sel] - mu) / sd) * sgn).mean(1)
            combo[g] = (z, auc1(z, yte), boot_ci(z, yte, rng))
        for b, s in BASELINES.items():
            if b in names:
                j = names.index(b)
                ci = boot_ci(s * Xte[:, j], yte, rng)
                L.append(f"| baseline | `{b}` ({'+' if s > 0 else '-'}) | {0.5 + s * A[ds][j]:.3f} | {auc1(s * Xte[:, j], yte):.3f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {s * spearmanr(Xte[:, j], te['score'])[0]:+.2f} |")
        for g, (z, ac, ci) in combo.items():
            L.append(f"| **combo top-{a.k}** | {g} (z-score mean) | - | **{ac:.3f}** | [{ci[0]:.2f}, {ci[1]:.2f}] | {spearmanr(z, te['score'])[0]:+.2f} |")
        # does the trajectory add information beyond classic confidence? (logistic fit on round 1, AUC on round 2)
        from sklearn.linear_model import LogisticRegression

        def scores(cols):
            mu, sd = Xtr[:, cols].mean(0), Xtr[:, cols].std(0) + 1e-9
            clf = LogisticRegression(C=0.3, max_iter=2000).fit((Xtr[:, cols] - mu) / sd, ytr)
            return clf.decision_function((Xte[:, cols] - mu) / sd)
        bc = [names.index(b) for b in ("base.conf_min", "base.conf_mean", "base.logp_mean") if b in names]
        tc = [int(j) for j in groups["consistent-traj"][:3]]
        pb, pbt, pt = scores(bc), scores(bc + tc), scores(tc)
        d = []
        for _ in range(1000):
            i = rng.integers(0, len(yte), len(yte))
            if 0 < yte[i].sum() < len(i):
                d.append(auc1(pbt[i], yte[i]) - auc1(pb[i], yte[i]))
        L.append(f"\n**Incremental value ({ds}, logistic fit on round 1, AUC on round 2):** confidence baselines only = {auc1(pb, yte):.3f}; "
                 f"top-3 consistent trajectory features only = {auc1(pt, yte):.3f}; baselines + trajectory = {auc1(pbt, yte):.3f}; "
                 f"paired-bootstrap difference (with - without trajectory) = {np.mean(d):+.3f} [95% CI {np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]. "
                 f"Trajectory features used: {', '.join('`' + names[j] + '`' for j in tc)}\n")
        L.append("")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "validation.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

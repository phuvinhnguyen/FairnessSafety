"""Which activation-trajectory descriptors relate to correctness?  (AUROC / Pearson / Spearman, with controls)

  python uq_dllm/analyze.py --out uq_dllm/results
Reads uq_dllm/out/<dataset>/*.pkl produced by run.py.  Everything here is cheap (no model needed).

Controls that matter when you screen ~800 features on a few dozen samples:
  * permutation null of the BEST |AUC-0.5| over all features (family-wise calibration of the selection)
  * split-half: choose features on one half, report AUC of the same features on the other half
  * length confound: partial Spearman after regressing out #answer tokens (and prompt length)
  * cross-dataset consistency: same sign of effect on every dataset
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr

LENS_STEPS = {"all": slice(0, 9), "early": slice(0, 3), "mid": slice(3, 6), "late": slice(6, 9), "last": slice(8, 9)}
LENS_LAYERS = {"all": slice(0, 6), "shallow": slice(0, 3), "deep": slice(3, 6)}


def split3(n):
    return {"all": np.arange(n), **dict(zip(("early", "mid", "late"), np.array_split(np.arange(n), 3)))}


def split_layers(n):
    return {"all": np.arange(n), **{f"q{i+1}": q for i, q in enumerate(np.array_split(np.arange(n), 4))}}


def featurize(F):
    f = {}
    for k, v in F.items():
        if k.startswith(("tr.", "dp.")):
            for sg, si in split3(v.shape[0]).items():
                for lg, li in split_layers(v.shape[1]).items():
                    f[f"{k}|{sg}|{lg}"] = v[np.ix_(si, li)].mean()
        elif k.startswith("sc."):
            for lg, li in split_layers(len(v)).items():
                f[f"{k}|{lg}"] = v[li].mean()
        elif k == "lens.first_layer":
            for sg, sl in LENS_STEPS.items():
                f[f"{k}|{sg}"] = v[sl].mean()
        elif k.startswith("lens.") and k != "lens.meta":
            for sg, ss in LENS_STEPS.items():
                for lg, ls in LENS_LAYERS.items():
                    f[f"{k}|{sg}|{lg}"] = v[ss, ls].mean()
        elif k.startswith("pd."):
            if np.ndim(v) == 1:
                for sg, si in split3(len(v)).items():
                    f[f"{k}|{sg}"] = v[si].mean()
        elif k.startswith("base."):
            f[k] = float(v)
    return f


def load(dataset, root):
    rows = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(f"{root}/{dataset}/*.pkl"))]
    if not rows:
        return None
    names = None
    X, meta = [], []
    for r in rows:
        f = featurize(r["feats"])
        names = names or sorted(f)
        X.append([f.get(n, np.nan) for n in names])
        meta.append((r["score"], r["label"], r["n_content"], r["P"], r["id"]))
    X = np.nan_to_num(np.array(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return dict(X=X, names=names, score=np.array([m[0] for m in meta]), label=np.array([m[1] if m[1] is not None else np.nan for m in meta]),
                n_content=np.array([m[2] for m in meta], float), P=np.array([m[3] for m in meta], float),
                ids=[m[4] for m in meta], rows=rows)


def binarize(d):
    """label = correct (gsm8k/arc); translation: chrF >= median (balanced), continuous chrF kept in d['score']."""
    y = d["label"].copy()
    if np.isnan(y).all():
        y = (d["score"] >= np.median(d["score"])).astype(float)
    return y.astype(int)


def auc_matrix(R, y):
    """R: ranks [N,F] (ties averaged). AUC = P(feature of positive > feature of negative)."""
    n1, n0 = y.sum(), (1 - y).sum()
    return (R.T @ y - n1 * (n1 + 1) / 2) / (n1 * n0)


def partial_spearman(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z.T])
    res = lambda a: a - Zr @ np.linalg.lstsq(Zr, a, rcond=None)[0]
    a, b = res(rx), res(ry)
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def analyse(d, rng, n_perm=300, n_split=60, top_k=5):
    y = binarize(d)
    X, names = d["X"], d["names"]
    keep = X.std(0) > 1e-12
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    R = np.apply_along_axis(rankdata, 0, X)
    auc = auc_matrix(R, y)
    # permutation null of max |AUC-.5| over all features
    null = np.array([np.abs(auc_matrix(R, rng.permutation(y)) - 0.5).max() for _ in range(n_perm)])
    # split-half: pick top_k on dev, evaluate (sign-aligned) on test
    test_auc, sel_count = [], {}
    for _ in range(n_split):
        idx = rng.permutation(len(y))
        dev, te = idx[: len(y) // 2], idx[len(y) // 2:]
        if min(y[dev].sum(), (1 - y[dev]).sum(), y[te].sum(), (1 - y[te]).sum()) < 2:
            continue
        a_dev = auc_matrix(np.apply_along_axis(rankdata, 0, X[dev]), y[dev])
        sel = np.argsort(-np.abs(a_dev - 0.5))[:top_k]
        a_te = auc_matrix(np.apply_along_axis(rankdata, 0, X[te]), y[te])
        test_auc.append(np.mean(0.5 + np.sign(a_dev[sel] - 0.5) * (a_te[sel] - 0.5)))
        for s in sel:
            sel_count[names[s]] = sel_count.get(names[s], 0) + 1
    # continuous association + length control for every feature (cheap: Pearson / Spearman vs score & label)
    sc = d["score"]
    sp_score = np.array([spearmanr(X[:, j], sc)[0] for j in range(X.shape[1])])
    pe_label = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
    return dict(names=names, auc=auc, y=y, null=null, split_test=np.array(test_auc), sel_count=sel_count,
                spearman_score=np.nan_to_num(sp_score), pearson_label=np.nan_to_num(pe_label), X=X, R=R)


def cell_arrays(d, key):
    return np.stack([r["feats"][key] for r in d["rows"]])          # [N, T, Lm]


def plots(d, y, name, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [m for m in ("d_ref", "vel", "acc", "turn", "mag_ref", "cos_prev") if f"tr.{m}.mean" in d["rows"][0]["feats"]]
    fig, ax = plt.subplots(2, len(metrics), figsize=(3.4 * len(metrics), 6.4))
    for j, m in enumerate(metrics):
        A = cell_arrays(d, f"tr.{m}.mean")                           # [N, T, L+1]
        R = np.apply_along_axis(rankdata, 0, A.reshape(len(y), -1))
        auc = auc_matrix(R, y).reshape(A.shape[1:])
        im = ax[0, j].imshow(auc.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=0.2, vmax=0.8)
        ax[0, j].set_title(f"AUC({m}) layer x step"); ax[0, j].set_xlabel("step"); ax[0, j].set_ylabel("layer")
        late = A[:, :, A.shape[2] // 2:].mean(2)                     # mean over upper half of layers
        t = np.arange(A.shape[1])
        for lab, col in ((1, "tab:green"), (0, "tab:red")):
            mu, se = late[y == lab].mean(0), late[y == lab].std(0) / np.sqrt(max(1, (y == lab).sum()))
            ax[1, j].plot(t, mu, color=col, label="correct/good" if lab else "wrong/poor"); ax[1, j].fill_between(t, mu - se, mu + se, color=col, alpha=.2)
        ax[1, j].set_title(f"{m}, upper-half layers"); ax[1, j].set_xlabel("step")
        if j == 0:
            ax[1, j].legend()
    plt.colorbar(im, ax=ax[0, :].tolist(), shrink=.7)
    fig.suptitle(f"{name}: N={len(y)} ({y.sum()} correct/good)")
    fig.savefig(outdir / f"traj_{name}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="uq_dllm/out")
    ap.add_argument("--out", default="uq_dllm/results")
    ap.add_argument("--datasets", default="gsm8k,wmt,arc")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res, lines = {}, []
    w = lines.append
    for ds in a.datasets.split(","):
        d = load(ds, a.root)
        if d is None:
            continue
        y = binarize(d)
        if min(y.sum(), (1 - y).sum()) < 5:
            w(f"## {ds}: skipped (only {y.sum()} positives of {len(y)})\n"); continue
        r = analyse(d, rng); r["d"] = d; res[ds] = r
        plots(d, y, ds, out)
        names, auc = r["names"], r["auc"]
        order = np.argsort(-np.abs(auc - 0.5))
        j_len = names.index("base.n_content")
        w(f"## {ds}: N={len(y)}, positives={y.sum()} ({y.mean():.0%}), features={len(names)}\n")
        w(f"- permutation null: 95th pct of best |AUC-0.5| over all features = {np.percentile(r['null'], 95):.3f}; "
          f"observed best = {np.abs(auc - 0.5).max():.3f}")
        w(f"- split-half (pick top-5 on one half, test on other): mean test AUC = {r['split_test'].mean():.3f} "
          f"(+-{r['split_test'].std():.3f}, {len(r['split_test'])} repeats; 0.5 = chance)")
        w(f"- length alone: AUC(n_content) = {auc[j_len]:.3f}\n")
        w("| rank | feature | AUC | Pearson(label) | Spearman(score) | partial Spearman | null p |\n|---|---|---|---|---|---|---|")
        Z = np.column_stack([d["n_content"], d["P"]])
        for rk, j in enumerate(order[:15], 1):
            ps = partial_spearman(r["X"][:, j], d["score"], Z)
            p = (np.sum(r["null"] >= abs(auc[j] - 0.5)) + 1) / (len(r["null"]) + 1)
            w(f"| {rk} | `{names[j]}` | {auc[j]:.3f} | {r['pearson_label'][j]:+.2f} | {r['spearman_score'][j]:+.2f} | {ps:+.2f} | {p:.3f} |")
        w("\nBaselines (classic UQ signals; AUC = P(correct has higher value)):\n")
        w("| baseline | AUC |\n|---|---|")
        for b in ("base.conf_mean", "base.conf_min", "base.conf_p10", "base.logp_mean", "base.ent_mean", "base.ent_max", "base.final_logp_mean",
                  "base.final_logp_min", "base.flips_mean", "base.frac_lowconf", "base.n_content"):
            if b in names:
                w(f"| {b} | {auc[names.index(b)]:.3f} |")
        w("")
    # cross-dataset consistency
    if len(res) >= 2:
        common = sorted(set.intersection(*[set(r["names"]) for r in res.values()]))
        A = np.array([[r["auc"][r["names"].index(n)] for n in common] for r in res.values()]) - 0.5
        sgn = np.sign(A.mean(0))
        al = A * sgn
        score = al.min(0)                                            # worst dataset, sign-aligned
        order = np.argsort(-score)
        w(f"## Cross-dataset consistency ({', '.join(res)}): features ranked by the WORST sign-aligned AUC-0.5 over datasets\n")
        w("| feature | " + " | ".join(f"AUC {k}" for k in res) + " | worst aligned |\n|" + "---|" * (len(res) + 2))
        for j in order[:20]:
            w(f"| `{common[j]}` | " + " | ".join(f"{0.5 + A[i, j]:.3f}" for i in range(len(res))) + f" | {score[j]:+.3f} |")
        w("")
    (out / "analysis.md").write_text("\n".join(lines))
    pickle.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("d",)} for k, v in res.items()}, open(out / "analysis.pkl", "wb"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

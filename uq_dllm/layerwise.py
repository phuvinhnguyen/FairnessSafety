"""Layer-resolved AUC of trajectory descriptors and logit-lens signals (pooled data).

  python uq_dllm/layerwise.py --root uq_dllm/pooled --out uq_dllm/results_pooled
AUC = P(correct/good answer has the HIGHER value); < 0.5 means 'lower in correct answers'.
Logit-lens layers are (8,12,16,20,24,28); 28 is the real output distribution, so 'lens layer 28' equals the
classic per-step confidence and anything better at layers 20/24 is genuinely extra information from the activations.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uq_dllm.analyze import auc_matrix, binarize, load  # noqa: E402
from uq_dllm.features import LENS_LAYERS  # noqa: E402


def boot(x, y, rng, B=500):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            r = rankdata(x[i]); n1 = y[i].sum(); n0 = len(i) - n1
            v.append((r[y[i] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return np.percentile(v, [2.5, 97.5])


def auc_cols(A, y):
    """A [N, K] -> AUC per column."""
    return auc_matrix(np.apply_along_axis(rankdata, 0, A), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="uq_dllm/pooled")
    ap.add_argument("--out", default="uq_dllm/results_pooled")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = ["# Layer-resolved AUC (pooled rounds)\n", "AUC = P(correct has higher value). <0.5: lower in correct answers.\n"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for c, ds in enumerate(("gsm8k", "wmt", "arc")):
        d = load(ds, a.root)
        y = binarize(d)
        rows = d["rows"]
        L.append(f"## {ds}: N={len(y)}, {y.sum()} correct/good\n")
        # --- logit lens per lens layer (mean over lens steps; and early third of steps)
        L.append("**Logit lens** (rows: metric; columns: lens layer; value = AUC over all 9 sampled steps / early steps only)\n")
        L.append("| metric | " + " | ".join(f"L{l}" for l in LENS_LAYERS) + " |\n|" + "---|" * (len(LENS_LAYERS) + 1))
        for m in ("maxp", "margin", "ent", "p_y", "top1_y", "kl_last"):
            A = np.stack([r["feats"][f"lens.{m}"] for r in rows])        # [N, 9, 6]
            allv, early = auc_cols(A.mean(1), y), auc_cols(A[:, :3].mean(1), y)
            L.append(f"| {m} | " + " | ".join(f"{p:.2f} / {q:.2f}" for p, q in zip(allv, early)) + " |")
            if m in ("maxp", "ent"):
                ax[0, c].plot(LENS_LAYERS, allv, "o-", label=f"lens {m} (all steps)")
                ax[0, c].plot(LENS_LAYERS, early, "s--", label=f"lens {m} (early steps)")
        L.append("")
        # --- trajectory metrics per layer (mean over steps in a window)
        L.append("**Trajectory descriptors per layer** (AUC; step window in brackets)\n")
        layers = list(range(0, 29, 2))
        L.append("| metric [window] | " + " | ".join(f"l{l}" for l in layers) + " |\n|" + "---|" * (len(layers) + 1))
        for m, win in (("d_ref", "all"), ("d_ref", "early"), ("cos_ref", "all"), ("mag_ref", "early"), ("vel", "early"), ("vel", "late"),
                       ("acc", "early"), ("turn", "all"), ("prog", "all")):
            A = np.stack([r["feats"][f"tr.{m}.mean"] for r in rows])     # [N, T, L+1]
            T = A.shape[1]
            sl = {"all": slice(0, T), "early": slice(0, T // 3), "late": slice(2 * T // 3, T)}[win]
            v = auc_cols(A[:, sl].mean(1), y)
            L.append(f"| {m} [{win}] | " + " | ".join(f"{v[l]:.2f}" for l in layers) + " |")
            if (m, win) in (("d_ref", "all"), ("vel", "early")):
                ax[1, c].plot(range(len(v)), v, label=f"{m} [{win}]")
        L.append("")
        for k, t in ((0, "logit lens AUC vs lens layer"), (1, "AUC vs layer")):
            ax[k, c].axhline(0.5, color="gray", lw=.8); ax[k, c].set_title(f"{ds}: {t}"); ax[k, c].set_xlabel("layer"); ax[k, c].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "layerwise.png", dpi=90)
    (out / "layerwise.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

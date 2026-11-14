"""Generate answers with the masked-diffusion LM, record activation trajectories, save features per sample.

  python uq_dllm/run.py --dataset gsm8k --n 20 --out uq_dllm/out
Resumable: finished samples (uq_dllm/out/<dataset>/<id>.pkl) are skipped.
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

from uq_dllm import tasks as D  # noqa: E402
from uq_dllm.features import trajectory_features  # noqa: E402
from uq_dllm.model import DiffusionQwen3  # noqa: E402
from uq_dllm.sampler import REPO, encode_prompt, generate, load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.SETTINGS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="uq_dllm/out")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--G", type=int), ap.add_argument("--steps", type=int), ap.add_argument("--block", type=int)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = dict(D.SETTINGS[a.dataset])
    for k in ("G", "steps", "block"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    out = Path(a.out) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load_tokenizer(a.repo), DiffusionQwen3(a.repo)
    for it in D.load(a.dataset, a.n, a.seed, a.offset):
        f = out / f"{it['id']}.pkl"
        if f.exists():
            continue
        t0 = time.time()
        rec = generate(model, tok, encode_prompt(tok, it["user"]), **cfg)
        text = tok.decode(rec["tokens"][:rec["n_content"]], skip_special_tokens=True)
        sc, lab = D.score(a.dataset, text, it["gold"])
        t1 = time.time()
        feats = trajectory_features(rec, model, tok)
        pickle.dump(dict(id=it["id"], text=text, gold=it["gold"], score=sc, label=lab, cfg=cfg, P=rec["P"],
                         n_content=rec["n_content"], feats=feats), open(f, "wb"))
        print(f"{a.dataset} {it['id']} score={sc:.2f} n_content={rec['n_content']} gen={t1-t0:.0f}s feat={time.time()-t1:.0f}s :: {text[:70]!r}", flush=True)


if __name__ == "__main__":
    main()

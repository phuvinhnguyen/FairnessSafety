"""Activation-trajectory descriptors for one generated answer.

Notation: h[s,l,t] = residual stream after layer l (l=0: embeddings), at denoising step s (s=S: clean final pass),
at answer position t.  ref[l,t] = h[S,l,t] ("the final answer").  Only answer positions before the first <|im_end|>
(the 'content' set C) are used.  All distances are RELATIVE (divided by a norm) so layers/models are comparable.

time-resolved (along denoising steps), per layer, aggregated over positions (mean / max):
  mag        |h|                         mag_ref   |h| / |ref|
  d_ref      |h - ref| / |ref|           cos_ref   cos(h, ref)            (change relative to the final answer)
  vel        |h_s - h_{s-1}| / |h_{s-1}| (velocity relative to the adjacent step)
  vel_ref    |h_s - h_{s-1}| / |ref|
  cos_prev   cos(h_s, h_{s-1})
  acc        |h_{s+1} - 2h_s + h_{s-1}| / |h_s|           jerk  (3rd difference, same normalisation)
  turn       angle(v_s, v_{s-1}) with v_s = h_s - h_{s-1}  (direction change of the trajectory)
  prog       cos(v_s, ref - h_{s-1})                        (is the motion heading to the final answer?)
per-trajectory scalars per layer: path (sum of |v|/|ref|), net (|ref-h_0|/|ref|), tort = path/net
depth-direction (along layers, at each step): dl_vel, dl_acc, dl_turn, dl_ref (|h_l - h_L|/|h_L|), dl_cos
logit lens at a few (steps x layers): entropy, max prob, p(final token), top1==final token, margin, KL(lens_l || lens_L)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6
LENS_LAYERS = (8, 12, 16, 20, 24, 28)


def _nrm(a):
    return a.norm(dim=-1)


def _cos(a, b):
    return (a * b).sum(-1) / (_nrm(a) * _nrm(b) + EPS)


def _ang(a, b):
    return torch.arccos(_cos(a, b).clamp(-1, 1))


def _agg(d, key, a):
    """a: [T, n] (positions last) -> store mean/max over positions as [T]."""
    d.setdefault(key + ".mean", []).append(a.mean(1).numpy())
    d.setdefault(key + ".max", []).append(a.max(1).values.numpy())


def trajectory_features(rec, model, tok, n_lens_pos=48, n_lens_steps=9):
    hid, S, G = rec["hidden"], rec["S"], rec["G"]          # [S+1, L+1, G, D]
    n = rec["n_content"]
    C = torch.arange(n)
    Lp1 = hid.shape[1]
    out = {}
    tr = {}
    for l in range(Lp1):
        X = hid[:, l, C, :]                                  # [S+1, n, D]
        ref = X[-1]
        nr = _nrm(ref) + EPS
        _agg(tr, "mag", _nrm(X))
        _agg(tr, "mag_ref", _nrm(X) / nr)
        _agg(tr, "d_ref", _nrm(X - ref) / nr)
        _agg(tr, "cos_ref", _cos(X, ref))
        v = X[1:] - X[:-1]                                   # [S, n, D]
        vn = _nrm(v)
        _agg(tr, "vel", vn / (_nrm(X[:-1]) + EPS))
        _agg(tr, "vel_ref", vn / nr)
        _agg(tr, "cos_prev", _cos(X[1:], X[:-1]))
        _agg(tr, "prog", _cos(v, ref - X[:-1]))
        a = v[1:] - v[:-1]                                   # [S-1, n, D]
        _agg(tr, "acc", _nrm(a) / (_nrm(X[1:-1]) + EPS))
        _agg(tr, "turn", _ang(v[1:], v[:-1]))
        j = a[1:] - a[:-1]
        _agg(tr, "jerk", _nrm(j) / (_nrm(X[2:-1]) + EPS))
        path = vn.sum(0) / nr
        net = _nrm(ref - X[0]) / nr
        for nm, val in (("path", path), ("net", net), ("tort", path / (net + EPS))):
            out.setdefault(f"sc.{nm}.mean", []).append(val.mean().numpy())
            out.setdefault(f"sc.{nm}.max", []).append(val.max().numpy())
    for k, v in tr.items():
        out["tr." + k] = np.stack(v, 1).astype(np.float32)   # [Tm, L+1]
    for k in [k for k in out if k.startswith("sc.")]:
        out[k] = np.array(out[k], dtype=np.float32)          # [L+1]

    dp = {}
    for s in range(S + 1):
        Y = hid[s][:, C, :]                                  # [L+1, n, D]
        Lf = Y[-1]
        dv = Y[1:] - Y[:-1]                                  # [L, n, D]
        _agg(dp, "dl_vel", _nrm(dv) / (_nrm(Y[:-1]) + EPS))
        da = dv[1:] - dv[:-1]
        _agg(dp, "dl_acc", _nrm(da) / (_nrm(Y[1:-1]) + EPS))
        _agg(dp, "dl_turn", _ang(dv[1:], dv[:-1]))
        _agg(dp, "dl_ref", _nrm(Y - Lf) / (_nrm(Lf) + EPS))
        _agg(dp, "dl_cos", _cos(Y, Lf))
    for k, v in dp.items():
        out["dp." + k] = np.stack(v, 0).astype(np.float32)   # [S+1, Lm]

    # ---- logit lens on a subset of (steps x layers x positions) -------------------------------
    y = torch.tensor(rec["tokens"][:n])
    pos = np.unique(np.linspace(0, n - 1, min(n, n_lens_pos)).astype(int))
    steps = np.unique(np.linspace(0, S, n_lens_steps).astype(int))
    ll = [l for l in LENS_LAYERS if l < Lp1]
    lens = {k: np.zeros((len(steps), len(ll)), np.float32) for k in ("ent", "maxp", "p_y", "top1_y", "margin", "kl_last")}
    first_layer = np.zeros(len(steps), np.float32)
    for i, s in enumerate(steps):
        P_last = None
        agree = []
        for j in range(len(ll) - 1, -1, -1):                 # last layer first -> reference distribution
            z = F.log_softmax(model.lens(hid[s, ll[j]][pos]).float(), -1)
            p = z.exp()
            if P_last is None:
                P_last, z_last = p, z
            top2 = p.topk(2, -1).values
            lens["ent"][i, j] = (-(p * z).sum(-1)).mean()
            lens["maxp"][i, j] = top2[:, 0].mean()
            lens["p_y"][i, j] = p[torch.arange(len(pos)), y[pos]].mean()
            lens["top1_y"][i, j] = (p.argmax(-1) == y[pos]).float().mean()
            lens["margin"][i, j] = (top2[:, 0] - top2[:, 1]).mean()
            lens["kl_last"][i, j] = (p * (z - z_last)).sum(-1).mean()
            agree.append((p.argmax(-1) == y[pos]))
        agree = torch.stack(agree[::-1])                     # [n_ll, n_pos] ordered shallow -> deep
        # shallowest lens layer from which the argmax equals the final token (and stays so)
        stay = torch.flip(torch.cummin(torch.flip(agree.int(), [0]), 0).values, [0]).bool()
        fl = torch.where(stay.any(0), stay.float().argmax(0).float(), torch.tensor(float(len(ll))))
        first_layer[i] = fl.mean()
    for k, v in lens.items():
        out["lens." + k] = v
    out["lens.first_layer"] = first_layer
    out["lens.meta"] = np.array([len(steps), len(ll)], dtype=np.float32)

    # ---- output-distribution trajectory (final layer) + commit / baseline statistics ----------
    yt = np.array(rec["tokens"][:n])
    top1, ent, am = rec["top1"][:, :n], rec["ent"][:, :n], rec["argmax"][:, :n]
    out["pd.top1.mean"], out["pd.top1.max"] = top1.mean(1), top1.max(1)
    out["pd.ent.mean"], out["pd.ent.max"] = ent.mean(1), ent.max(1)
    out["pd.agree"] = (am == yt[None]).mean(1)
    flips = (am[1:] != am[:-1]).sum(0)
    c_conf, c_ent, c_step = rec["c_conf"][:n], rec["c_ent"][:n], rec["c_step"][:n]
    fp = rec["fin_p"][:n]
    sc = {"base.conf_mean": c_conf.mean(), "base.conf_min": c_conf.min(), "base.conf_p10": np.percentile(c_conf, 10),
          "base.ent_mean": c_ent.mean(), "base.ent_max": c_ent.max(), "base.logp_mean": np.log(c_conf + 1e-9).mean(),
          "base.frac_lowconf": (c_conf < 0.5).mean(), "base.flips_mean": flips.mean(), "base.flips_max": flips.max(),
          "base.final_logp_mean": np.log(fp + 1e-9).mean(), "base.final_logp_min": np.log(fp + 1e-9).min(),
          "base.final_top1_mean": rec["fin_top1"][:n].mean(), "base.final_ent_mean": rec["fin_ent"][:n].mean(),
          "base.final_agree": (fp >= rec["fin_top1"][:n] - 1e-6).mean(), "base.commit_step_mean": c_step.mean() / S,
          "base.commit_step_std": c_step.std() / S, "base.n_content": float(n)}
    for k, v in sc.items():
        out[k] = np.float32(v)
    return out

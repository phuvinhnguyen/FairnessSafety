## gsm8k: N=80, positives=24 (30%), features=802

- permutation null: 95th pct of best |AUC-0.5| over all features = 0.263; observed best = 0.359
- split-half (pick top-5 on one half, test on other): mean test AUC = 0.718 (+-0.080, 60 repeats; 0.5 = chance)
- length alone: AUC(n_content) = 0.350

| rank | feature | AUC | Pearson(label) | Spearman(score) | partial Spearman | null p |
|---|---|---|---|---|---|---|
| 1 | `tr.d_ref.mean|all|q3` | 0.141 | -0.53 | -0.57 | -0.48 | 0.003 |
| 2 | `tr.cos_ref.mean|all|q3` | 0.844 | +0.52 | +0.55 | +0.47 | 0.003 |
| 3 | `tr.d_ref.mean|early|q3` | 0.163 | -0.51 | -0.54 | -0.51 | 0.003 |
| 4 | `tr.cos_ref.mean|mid|q3` | 0.836 | +0.51 | +0.53 | +0.46 | 0.003 |
| 5 | `tr.d_ref.mean|mid|q3` | 0.173 | -0.50 | -0.52 | -0.42 | 0.007 |
| 6 | `tr.d_ref.max|early|q3` | 0.198 | -0.46 | -0.48 | -0.48 | 0.017 |
| 7 | `tr.cos_ref.mean|early|q3` | 0.791 | +0.44 | +0.46 | +0.45 | 0.020 |
| 8 | `tr.d_ref.mean|all|all` | 0.222 | -0.40 | -0.44 | -0.37 | 0.037 |
| 9 | `tr.cos_ref.mean|mid|all` | 0.764 | +0.39 | +0.42 | +0.34 | 0.047 |
| 10 | `tr.d_ref.max|all|q3` | 0.238 | -0.41 | -0.42 | -0.35 | 0.063 |
| 11 | `tr.cos_ref.mean|all|all` | 0.762 | +0.40 | +0.42 | +0.34 | 0.063 |
| 12 | `tr.prog.max|mid|q4` | 0.761 | +0.40 | +0.41 | +0.41 | 0.063 |
| 13 | `tr.mag.mean|late|q3` | 0.760 | +0.44 | +0.41 | +0.29 | 0.063 |
| 14 | `tr.d_ref.max|mid|q3` | 0.240 | -0.42 | -0.41 | -0.38 | 0.063 |
| 15 | `tr.d_ref.max|early|q2` | 0.240 | -0.40 | -0.41 | -0.40 | 0.066 |

Baselines (classic UQ signals; AUC = P(correct has higher value)):

| baseline | AUC |
|---|---|
| base.conf_mean | 0.682 |
| base.conf_min | 0.650 |
| base.conf_p10 | 0.628 |
| base.logp_mean | 0.679 |
| base.ent_mean | 0.303 |
| base.ent_max | 0.328 |
| base.final_logp_mean | 0.449 |
| base.final_logp_min | 0.501 |
| base.flips_mean | 0.564 |
| base.frac_lowconf | 0.344 |
| base.n_content | 0.350 |

## wmt: N=120, positives=60 (50%), features=802

- permutation null: 95th pct of best |AUC-0.5| over all features = 0.196; observed best = 0.268
- split-half (pick top-5 on one half, test on other): mean test AUC = 0.646 (+-0.040, 60 repeats; 0.5 = chance)
- length alone: AUC(n_content) = 0.579

| rank | feature | AUC | Pearson(label) | Spearman(score) | partial Spearman | null p |
|---|---|---|---|---|---|---|
| 1 | `tr.cos_prev.max|all|q2` | 0.768 | +0.35 | +0.49 | +0.30 | 0.003 |
| 2 | `tr.cos_ref.max|early|q3` | 0.741 | +0.36 | +0.42 | +0.22 | 0.003 |
| 3 | `tr.cos_ref.max|early|q4` | 0.740 | +0.35 | +0.44 | +0.24 | 0.003 |
| 4 | `lens.maxp|all|deep` | 0.727 | +0.37 | +0.41 | +0.14 | 0.007 |
| 5 | `tr.vel_ref.mean|early|q4` | 0.275 | -0.39 | -0.41 | -0.27 | 0.007 |
| 6 | `sc.net.mean|q3` | 0.278 | -0.37 | -0.51 | -0.36 | 0.007 |
| 7 | `tr.cos_prev.max|all|q3` | 0.719 | +0.26 | +0.37 | +0.23 | 0.017 |
| 8 | `tr.vel_ref.mean|early|q3` | 0.282 | -0.37 | -0.38 | -0.24 | 0.017 |
| 9 | `tr.cos_prev.max|early|q2` | 0.718 | +0.32 | +0.36 | +0.12 | 0.017 |
| 10 | `lens.margin|all|deep` | 0.718 | +0.35 | +0.41 | +0.16 | 0.017 |
| 11 | `tr.cos_prev.max|early|q3` | 0.716 | +0.25 | +0.36 | +0.21 | 0.017 |
| 12 | `lens.maxp|early|deep` | 0.715 | +0.36 | +0.33 | +0.03 | 0.020 |
| 13 | `lens.ent|all|deep` | 0.285 | -0.36 | -0.38 | -0.11 | 0.020 |
| 14 | `tr.vel.mean|early|q3` | 0.286 | -0.37 | -0.38 | -0.22 | 0.023 |
| 15 | `tr.jerk.mean|early|q4` | 0.289 | -0.34 | -0.39 | -0.28 | 0.027 |

Baselines (classic UQ signals; AUC = P(correct has higher value)):

| baseline | AUC |
|---|---|
| base.conf_mean | 0.636 |
| base.conf_min | 0.672 |
| base.conf_p10 | 0.683 |
| base.logp_mean | 0.660 |
| base.ent_mean | 0.353 |
| base.ent_max | 0.298 |
| base.final_logp_mean | 0.474 |
| base.final_logp_min | 0.480 |
| base.flips_mean | 0.326 |
| base.frac_lowconf | 0.328 |
| base.n_content | 0.579 |

## arc: N=200, positives=154 (77%), features=802

- permutation null: 95th pct of best |AUC-0.5| over all features = 0.175; observed best = 0.294
- split-half (pick top-5 on one half, test on other): mean test AUC = 0.742 (+-0.048, 60 repeats; 0.5 = chance)
- length alone: AUC(n_content) = 0.435

| rank | feature | AUC | Pearson(label) | Spearman(score) | partial Spearman | null p |
|---|---|---|---|---|---|---|
| 1 | `lens.maxp|early|deep` | 0.794 | +0.43 | +0.43 | +0.40 | 0.003 |
| 2 | `lens.margin|all|deep` | 0.785 | +0.40 | +0.41 | +0.38 | 0.003 |
| 3 | `lens.margin|early|deep` | 0.780 | +0.40 | +0.41 | +0.38 | 0.003 |
| 4 | `lens.maxp|all|deep` | 0.775 | +0.40 | +0.40 | +0.37 | 0.003 |
| 5 | `lens.ent|early|deep` | 0.226 | -0.40 | -0.40 | -0.36 | 0.003 |
| 6 | `lens.margin|early|all` | 0.757 | +0.37 | +0.37 | +0.35 | 0.003 |
| 7 | `lens.margin|mid|deep` | 0.753 | +0.37 | +0.37 | +0.33 | 0.003 |
| 8 | `lens.maxp|early|all` | 0.750 | +0.37 | +0.36 | +0.35 | 0.003 |
| 9 | `pd.ent.max|mid` | 0.259 | -0.36 | -0.35 | -0.35 | 0.003 |
| 10 | `lens.maxp|mid|deep` | 0.739 | +0.36 | +0.35 | +0.30 | 0.003 |
| 11 | `pd.ent.mean|early` | 0.267 | -0.35 | -0.34 | -0.36 | 0.003 |
| 12 | `pd.top1.mean|early` | 0.728 | +0.34 | +0.33 | +0.34 | 0.003 |
| 13 | `pd.ent.mean|all` | 0.274 | -0.34 | -0.33 | -0.37 | 0.003 |
| 14 | `pd.top1.mean|all` | 0.725 | +0.33 | +0.33 | +0.36 | 0.003 |
| 15 | `lens.margin|all|all` | 0.723 | +0.32 | +0.33 | +0.29 | 0.003 |

Baselines (classic UQ signals; AUC = P(correct has higher value)):

| baseline | AUC |
|---|---|
| base.conf_mean | 0.704 |
| base.conf_min | 0.699 |
| base.conf_p10 | 0.716 |
| base.logp_mean | 0.706 |
| base.ent_mean | 0.296 |
| base.ent_max | 0.283 |
| base.final_logp_mean | 0.538 |
| base.final_logp_min | 0.591 |
| base.flips_mean | 0.321 |
| base.frac_lowconf | 0.471 |
| base.n_content | 0.435 |

## Cross-dataset consistency (gsm8k, wmt, arc): features ranked by the WORST sign-aligned AUC-0.5 over datasets

| feature | AUC gsm8k | AUC wmt | AUC arc | worst aligned |
|---|---|---|---|---|
| `lens.ent|all|deep` | 0.292 | 0.285 | 0.288 | +0.208 |
| `lens.margin|all|all` | 0.693 | 0.707 | 0.723 | +0.193 |
| `lens.maxp|all|all` | 0.707 | 0.692 | 0.696 | +0.192 |
| `lens.maxp|mid|deep` | 0.696 | 0.679 | 0.739 | +0.179 |
| `lens.margin|mid|deep` | 0.678 | 0.694 | 0.753 | +0.178 |
| `lens.maxp|all|deep` | 0.674 | 0.727 | 0.775 | +0.174 |
| `base.ent_max` | 0.328 | 0.298 | 0.283 | +0.172 |
| `lens.margin|all|deep` | 0.665 | 0.718 | 0.785 | +0.165 |
| `lens.ent|mid|deep` | 0.295 | 0.338 | 0.292 | +0.162 |
| `lens.margin|mid|all` | 0.718 | 0.660 | 0.696 | +0.160 |
| `base.logp_mean` | 0.679 | 0.660 | 0.706 | +0.160 |
| `base.conf_min` | 0.650 | 0.672 | 0.699 | +0.150 |
| `lens.maxp|late|deep` | 0.666 | 0.649 | 0.690 | +0.149 |
| `base.ent_mean` | 0.303 | 0.353 | 0.296 | +0.147 |
| `lens.ent|late|deep` | 0.315 | 0.349 | 0.359 | +0.141 |
| `pd.top1.mean|all` | 0.654 | 0.638 | 0.725 | +0.138 |
| `base.conf_mean` | 0.682 | 0.636 | 0.704 | +0.136 |
| `lens.margin|late|deep` | 0.632 | 0.644 | 0.699 | +0.132 |
| `pd.ent.mean|all` | 0.351 | 0.368 | 0.274 | +0.132 |
| `lens.maxp|mid|all` | 0.714 | 0.630 | 0.663 | +0.130 |

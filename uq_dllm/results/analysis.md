## gsm8k: N=40, positives=11 (28%), features=802

- permutation null: 95th pct of best |AUC-0.5| over all features = 0.375; observed best = 0.437
- split-half (pick top-5 on one half, test on other): mean test AUC = 0.706 (+-0.107, 60 repeats; 0.5 = chance)
- length alone: AUC(n_content) = 0.292

| rank | feature | AUC | Pearson(label) | Spearman(score) | partial Spearman | null p |
|---|---|---|---|---|---|---|
| 1 | `tr.d_ref.mean|all|q3` | 0.063 | -0.67 | -0.68 | -0.61 | 0.003 |
| 2 | `tr.d_ref.mean|early|q3` | 0.066 | -0.68 | -0.67 | -0.67 | 0.007 |
| 3 | `tr.d_ref.mean|mid|q3` | 0.107 | -0.60 | -0.61 | -0.52 | 0.030 |
| 4 | `tr.cos_ref.mean|all|q3` | 0.881 | +0.59 | +0.59 | +0.50 | 0.040 |
| 5 | `dp.dl_ref.mean|late|q2` | 0.122 | -0.57 | -0.58 | -0.61 | 0.053 |
| 6 | `tr.mag_ref.mean|early|q3` | 0.138 | -0.57 | -0.56 | -0.62 | 0.083 |
| 7 | `dp.dl_ref.mean|late|q3` | 0.141 | -0.54 | -0.56 | -0.60 | 0.113 |
| 8 | `tr.cos_ref.mean|early|q3` | 0.856 | +0.55 | +0.55 | +0.56 | 0.126 |
| 9 | `dp.dl_cos.mean|late|q2` | 0.850 | +0.48 | +0.54 | +0.55 | 0.153 |
| 10 | `sc.path.max|q2` | 0.157 | -0.47 | -0.53 | -0.50 | 0.186 |
| 11 | `dp.dl_ref.mean|all|q2` | 0.157 | -0.53 | -0.53 | -0.56 | 0.186 |
| 12 | `tr.prog.max|mid|q3` | 0.840 | +0.51 | +0.53 | +0.50 | 0.189 |
| 13 | `lens.maxp|last|all` | 0.837 | +0.54 | +0.52 | +0.52 | 0.206 |
| 14 | `tr.prog.max|all|q3` | 0.837 | +0.48 | +0.52 | +0.51 | 0.206 |
| 15 | `tr.d_ref.max|early|q3` | 0.163 | -0.49 | -0.52 | -0.54 | 0.206 |

Baselines (classic UQ signals; AUC = P(correct has higher value)):

| baseline | AUC |
|---|---|
| base.conf_mean | 0.712 |
| base.conf_min | 0.712 |
| base.conf_p10 | 0.639 |
| base.logp_mean | 0.705 |
| base.ent_mean | 0.254 |
| base.ent_max | 0.279 |
| base.final_logp_mean | 0.317 |
| base.final_logp_min | 0.350 |
| base.flips_mean | 0.621 |
| base.frac_lowconf | 0.329 |
| base.n_content | 0.292 |

## wmt: N=60, positives=30 (50%), features=802

- permutation null: 95th pct of best |AUC-0.5| over all features = 0.272; observed best = 0.313
- split-half (pick top-5 on one half, test on other): mean test AUC = 0.676 (+-0.077, 60 repeats; 0.5 = chance)
- length alone: AUC(n_content) = 0.648

| rank | feature | AUC | Pearson(label) | Spearman(score) | partial Spearman | null p |
|---|---|---|---|---|---|---|
| 1 | `tr.vel_ref.mean|early|q3` | 0.187 | -0.51 | -0.50 | -0.40 | 0.007 |
| 2 | `tr.prog.mean|early|q4` | 0.188 | -0.50 | -0.59 | -0.57 | 0.007 |
| 3 | `tr.vel.mean|early|q3` | 0.189 | -0.50 | -0.49 | -0.34 | 0.007 |
| 4 | `tr.vel_ref.mean|early|q4` | 0.194 | -0.52 | -0.50 | -0.35 | 0.010 |
| 5 | `tr.jerk.mean|early|q4` | 0.200 | -0.40 | -0.49 | -0.36 | 0.010 |
| 6 | `tr.cos_prev.mean|early|q3` | 0.796 | +0.46 | +0.46 | +0.36 | 0.013 |
| 7 | `tr.acc.mean|early|q4` | 0.207 | -0.39 | -0.47 | -0.33 | 0.013 |
| 8 | `tr.vel.mean|early|q4` | 0.209 | -0.49 | -0.46 | -0.33 | 0.013 |
| 9 | `tr.acc.mean|early|q3` | 0.216 | -0.41 | -0.43 | -0.27 | 0.020 |
| 10 | `sc.net.mean|q3` | 0.220 | -0.49 | -0.57 | -0.43 | 0.037 |
| 11 | `tr.jerk.mean|early|q3` | 0.224 | -0.40 | -0.42 | -0.25 | 0.040 |
| 12 | `tr.vel_ref.mean|early|all` | 0.226 | -0.44 | -0.43 | -0.39 | 0.043 |
| 13 | `tr.cos_prev.mean|early|q4` | 0.770 | +0.41 | +0.44 | +0.33 | 0.060 |
| 14 | `tr.cos_prev.max|early|q3` | 0.769 | +0.28 | +0.40 | +0.25 | 0.060 |
| 15 | `tr.cos_prev.max|all|q2` | 0.765 | +0.41 | +0.46 | +0.24 | 0.070 |

Baselines (classic UQ signals; AUC = P(correct has higher value)):

| baseline | AUC |
|---|---|
| base.conf_mean | 0.679 |
| base.conf_min | 0.748 |
| base.conf_p10 | 0.660 |
| base.logp_mean | 0.702 |
| base.ent_mean | 0.313 |
| base.ent_max | 0.293 |
| base.final_logp_mean | 0.397 |
| base.final_logp_min | 0.420 |
| base.flips_mean | 0.336 |
| base.frac_lowconf | 0.300 |
| base.n_content | 0.648 |

## arc: N=100, positives=72 (72%), features=802

- permutation null: 95th pct of best |AUC-0.5| over all features = 0.233; observed best = 0.281
- split-half (pick top-5 on one half, test on other): mean test AUC = 0.661 (+-0.071, 60 repeats; 0.5 = chance)
- length alone: AUC(n_content) = 0.447

| rank | feature | AUC | Pearson(label) | Spearman(score) | partial Spearman | null p |
|---|---|---|---|---|---|---|
| 1 | `lens.maxp|early|deep` | 0.781 | +0.40 | +0.44 | +0.41 | 0.003 |
| 2 | `lens.margin|early|deep` | 0.775 | +0.39 | +0.43 | +0.40 | 0.010 |
| 3 | `lens.margin|all|deep` | 0.760 | +0.38 | +0.41 | +0.37 | 0.010 |
| 4 | `lens.margin|mid|deep` | 0.754 | +0.38 | +0.40 | +0.36 | 0.013 |
| 5 | `lens.maxp|all|deep` | 0.751 | +0.37 | +0.39 | +0.36 | 0.020 |
| 6 | `lens.margin|early|all` | 0.739 | +0.34 | +0.37 | +0.35 | 0.033 |
| 7 | `lens.maxp|mid|deep` | 0.739 | +0.35 | +0.37 | +0.33 | 0.033 |
| 8 | `lens.margin|all|all` | 0.729 | +0.32 | +0.36 | +0.33 | 0.063 |
| 9 | `lens.margin|mid|all` | 0.727 | +0.32 | +0.35 | +0.33 | 0.066 |
| 10 | `pd.ent.max|mid` | 0.277 | -0.32 | -0.35 | -0.34 | 0.086 |
| 11 | `lens.ent|early|deep` | 0.278 | -0.32 | -0.34 | -0.30 | 0.090 |
| 12 | `lens.first_layer|early` | 0.292 | -0.34 | -0.33 | -0.31 | 0.159 |
| 13 | `lens.maxp|early|all` | 0.704 | +0.31 | +0.32 | +0.31 | 0.183 |
| 14 | `lens.maxp|all|all` | 0.702 | +0.29 | +0.31 | +0.28 | 0.186 |
| 15 | `lens.maxp|mid|all` | 0.701 | +0.27 | +0.31 | +0.29 | 0.193 |

Baselines (classic UQ signals; AUC = P(correct has higher value)):

| baseline | AUC |
|---|---|
| base.conf_mean | 0.679 |
| base.conf_min | 0.666 |
| base.conf_p10 | 0.671 |
| base.logp_mean | 0.679 |
| base.ent_mean | 0.332 |
| base.ent_max | 0.330 |
| base.final_logp_mean | 0.543 |
| base.final_logp_min | 0.587 |
| base.flips_mean | 0.340 |
| base.frac_lowconf | 0.464 |
| base.n_content | 0.447 |

## Cross-dataset consistency (gsm8k, wmt, arc): features ranked by the WORST sign-aligned AUC-0.5 over datasets

| feature | AUC gsm8k | AUC wmt | AUC arc | worst aligned |
|---|---|---|---|---|
| `lens.margin|all|deep` | 0.687 | 0.684 | 0.760 | +0.184 |
| `base.logp_mean` | 0.705 | 0.702 | 0.679 | +0.179 |
| `lens.maxp|all|deep` | 0.721 | 0.679 | 0.751 | +0.179 |
| `base.conf_mean` | 0.712 | 0.679 | 0.679 | +0.179 |
| `base.ent_max` | 0.279 | 0.293 | 0.330 | +0.170 |
| `base.ent_mean` | 0.254 | 0.313 | 0.332 | +0.168 |
| `base.conf_min` | 0.712 | 0.748 | 0.666 | +0.166 |
| `lens.margin|all|all` | 0.787 | 0.661 | 0.729 | +0.161 |
| `lens.margin|mid|deep` | 0.705 | 0.653 | 0.754 | +0.153 |
| `pd.top1.mean|all` | 0.652 | 0.643 | 0.693 | +0.143 |
| `lens.ent|all|deep` | 0.241 | 0.358 | 0.344 | +0.142 |
| `base.conf_p10` | 0.639 | 0.660 | 0.671 | +0.139 |
| `lens.maxp|mid|deep` | 0.746 | 0.639 | 0.739 | +0.139 |
| `lens.ent|mid|deep` | 0.235 | 0.366 | 0.307 | +0.134 |
| `lens.maxp|all|all` | 0.809 | 0.632 | 0.702 | +0.132 |
| `tr.mag_ref.mean|early|q3` | 0.138 | 0.369 | 0.348 | +0.131 |
| `lens.maxp|early|deep` | 0.627 | 0.644 | 0.781 | +0.127 |
| `tr.mag_ref.mean|mid|all` | 0.373 | 0.361 | 0.323 | +0.127 |
| `dp.dl_ref.mean|late|q2` | 0.122 | 0.324 | 0.376 | +0.124 |
| `sc.net.mean|q3` | 0.245 | 0.220 | 0.380 | +0.120 |

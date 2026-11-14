# Round-1 selection -> round-2 (new samples) validation

## gsm8k: train N=40 (28% pos) -> test N=40 (32% pos)

| group | feature | AUC train (selected on it) | AUC test | test 95% CI | Spearman(test, score) |
|---|---|---|---|---|---|
| per-dataset | `tr.d_ref.mean|all|q3` (-) | 0.937 | 0.801 | [0.62, 0.95] | +0.49 |
| per-dataset | `tr.d_ref.mean|early|q3` (-) | 0.934 | 0.746 | [0.56, 0.91] | +0.40 |
| per-dataset | `tr.d_ref.mean|mid|q3` (-) | 0.893 | 0.778 | [0.60, 0.93] | +0.45 |
| per-dataset | `tr.cos_ref.mean|all|q3` (+) | 0.881 | 0.812 | [0.66, 0.94] | +0.51 |
| per-dataset | `dp.dl_ref.mean|late|q2` (-) | 0.878 | 0.558 | [0.35, 0.77] | +0.09 |
| per-dataset | `tr.mag_ref.mean|early|q3` (-) | 0.862 | 0.667 | [0.47, 0.85] | +0.27 |
| per-dataset | `dp.dl_ref.mean|late|q3` (-) | 0.859 | 0.598 | [0.40, 0.80] | +0.16 |
| per-dataset | `tr.cos_ref.mean|early|q3` (+) | 0.856 | 0.718 | [0.54, 0.88] | +0.35 |
| per-dataset | `dp.dl_cos.mean|late|q2` (+) | 0.850 | 0.627 | [0.46, 0.79] | +0.21 |
| per-dataset | `sc.path.max|q2` (-) | 0.843 | 0.504 | [0.29, 0.71] | +0.01 |
| consistent | `lens.margin|all|deep` (+) | 0.687 | 0.647 | [0.43, 0.85] | +0.24 |
| consistent | `base.logp_mean` (+) | 0.705 | 0.667 | [0.47, 0.84] | +0.27 |
| consistent | `lens.maxp|all|deep` (+) | 0.721 | 0.632 | [0.43, 0.83] | +0.22 |
| consistent | `base.conf_mean` (+) | 0.712 | 0.661 | [0.47, 0.82] | +0.26 |
| consistent | `base.ent_max` (-) | 0.721 | 0.615 | [0.42, 0.80] | +0.19 |
| consistent | `base.ent_mean` (-) | 0.746 | 0.667 | [0.49, 0.83] | +0.27 |
| consistent | `base.conf_min` (+) | 0.712 | 0.575 | [0.39, 0.76] | +0.12 |
| consistent | `lens.margin|all|all` (+) | 0.787 | 0.598 | [0.40, 0.80] | +0.16 |
| consistent | `lens.margin|mid|deep` (+) | 0.705 | 0.650 | [0.45, 0.83] | +0.24 |
| consistent | `pd.top1.mean|all` (+) | 0.652 | 0.658 | [0.48, 0.81] | +0.26 |
| consistent-traj | `tr.mag_ref.mean|early|q3` (-) | 0.862 | 0.667 | [0.46, 0.85] | +0.27 |
| consistent-traj | `tr.mag_ref.mean|mid|all` (-) | 0.627 | 0.621 | [0.43, 0.81] | +0.20 |
| consistent-traj | `dp.dl_ref.mean|late|q2` (-) | 0.878 | 0.558 | [0.36, 0.75] | +0.09 |
| consistent-traj | `sc.net.mean|q3` (-) | 0.755 | 0.715 | [0.53, 0.87] | +0.35 |
| consistent-traj | `dp.dl_ref.mean|late|q3` (-) | 0.859 | 0.598 | [0.41, 0.79] | +0.16 |
| consistent-traj | `dp.dl_ref.mean|late|q1` (-) | 0.831 | 0.573 | [0.37, 0.78] | +0.12 |
| consistent-traj | `tr.prog.max|early|q4` (+) | 0.705 | 0.330 | [0.15, 0.52] | -0.28 |
| consistent-traj | `sc.path.max|q3` (-) | 0.755 | 0.524 | [0.31, 0.74] | +0.04 |
| consistent-traj | `dp.dl_cos.mean|late|q3` (+) | 0.824 | 0.641 | [0.46, 0.82] | +0.23 |
| consistent-traj | `dp.dl_ref.mean|all|q1` (-) | 0.803 | 0.556 | [0.36, 0.75] | +0.09 |
| baseline | `base.conf_min` (+) | 0.712 | 0.575 | [0.41, 0.74] | +0.12 |
| baseline | `base.conf_mean` (+) | 0.712 | 0.661 | [0.47, 0.83] | +0.26 |
| baseline | `base.logp_mean` (+) | 0.705 | 0.667 | [0.48, 0.85] | +0.27 |
| baseline | `base.ent_max` (-) | 0.721 | 0.615 | [0.43, 0.81] | +0.19 |
| baseline | `base.ent_mean` (-) | 0.746 | 0.667 | [0.48, 0.83] | +0.27 |
| baseline | `base.final_logp_mean` (+) | 0.317 | 0.570 | [0.36, 0.77] | +0.11 |
| baseline | `base.flips_mean` (-) | 0.379 | 0.496 | [0.28, 0.69] | -0.01 |
| **combo top-10** | per-dataset (z-score mean) | - | **0.732** | [0.55, 0.90] | +0.38 |
| **combo top-10** | consistent (z-score mean) | - | **0.658** | [0.46, 0.83] | +0.26 |
| **combo top-10** | consistent-traj (z-score mean) | - | **0.647** | [0.44, 0.82] | +0.24 |

**Incremental value (gsm8k, logistic fit on round 1, AUC on round 2):** confidence baselines only = 0.624; top-3 consistent trajectory features only = 0.672; baselines + trajectory = 0.681; paired-bootstrap difference (with - without trajectory) = +0.051 [95% CI -0.129, +0.231]. Trajectory features used: `tr.mag_ref.mean|early|q3`, `tr.mag_ref.mean|mid|all`, `dp.dl_ref.mean|late|q2`


## wmt: train N=60 (50% pos) -> test N=60 (50% pos)

| group | feature | AUC train (selected on it) | AUC test | test 95% CI | Spearman(test, score) |
|---|---|---|---|---|---|
| per-dataset | `tr.vel_ref.mean|early|q3` (-) | 0.813 | 0.604 | [0.44, 0.75] | +0.26 |
| per-dataset | `tr.prog.mean|early|q4` (-) | 0.812 | 0.528 | [0.38, 0.67] | +0.16 |
| per-dataset | `tr.vel.mean|early|q3` (-) | 0.811 | 0.599 | [0.45, 0.74] | +0.26 |
| per-dataset | `tr.vel_ref.mean|early|q4` (-) | 0.806 | 0.639 | [0.49, 0.78] | +0.32 |
| per-dataset | `tr.jerk.mean|early|q4` (-) | 0.800 | 0.622 | [0.48, 0.76] | +0.31 |
| per-dataset | `tr.cos_prev.mean|early|q3` (+) | 0.796 | 0.591 | [0.43, 0.74] | +0.25 |
| per-dataset | `tr.acc.mean|early|q4` (-) | 0.793 | 0.627 | [0.48, 0.77] | +0.33 |
| per-dataset | `tr.vel.mean|early|q4` (-) | 0.791 | 0.610 | [0.45, 0.75] | +0.28 |
| per-dataset | `tr.acc.mean|early|q3` (-) | 0.784 | 0.614 | [0.47, 0.76] | +0.27 |
| per-dataset | `sc.net.mean|q3` (-) | 0.780 | 0.677 | [0.53, 0.82] | +0.45 |
| consistent | `lens.margin|all|deep` (+) | 0.684 | 0.743 | [0.61, 0.86] | +0.48 |
| consistent | `base.logp_mean` (+) | 0.702 | 0.594 | [0.44, 0.74] | +0.12 |
| consistent | `lens.maxp|all|deep` (+) | 0.679 | 0.763 | [0.64, 0.88] | +0.50 |
| consistent | `base.conf_mean` (+) | 0.679 | 0.572 | [0.42, 0.71] | +0.09 |
| consistent | `base.ent_max` (-) | 0.707 | 0.669 | [0.52, 0.80] | +0.30 |
| consistent | `base.ent_mean` (-) | 0.687 | 0.590 | [0.44, 0.75] | +0.14 |
| consistent | `base.conf_min` (+) | 0.748 | 0.631 | [0.48, 0.78] | +0.22 |
| consistent | `lens.margin|all|all` (+) | 0.661 | 0.762 | [0.63, 0.88] | +0.47 |
| consistent | `lens.margin|mid|deep` (+) | 0.653 | 0.707 | [0.57, 0.84] | +0.38 |
| consistent | `pd.top1.mean|all` (+) | 0.643 | 0.592 | [0.45, 0.75] | +0.10 |
| consistent-traj | `tr.mag_ref.mean|early|q3` (-) | 0.631 | 0.489 | [0.34, 0.65] | +0.06 |
| consistent-traj | `tr.mag_ref.mean|mid|all` (-) | 0.639 | 0.462 | [0.32, 0.63] | +0.03 |
| consistent-traj | `dp.dl_ref.mean|late|q2` (-) | 0.676 | 0.536 | [0.38, 0.71] | +0.18 |
| consistent-traj | `sc.net.mean|q3` (-) | 0.780 | 0.677 | [0.53, 0.81] | +0.45 |
| consistent-traj | `dp.dl_ref.mean|late|q3` (-) | 0.654 | 0.500 | [0.36, 0.65] | +0.11 |
| consistent-traj | `dp.dl_ref.mean|late|q1` (-) | 0.657 | 0.537 | [0.39, 0.69] | +0.23 |
| consistent-traj | `tr.prog.max|early|q4` (+) | 0.608 | 0.631 | [0.49, 0.76] | +0.34 |
| consistent-traj | `sc.path.max|q3` (-) | 0.609 | 0.663 | [0.52, 0.79] | +0.22 |
| consistent-traj | `dp.dl_cos.mean|late|q3` (+) | 0.651 | 0.514 | [0.36, 0.66] | +0.20 |
| consistent-traj | `dp.dl_ref.mean|all|q1` (-) | 0.641 | 0.537 | [0.38, 0.68] | +0.23 |
| baseline | `base.conf_min` (+) | 0.748 | 0.631 | [0.49, 0.78] | +0.22 |
| baseline | `base.conf_mean` (+) | 0.679 | 0.572 | [0.41, 0.71] | +0.09 |
| baseline | `base.logp_mean` (+) | 0.702 | 0.594 | [0.45, 0.74] | +0.12 |
| baseline | `base.ent_max` (-) | 0.707 | 0.669 | [0.52, 0.80] | +0.30 |
| baseline | `base.ent_mean` (-) | 0.687 | 0.590 | [0.44, 0.74] | +0.14 |
| baseline | `base.final_logp_mean` (+) | 0.397 | 0.559 | [0.41, 0.70] | +0.13 |
| baseline | `base.flips_mean` (-) | 0.664 | 0.678 | [0.52, 0.81] | +0.28 |
| **combo top-10** | per-dataset (z-score mean) | - | **0.616** | [0.47, 0.75] | +0.31 |
| **combo top-10** | consistent (z-score mean) | - | **0.691** | [0.56, 0.82] | +0.31 |
| **combo top-10** | consistent-traj (z-score mean) | - | **0.579** | [0.44, 0.72] | +0.30 |

**Incremental value (wmt, logistic fit on round 1, AUC on round 2):** confidence baselines only = 0.633; top-3 consistent trajectory features only = 0.526; baselines + trajectory = 0.604; paired-bootstrap difference (with - without trajectory) = -0.030 [95% CI -0.088, +0.032]. Trajectory features used: `tr.mag_ref.mean|early|q3`, `tr.mag_ref.mean|mid|all`, `dp.dl_ref.mean|late|q2`


## arc: train N=100 (72% pos) -> test N=100 (82% pos)

| group | feature | AUC train (selected on it) | AUC test | test 95% CI | Spearman(test, score) |
|---|---|---|---|---|---|
| per-dataset | `lens.maxp|early|deep` (+) | 0.781 | 0.811 | [0.67, 0.91] | +0.41 |
| per-dataset | `lens.margin|early|deep` (+) | 0.775 | 0.781 | [0.64, 0.89] | +0.37 |
| per-dataset | `lens.margin|all|deep` (+) | 0.760 | 0.832 | [0.72, 0.92] | +0.44 |
| per-dataset | `lens.margin|mid|deep` (+) | 0.754 | 0.764 | [0.64, 0.88] | +0.35 |
| per-dataset | `lens.maxp|all|deep` (+) | 0.751 | 0.823 | [0.71, 0.92] | +0.43 |
| per-dataset | `lens.margin|early|all` (+) | 0.739 | 0.781 | [0.65, 0.89] | +0.37 |
| per-dataset | `lens.maxp|mid|deep` (+) | 0.739 | 0.755 | [0.63, 0.87] | +0.34 |
| per-dataset | `lens.margin|all|all` (+) | 0.729 | 0.741 | [0.60, 0.86] | +0.32 |
| per-dataset | `lens.margin|mid|all` (+) | 0.727 | 0.675 | [0.53, 0.81] | +0.23 |
| per-dataset | `pd.ent.max|mid` (-) | 0.723 | 0.773 | [0.65, 0.87] | +0.36 |
| consistent | `lens.margin|all|deep` (+) | 0.760 | 0.832 | [0.72, 0.93] | +0.44 |
| consistent | `base.logp_mean` (+) | 0.679 | 0.750 | [0.63, 0.86] | +0.33 |
| consistent | `lens.maxp|all|deep` (+) | 0.751 | 0.823 | [0.72, 0.92] | +0.43 |
| consistent | `base.conf_mean` (+) | 0.679 | 0.746 | [0.62, 0.85] | +0.33 |
| consistent | `base.ent_max` (-) | 0.670 | 0.799 | [0.68, 0.90] | +0.40 |
| consistent | `base.ent_mean` (-) | 0.668 | 0.760 | [0.64, 0.86] | +0.35 |
| consistent | `base.conf_min` (+) | 0.666 | 0.763 | [0.64, 0.87] | +0.35 |
| consistent | `lens.margin|all|all` (+) | 0.729 | 0.741 | [0.61, 0.86] | +0.32 |
| consistent | `lens.margin|mid|deep` (+) | 0.754 | 0.764 | [0.64, 0.87] | +0.35 |
| consistent | `pd.top1.mean|all` (+) | 0.693 | 0.776 | [0.65, 0.88] | +0.37 |
| consistent-traj | `tr.mag_ref.mean|early|q3` (-) | 0.652 | 0.430 | [0.28, 0.58] | -0.09 |
| consistent-traj | `tr.mag_ref.mean|mid|all` (-) | 0.677 | 0.646 | [0.50, 0.78] | +0.19 |
| consistent-traj | `dp.dl_ref.mean|late|q2` (-) | 0.624 | 0.486 | [0.35, 0.62] | -0.02 |
| consistent-traj | `sc.net.mean|q3` (-) | 0.620 | 0.451 | [0.31, 0.61] | -0.06 |
| consistent-traj | `dp.dl_ref.mean|late|q3` (-) | 0.616 | 0.432 | [0.28, 0.57] | -0.09 |
| consistent-traj | `dp.dl_ref.mean|late|q1` (-) | 0.613 | 0.485 | [0.35, 0.63] | -0.02 |
| consistent-traj | `tr.prog.max|early|q4` (+) | 0.610 | 0.605 | [0.45, 0.75] | +0.14 |
| consistent-traj | `sc.path.max|q3` (-) | 0.606 | 0.611 | [0.47, 0.75] | +0.15 |
| consistent-traj | `dp.dl_cos.mean|late|q3` (+) | 0.601 | 0.377 | [0.26, 0.50] | -0.16 |
| consistent-traj | `dp.dl_ref.mean|all|q1` (-) | 0.599 | 0.527 | [0.39, 0.66] | +0.04 |
| baseline | `base.conf_min` (+) | 0.666 | 0.763 | [0.63, 0.88] | +0.35 |
| baseline | `base.conf_mean` (+) | 0.679 | 0.746 | [0.62, 0.87] | +0.33 |
| baseline | `base.logp_mean` (+) | 0.679 | 0.750 | [0.63, 0.86] | +0.33 |
| baseline | `base.ent_max` (-) | 0.670 | 0.799 | [0.69, 0.89] | +0.40 |
| baseline | `base.ent_mean` (-) | 0.668 | 0.760 | [0.65, 0.87] | +0.35 |
| baseline | `base.final_logp_mean` (+) | 0.543 | 0.537 | [0.37, 0.70] | +0.05 |
| baseline | `base.flips_mean` (-) | 0.660 | 0.701 | [0.56, 0.83] | +0.27 |
| **combo top-10** | per-dataset (z-score mean) | - | **0.802** | [0.69, 0.90] | +0.40 |
| **combo top-10** | consistent (z-score mean) | - | **0.825** | [0.72, 0.92] | +0.43 |
| **combo top-10** | consistent-traj (z-score mean) | - | **0.493** | [0.35, 0.64] | -0.01 |

**Incremental value (arc, logistic fit on round 1, AUC on round 2):** confidence baselines only = 0.764; top-3 consistent trajectory features only = 0.562; baselines + trajectory = 0.661; paired-bootstrap difference (with - without trajectory) = -0.101 [95% CI -0.203, -0.010]. Trajectory features used: `tr.mag_ref.mean|early|q3`, `tr.mag_ref.mean|mid|all`, `dp.dl_ref.mean|late|q2`


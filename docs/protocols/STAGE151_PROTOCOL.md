# Stage 151: conditional directional modality supervision

2026-11-11. Adaptive DEVELOPMENT follow-up to Stage150, whose primary excess-risk penalty failed the simple-residual and true-vs-shuffle tests. All old and new holdouts are historical. No claim of fresh confirmation or population safety.

## Mechanism

A scalar opposition flag discards which modality differs, its direction, and its magnitude. Instead learn modality-specific sentiment projections with independent human u_T,u_A,u_V supervision. Preserve differences between clips at the SAME source and EXACT overall score, rather than making all modality representations agree. For cell s, center both auxiliary predictions and modality targets within s: L_cond=mean_i,m |(u_hat_im-mean_s u_hat_m)-(u_im-mean_s u_m)|, restricted to cells with at least two FIT examples. Singleton cells get zero conditional loss and still receive ordinary task/auxiliary supervision. The overall rating is an independent target. Both auxiliary objectives act on the modality encoders; inference consumes only modality features.

Primary objective is L_task + .25 L_uni + .25 L_cond. This is a candidate training objective, not a proved novel architecture or a no-harm guarantee. Within-cell supervision removes additive source-score means at the loss level, not all feature confounding. Centered L1 is not identical to an all-pairs L1 loss. The full formula and implementation must use the centered form consistently.

## Factorial comparison

- Two FIT-only common modality interfaces: PCA 8 per modality (24 total) and PCA 32 per modality (96 total). Shared raw mean/std pooling and normalization as Stage123. Fit projections once per outer fold; rank is not selected on outer scores.
- Original three source-disjoint folds of v1, v2, UniC; seeds 17,43,89,131,197. Three modality GELU encoders each map to width32; concatenated task head 96->47->1. Three linear auxiliary heads and one conflict-logit head exist in every arm, even when their objective is disabled. Same per-arm initialization, 100 full-batch AdamW steps at .003, decay .001. Select each arm's checkpoint on its OWN ordinary TUNE MAE (same budget; no TUNE group endpoint). Save full optimizer state and all selected epochs.
- Arms: task_only; ordinary uni_mtl (.25); conflict-weighted uni_mtl (mean task + .5 mean conflict task); binary_conflict (.25 BCE in addition to uni_mtl); centered_only (.25 centered, no uni); conditional_direction (.25 uni + .25 centered); conditional_low (.25 uni + .0625 centered); conditional_high (.25 uni + 1 centered); uni_shuffled (.25 shuffled uni); and 19 conditional_direction replicas with unimodal target vectors shuffled jointly within FIT source/exact-y cells. Primary is conditional_direction, never the best outer arm.
- Exact-cell shuffles preserve each cell's target multiset and source/score relation; report changed proportions and coverage. They test sample assignment, not conditional exchangeability or causal effects.
- Main contrasts are primary vs ordinary uni_mtl (extra mechanism), vs binary_conflict (directional information vs binary flag), vs task_only (additional supervision), and vs conditionally shuffled primary. Report both groups, full/matched populations, outside-overlap group, overall MAE, five seeds and 3000 crossed source/seed/shuffle intervals.
- All 90 banks and their selected checkpoints are sealed before holdout scoring. 90*28=2520 shallow-network arms; count does not imply 2520 independent datasets or large encoder runs. Genuine original checkpoints are not overwritten. Resume only compatible complete receipts/checkpoints. Validate power_guard save/exit handshake before this controller starts.

If improvements are limited to one rank/cohort or disappear versus ordinary multi-task supervision, retain the conditional mechanism as exploratory and do not advertise a solved method. A modern prior-art reproduction and fresh source holdout remain separate work.

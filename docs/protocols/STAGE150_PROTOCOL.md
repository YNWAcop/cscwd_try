# Stage 150: cross-fitted, matched excess-risk correction

Date: 2026-11-11. Authorized by the current user request to implement and test algorithmic improvements. This is a new development study; none of v1, v2, or UniC is a pristine confirmation set. The former overnight deadline and shutdown authorization are not renewed.

## Hypothesis and distinction

Absolute-loss group DRO can prioritize irreducibly difficult examples and shift errors to unsupported nonconflict examples. Test a frozen, source-cross-fitted ordinary reference and bounded neural correction. The reference errors are out-of-source on the correction training set, unlike previous in-sample gradient-projection experiments. Compare penalizing positive individual excess error against penalizing only aggregate excess error. Source-and-exact-score matching limits the conflict incentive to comparable examples; the harm term covers ALL nonconflict FIT examples, including examples outside matching support.

For FIT sample i use OOF reference b_i and prediction f_i=clip(b_i+0.25*tanh(r(x_i,b_i)),-1,1). At inference use the mean of the three reference predictors and the same residual head. Let l_i=|f_i-y_i|, l0_i=|b_i-y_i|. Primary objective: mean(l)+0.5 sum(a_i l_i)+2 sum(n_i max(l_i-l0_i,0)). The a_i are normalized overlap weights on conflict examples within mixed source/exact-score cells (cell mass nC*nN/(nC+nN), divided by nC per conflict example); n_i are normalized weights over all nonconflict examples. No matching cell gives a zero conflict incentive, not fabricated support. Inference needs no human rating, group flag or source identifier.

These are empirical penalties, NOT a population no-harm guarantee. Positive regret can penalize useful corrections too. Training with OOF references and inference with their mean introduces a distribution change that must be measured. Matching and conditional shuffling do not establish causal exchangeability given features.

## Locked design

- Reuse the original three source-disjoint FIT/TUNE/holdout folds and the verified Stage123 common 24-dimensional modality-wise PCA inputs. FIT normalization/PCA is fixed; the inner cross-fit excludes outcome labels, but shares FIT-only unsupervised preprocessing.
- Five initialization seeds: 17,43,89,131,197. For each FIT, assign source groups deterministically to three balanced inner folds. Train three ordinary 47-unit GELU heads for exactly 100 full-batch AdamW steps (lr .003, decay .001); never select them on TUNE. Produce OOF predictions for FIT and ensemble predictions for TUNE/holdout. All comparison arms share these references.
- Train residual heads (47 GELU units, zero last layer, correction capped at .25) for 100 steps with AdamW at .003/.001. Same initialization in each bank. Select a common epoch using ordinary residual TUNE MAE; store every arm at that same epoch. TUNE/holdout conflict labels never enter training or checkpoint selection.
- Arms: ordinary residual; 0.5 unbalanced conflict incentive; matched incentive alone; matched mean-regret penalty; unbalanced positive-regret penalty; matched positive-regret beta .5,2,8; matched squared-displacement control; matched source-mean positive regret; 19 independently exact-cell shuffled versions of the primary objective. Frozen reference is a zero-fit comparator. All arms and negative results retained. Primary is matched positive regret beta=2, not the later winner.
- All cohorts/folds/seeds finish and receive a hash seal before ANY holdout metric is computed. Save completed reference and correction checkpoints; no repeated training on restart. Metadata and predicted values may be materialized before sealing but no performance selection on holdout is allowed.
- Report overall MAE, full conflict/nonconflict gains, outside-support nonconflict gain, matched G_C/G_N/A, per-seed directions, individual harm rate and magnitude. Comparators: frozen ordinary reference, ordinary residual, simple conflict incentive, mean-regret, displacement control, matched shuffled primary. Use 3000 crossed source/seed bootstrap draws, and independently resample shuffle replicates for the null comparison. Conditional intervals are exploratory, not multiplicity-corrected confirmations or new training-corpus uncertainty.

## Diagnostics and integrity

After the prediction seal, fit score-only, features-plus-score, and within-cell conditional ranking diagnostics of c with FIT-only training and source-disjoint scoring. Report AUC/Brier/calibration and matched within-cell discrimination; they diagnose residual association, not prove exchangeability. Preserve seeds and source manifests, data hashes, all failures, optimizer state, receipts and exact replay.

Before training, validate the existing power_guard boundary with an isolated new shutdown request and a saved checkpoint; honor cancellation and NEW requests. The historical consumed request must pass its old acknowledgment hash/PID/expiry checks. Current request authorizes research only. CPU ready is false during writes, and true with the existing authorization_id only after all owned writes finish. No shutdown timer, reboot, quota change or account operations.

Promotion is separate from exploration: retain a candidate with coherent incremental benefit, but do not replace the paper's claims without comparison to simple controls, cross-setting robustness and fresh confirmation. No numerical acceptance probability can be certified by this experiment.

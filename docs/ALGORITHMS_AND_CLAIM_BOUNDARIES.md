# Algorithm definitions and claim boundaries

## Original algorithm: conditional KL reweighting audit

Inputs: fixed source-disjoint FIT/TUNE/TEST folds; modality features X; overall targets y; independent modality targets U; FIT normalization; signed bands; KL radius .04; ordinary-selected hyperparameters.

1. Compute the human opposition flag only for training grouping and offline evaluation: min(U)<-.1 and max(U)>.1.
2. Construct observed band x conflict FIT groups and empirical masses pi. Fix each band's mass.
3. For each training step, compute per-example epsilon-insensitive absolute loss, then group means L_g.
4. Maximize sum q_g L_g subject to KL(q||pi)<=.04 and fixed per-band masses. Inside each band the solution has q_g proportional to pi_g exp(L_g/tau), with a common positive temperature chosen to meet the total KL budget when active. Ties/zero radius return the appropriate limiting solution. The current optimizer implementation is KLAdversary in the original stage41_cscwd.py; feasibility and independent constrained-solver comparisons are recorded in implementation_audit.json.
5. MLP update: hold q fixed and take an AdamW step on mean_i [(q_group/pi_group)*loss_i]. The group weights are recomputed each step. All arm parameters and optimizer states are independent, with paired initialization. This is a stochastic/nonconvex training procedure, not an exact minimax optimum certificate.
6. SVR uses a separate weighted kernel optimization and mass optimization loop. Its solver receipts and numerical gap checks must be reported separately; do not describe it as AdamW or claim a shared convergence result.
7. Select the ordinary MLP's TUNE-MAE-minimizing epoch and use the same epoch for every original audit arm. Original audit inference clips scalar predictions to [-1,1].
8. Repeat for true FIT flags, intensity-only grouping, and each independently shuffled FIT assignment. Exact shuffles act only within source/exact-y cells and preserve cell counts; pure cells are unchanged. Do not average their predictions before computing errors.
9. Evaluate paired absolute-error differences on untouched-by-this-run TEST predictions, both full population and mixed-cell population. Historical cohorts are not pristine holdouts. Report C/N/A/overall together and state the resampling units.

## Stage150: positive excess-error correction

Three inner source-disjoint ordinary reference heads supply out-of-source FIT predictions; the ensemble is used on TUNE/TEST. A bounded residual head corrects these predictions. The primary objective is task MAE + .5 matched-conflict MAE + 2 mean nonconflict positive excess MAE. Each positive error increase is penalized before averaging, so harms cannot cancel improvements inside this penalty. This does not prove no harm on TEST, and OOF-to-ensemble deployment is a distribution change. Stage150 failed the simpler-residual and cross-setting tests; do not promote it.

This belongs to the same broad family as the project's previous cross-fitted kernel source-regret work (Stage93). Its narrower differences are bounded neural corrections, individual positive excess error, and a matched training incentive. It is NOT the first use of cross-fitting or regret protection in this project or literature.

## Stage151: source-score-centered directional supervision

Three modality encoders feed the task head. Each has a unimodal auxiliary sentiment head. With E_im=Uhat_im-U_im and source/exact-y cell s(i), the conditional penalty is mean |E_im - mean_{j in s(i)} E_jm| over FIT cells of size >=2. Primary total loss: task MAE + .25 auxiliary MAE + .25 conditional penalty. All heads exist in all controls, and each arm selects its own ordinary TUNE MAE checkpoint within the same 100-step budget. This differs from the original audit's common-epoch rule; the paper must not conflate the two studies.

The penalty preserves continuous modality information, unlike a scalar conflict BCE. It is not all-pairs L1, contrastive InfoNCE, a causal effect, or an invariance guarantee on unseen sources. It removes cell-level additive mean errors. Exact target-vector shuffles test whether assignment within the cell adds information.

## Stage153: two-way relative-disagreement supervision

Let P_s subtract the per-cell mean across FIT examples, and P_m=I-11^T/3 subtract a sample's mean across its modalities. Define L_rel=mean |P_s (Uhat-U) P_m| over nonsingleton FIT cells. Total loss: task MAE + .25 auxiliary MAE + .25 L_rel. Inference uses only X; neither true y, U, source ID, nor conflict labels is an input.

For any sample offset a_i and cell/modality offset B_s,m, L_rel(E+a 1^T+B_cell)=L_rel(E). Both shift invariances are verified numerically before training. They isolate relative modality discrepancies from these additive common components in the auxiliary loss. They DO NOT make the full encoder's task gradient orthogonal to the auxiliary gradient, establish causal deconfounding, or ensure nonconflict risk cannot increase. Two-way centering is standard mathematics; novelty, if supported, would reside in its supervised use for conditional modality-disagreement learning and the demonstrated incremental evidence.

## Closest literature and what must be compared

- MISR (Findings ACL 2025): https://aclanthology.org/2025.findings-acl.761/ . Its method learns Gaussian latent joint representations and invariance across multimodal/unimodal environments. Our candidate preserves conditional relative disagreements instead of claiming all modality sentiment signals should be invariant. Its full implementation has NOT been reproduced in this round.
- MoSEAR (ACM MM 2025): https://arxiv.org/html/2508.01181v2 . Modality-specific LoRA experts and attention reallocation address audio/visual imbalance in multimodal emotion reasoning. It already aims to improve conflict and consistent cases; we cannot claim that goal is new. Task, backbone and labels differ, so paper numbers cannot be copied into our table as equal-protocol baselines.
- TAME (arXiv v2, 2026-10-21): https://arxiv.org/html/2610.27571v2 . A text-anchored shared expert, cross-modal auxiliary experts and uncertainty-guided routing improve MSA. Our candidate has no fixed text anchor or uncertainty router; its contribution would concern conditional supervision. It is a recent preprint, not verified here as an accepted conference paper.
- DiffEmo: https://arxiv.org/html/2406.03004v1 . Modality inconsistency and its adverse effect are already studied. Do not claim first discovery of multimodal sentiment conflict.
- Learning to Reweight Examples (ICML 2018): https://proceedings.mlr.press/v80/ren18a.html . Validation-gradient example weighting is established; no novelty claim for generic reweighting or merely adding a guard penalty.

Literature was checked against author/venue full text or official abstract/method sections. This is a bounded nearest-work comparison, not an exhaustive priority certificate. Next official baseline reproduction should be MISR under our source splits, preprocessing, fixed search budget and full/subgroup metrics. A parameter-matched ordinary MTL control is necessary but not a substitute for that reproduction.

# Stage154: four null assignments for the fixed Stage151 candidate

2026-11-11. Adaptive retrospective audit, not fresh confirmation. No method or coefficient selection is permitted from this stage.

Keep Stage151's source-score-centered directional supervision, .25 ordinary unimodal MAE and .25 conditional MAE, three modality encoders, fixed 100 AdamW steps, own ordinary TUNE checkpoint selection, five seeds, three source folds, and both ranks 8/32. Reuse every real-target and exact-cell-shuffled checkpoint from Stage151 without refitting.

Only add three assignment schemes, each with 19 independent joint permutations of the three-component unimodal target vector on FIT:

1. global: preserve the full FIT target multiset;
2. source_only: preserve each FIT source's target multiset;
3. score_only: preserve each exact overall score's target multiset.

The conditional loss still centers on the ORIGINAL FIT source/exact-score cells in every arm; the permutation does not change examples, overall y, heldout data, or grouping. The permutations deliberately remove different kinds of target correspondence. They do not simulate exchangeable treatment assignment or identify a causal effect.

New fits: 2 ranks x 3 cohorts x 3 folds x 5 seeds x 3 schemes x 19 repetitions = 5130 lightweight heads. All 90 new banks must be sealed before evaluation. Bank shape, labels and ordering must agree with parent predictions. Report real-minus-null improvements (implemented as null error minus real error) for full C/N/overall, matched C/N/A and outside-overlap N, using 3000 crossed source/seed/permutation draws. Preserve each scheme's changed-vector fractions. Exact null evaluation is reused from Stage151, not called a new independent replicate. Never pick the most favorable null scheme as the headline conclusion.

Before training: pass current power_guard checkpoint/exit handshake, check active PID plus creation time, and verify permutation multiset invariants. Every ten epochs save model and optimizer state; reuse complete compatible banks. Before scoring, replay selected checkpoints against saved predictions and check hashes. A failed baseline or control remains in the ledger.

# Method, data splits and numerical audits

## Design readouts

The data are the Product portion of QMSum, repository revision 83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e. Combine the original train, validation and test meetings for the five outer **team** rotations. query_index.json links every out-of-team record to a question ID, meeting, outer fold and official source split. calibration.json lists four disjoint team roles per fold. role_queries.json lists the corresponding query IDs. endpoint_fits.json records training IDs in fitting order and the fitted coefficients, intercepts, means and scales. Each eligible specific question has at least one overlapping human evidence span. A candidate is a contiguous, nonoverlapping eight-turn window; the final window can be shorter.

Both heads share the following features, in this order.

1. Word TF–IDF cosine between query and joined passage, peak query-to-turn cosine inside the passage, and passage rank by the first cosine. The rank is stable descending and zero-based divided by max(1, number of passages minus one). Use lowercase word unigrams and bigrams, English stop words, min_df=2, sublinear term frequency, smoothed IDF and L2 normalization.
2. The corresponding three scores for character word-boundary 3–4 grams with min_df=8 and at most 24,000 features, using sublinear term frequency. Both vocabularies fit the individual utterances in endpoint-fit meetings only. These lexical inputs omit speaker prefixes.
3. Normalized window position and log(1 + whitespace word count).
4. Query-to-whole-passage embedding cosine, then the fraction of turns whose lowercased speaker-role string occurs in the lowercased query.

For the contextual head, append three features. Within the same passage, resolve up to four nonoverlapping two-turn blocks. Compute the maximum query-to-block embedding cosine, the mean of the largest two available cosines (one when only one exists), and maximum minus whole-passage cosine. Both heads use the same eight-turn boundaries. An intermediate preprocessing array contains neighboring-window lexical features, but the final enrichment step overwrites that array before fitting. Those intermediate features do not enter either prediction.

The embedding model is sentence-transformers/all-MiniLM-L6-v2 at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Join speaker name, colon, space and content for each turn; join turns with spaces. Encode questions separately. Use the archived tokenizer, truncate to 256 tokens including special tokens, use attention-mask mean pooling and L2-normalize the last hidden states. The local and contextual heads each fit a separate unweighted StandardScaler and then an L2 logistic model with intercept, lbfgs, C=1 and max_iter=600. The binary target is any overlap with a labeled span. A passage receives weight 1/n for its query of n passages, rescaled so the mean weight across the fit set is one. The output sigmoid is the positive-class probability. The verifier is a separate model fitted on separate teams. Its model identity, commit and weight digest are in source_manifest.json.

heldout_features.npz contains the 13-feature matrix in index order. The local model takes columns 0–9, the contextual model all 13 columns. endpoints.npz stores r, b, verifier probability and observed label in that order, with query offsets and workload labels. For full reconstruction using the public source files, put the QMSum zip under qmsum.zip and the pinned encoder files under minilm/ (with revision.txt). The assets directory must be writable if a missing embedding cache has to be created. Run:

    python -m experiments.refit_endpoints PATH_TO_ASSETS
    python -m experiments.check_encoder PATH_TO_ASSETS

This optional path needs scikit-learn 1.3.2, threadpoolctl, Transformers and PyTorch. Precomputed embeddings may be reused; without them the encoder populates embeddings/. Refitting from the archived embeddings exactly reproduced both readouts on all 57,140 held-out question–passage pairs. In a separate cold encoder check we recomputed all six questions, 41 passage embeddings and 163 two-turn embeddings of ES2003c from its transcript. All three arrays matched the cache on the study machine. Re-encoding all meetings on other hardware can introduce floating-point differences. The saved endpoints define the exact replay target.

## Audit manifest

The joint audit compares the full feasible action family, minimum movement and winning score vectors under the archived probability and the label-visible objective. See source_manifest.json for cache paths and SHA-256 digests, and forum_manifest.json for corpus revisions. The 768 external queries here use title-generated pools. Reusing those questions for the full-text diagnosis does not double this count.

| Workload | Selection | Queries | Readout source |
|---|---|---:|---|
| Product design | 137 meetings, 35 teams, five outer folds | 847 | QMSum Product fold files, local and contextual heads above |
| Academic meetings | 17 remaining meetings from the official training partition, used as transfer evaluation; test/validation meeting IDs excluded, after the first 24 source-hash-ranked eligible training meetings | 93 | Frozen Product protocol, Academic_remaining |
| Committee meetings | 12 additional official training meetings, source-hash ranks 13–24 after excluding test/validation meeting IDs | 147 | Frozen Product protocol, Committee_more |
| City council | Eligible official MeetingBank test queries in 446 meetings | 742 | Independently fitted segment/window lexical heads and neural verifier |
| City council reserve | 76 additional eligible official validation meetings, outside the 120 used for policy calibration and disjoint from test | 122 | MeetingBank reserved readouts |
| MEED | Official test dialogues, one list of utterances per dialogue | 42 | Audio masking/reconstruction vs bypass; confusion or frustration label |
| Community duplicates | Twelve forums, first 64 official test questions per forum by SHA256 ordering, 24 candidates | 768 | Title cosine, mean of title and full-text cosines, sigmoid verifier |
| Total | Seven workloads | 2,761 | All included in joint enumeration |

MeetingBank revision 5b4fb6f67f490be93e249d70b732780932d19fe3. MEED revision ec13f7ad67c52f69b9dae23af40a79d0dc627c32. Full model and data fingerprints are in the manifest.

**Original LP sample.** Traverse the original non-forum records in stored order. If n≤10 and fewer than 900 actions have been checked on earlier queries, finish all candidate sets with one or two exchanges on the current query. The last complete query brings the count to 924 across 33 MeetingBank test queries. Compare affordable set membership for every test. Compare exact minimum costs for the ten affordable actions; the other 914 were infeasible or exceeded the budget. The independent LP uses score and absolute-movement variables, box constraints and pairwise separation of at least 1e-8. scipy.optimize.linprog uses HiGHS with primal and dual feasibility tolerances of 1e-10. Budget slack is 1e-12 and affordable-cost comparison tolerance is 1e-9. Individual sample indices are in replay_verification.json.

**Additional LP coverage.** For each workload order queries by (length, archive index) and take five evenly spaced integer ranks, including both ends. For one and two exchanges check the nearest boundary swap, the extreme swap and the cheapest affordable swap when available; deduplicate target sets within a query. No relevance label or verifier probability enters the selection. Across 35 lists this yields 130 tests, including 23 affordable ones. stratified_lp.json identifies each target set, budget and LP cost (null for infeasibility). This targeted sample and the original sample have different selection rules and are reported separately.

**Numeric conventions.** Use the closed score range when protection is off or rho=0. Otherwise require 0<rho≤1. Ties in the greedy center allocation use original input order. Fit the local budget factor only on positive local movement masses. All 125 development ratios were valid; their median was 5.402338282289333. An empty valid set is assigned multiplier zero, a conservative extension unused in the original data. development_masses.json preserves the input. Exact comparisons use array equality for sparse vs exhaustive outputs under the same floating-point convention. Constraint auditing uses the stated slack.

## Paired outcomes and reuse example

A full replay writes runs/paired_rows.json and reports the two annotated hit counts on each of 847 queries. Compute their paired difference divided by three, then leave out each team and recompute the query-weighted mean. The interval uses the jackknife standard error and a t critical value with 34 degrees of freedom. Meeting ES2003c_1 illustrates a proposed handoff that imports scores subject to the retention policy. The numerical case is in results/meeting_case.json; the public QMSum transcript provides the passage at the given zero-based indices. Case selection favors identical lists, a displayed candidate, and an observed-label improvement, then uses a deterministic identifier tie-break after lock status. It is a mechanism illustration. The import check is a proposed design condition; manual effort and user preference were not measured. A separate illustrative .20 fixed-cutoff check on upward-promoted locked entries returned no examples; it is not used to claim operational benefit.

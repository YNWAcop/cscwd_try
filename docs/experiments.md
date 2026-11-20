# Running the study

Run commands from the repository root. The baseline environment used Python 3.10, NumPy 1.24.4 and SciPy 1.11.4. No training seed is selected by the replay. Run Python with assertions enabled.

## CPU replay

    python -m pip install -r requirements.txt
    python -m experiments.reproduce

The runner regenerates the matched mechanisms, recovers the paired query outcomes, checks protected realizations of violating lists, compares indexed and exhaustive decisions, and recomputes Tables I–III. Output goes to the ignored runs/ directory. For Table II it aggregates recorded warm GPU measurements. A full batch has three measurements, so the published table gives its median only.

| Path | Inputs or output |
|---|---|
| data/audit/endpoints.npz | r (contextual score), b (local score), p (verifier estimate), y (observed label), with query offsets, domain and group arrays |
| data/audit/forum.npz | Frozen title-pool records in query order, using the same four columns |
| data/design/ | Held-out numerical features, endpoint coefficients, team roles, query IDs and budget calibration masses |
| data/forum/body_rerun_queries.json | Full-text candidate-pool diagnostic on the same 768 questions, including the candidate IDs, two readings, verifier output and qrels-derived hits |
| data/timing/ | Design batch timings, forum text-to-list timings, and official-split design readouts for optional hardware benchmarks |
| results/ | Reference aggregates and published tables |
| runs/ | New predictions, per-query comparisons, LP checks, tables and optional figures |

One probe is one query–passage verifier score. Hit counts refer to supplied labels. The diagnostic using body-generated pools was retrospective. Fixed endpoint replay can check exact decision equality on these inputs; it does not establish the same latency on a new GPU.

## Optional feature fitting, corpus retrieval and figures

Install the additional dependencies in requirements-optional.txt, choosing a CUDA-capable PyTorch build for GPU timing. Paths below are arguments supplied by the researcher. The raw corpora, raw meeting text and pretrained model weights are obtained separately from their publishers.

| Command | Task and required assets |
|---|---|
| `python -m experiments.refit_verifier` | Refit the five fold-specific verifier calibrations using the bundled numerical logits and check the held-out probabilities. |
| `python -m experiments.refit_endpoints ASSETS` | Refit five QMSum source-team readouts. ASSETS contains qmsum.zip and minilm/ with tokenizer, model and revision.txt. Embeddings can be reused in ASSETS/embeddings/ or generated there. Saves coefficients and held-out features under runs/. |
| `python -m experiments.check_encoder ASSETS` | Re-encode the manuscript example and compare it with cached embeddings. Run after populating the cache. |
| `python -m experiments.forum_retrieval FORUM_DIR --model-dir MODEL_DIR` | Recompute title and full-text lexical retrieval and verify the body-generated pools with both sparse and dense neural inference. Omit --model-dir for lexical results alone. |
| `python -m experiments.benchmark_design QMSUM_ZIP MODEL_DIR` | Remeasure design timings, selecting adaptive widths on the development workload and evaluating 129 official-split queries on CUDA. |
| `python -m experiments.benchmark_forum FORUM_DIR MODEL_DIR` | Remeasure query-text-to-list service time on the title-generated candidates using a local corpus index and CUDA. |
| `python -m experiments.figures` | Regenerate the two figures from the reference results under runs/figures/. Uses a portable font. |

FORUM_DIR must contain the twelve forum subdirectories. Each contains corpus/*.parquet, queries/*.parquet and test.tsv. MODEL_DIR is a local copy of the frozen MS MARCO cross-encoder. The fold-specific design calibration differs from the external raw-logit sigmoid mapping. Version and source-file fingerprints are in [the source manifest](../provenance/source_manifest.json) and [the forum manifest](../provenance/forum_manifest.json).

The original data projects are [QMSum](https://github.com/Yale-LILY/QMSum), [MeetingBank](https://huggingface.co/datasets/huuuyeah/meetingbank), [MEED](https://github.com/Nemo9303/MEED-dataset) and the CQADupStack forums used in [BEIR](https://github.com/beir-cellar/beir). The frozen models are [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). See [method and audits](method-and-audits.md) for revisions, feature definitions, and the exact selection of the LP checks. The product and forum paths above can be rerun from external assets; the auxiliary MeetingBank, MEED, Academic and Committee audits start from archived readouts.

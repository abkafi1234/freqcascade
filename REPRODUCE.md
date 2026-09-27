# Reproducing the paper

Every table and figure is produced by `scripts/`, from the loaders in
`freqcascade.datasets` and the committed fold indices in `data/folds/`.

## Setup

```bash
pip install -e ".[all]"        # torch, imbalanced-learn, statsmodels, datasets, nltk
python -c "import nltk; nltk.download('reuters')"
python scripts/verify_datasets.py           # fetch every corpus, print a summary
```

Downloaded corpora and cached embeddings live under `~/.cache/freqcascade`
(override with `FREQCASCADE_DATA`).

## Datasets

| `load(...)` | track | n | classes | IR | source |
|---|---|---|---|---|---|
| `clinc150` | single-label | 18,025 | 150 | 4.0 | oos-eval `data_imbalanced.json` (fixed split) |
| `20newsgroups` | single-label | 5,272 | 20 | 50 | sklearn + geometric imbalance injection |
| `wos46985` | single-label | 46,985 | 134 | 17 | HF `bakirgrbic/web-of-science` |
| `ohsumed_23` | single-label | 23,166 | 23 | 29 | Moschitti `ohsumed-first-20000-docs` (medical) |
| `drug_reviews` | single-label | 50,000 | ~356 | ~1800 | HF `lewtun/drug-reviews`, `condition` from text (medical) |
| `reuters21578` | multi-label | 10,788 | 90 | 1982 | NLTK `reuters` ModApte |
| `hoc` | multi-label | 1,580 | 10 | 4.4 | `sb895/Hallmarks-of-Cancer` (medical) |
| `litcovid` | multi-label | 33,699 | 7 | 17 | HF `KushT/LitCovid_BioCreative` (medical) |

`20newsgroups`: the original submission's per-class subsample counts were not
archived, so the loader reproduces the target IR (50.5) and the geometric
construction, not necessarily the first run's exact row count.

## Environment

Exact versions are pinned in the KBS bundle's `setup/requirements.txt` (PyTorch 2.11.0, CUDA 12.8
wheels; scikit-learn 1.9.0; imbalanced-learn 0.14.2). CPU-only results reproduce exactly across
machines. Neural results reproduce within GPU-kernel nondeterminism: on a second GPU a single-label
cell matched to 4e-5 macro-F1, a multi-label chain cell to 2e-3, and individual chain links can
differ by more, so no neural experiment should mix results from two GPUs.

Heavy scripts are resumable: each appends one row per finished cell to `results/*.jsonl` and skips
cells already present. Parallel Random Forest work uses `FREQCASCADE_RF_NJOBS` (default 12). It
changes cost, not results; on a 15 GB machine use 4.

## 1. Main campaign

```bash
python scripts/make_folds.py                      # data/folds/*.npz (committed; regenerates identically)
python scripts/precompute_embeddings.py           # data/cache/embeddings/*.npy
python scripts/run_single_label_benchmark.py      # results/<dataset>.jsonl
python scripts/run_drug_reviews_nn.py             # RFOED-NN on drug_reviews, capped cascade
python scripts/run_multilabel_benchmark.py        # results/{reuters21578,hoc,litcovid}.jsonl
python scripts/run_factorial_ablation.py clinc150 20newsgroups wos46985 drug_reviews ohsumed_23
python scripts/run_factorial_ablation.py --multilabel reuters21578 hoc litcovid   # results/ablation_*.jsonl
python scripts/run_representation_study.py        # results/rq5_*.jsonl (RQ5)
python scripts/scaling_curve.py                   # results/scaling_clinc150.jsonl (class-count sweep)
python scripts/dump_per_class_recall.py           # results/perclass_clinc150.jsonl
python scripts/run_unit_pilot.py                  # results/unit_pilot.jsonl
python scripts/diagnose_wos_sweep.py              # results/wos46985_sweep.jsonl
python scripts/run_wos_remediation.py             # results/wos46985_remediated.jsonl
python scripts/diagnose_wos.py --bases rf         # results/wos46985_diagnosis.jsonl (CPU)
python scripts/diagnose_wos.py --bases nn         #   neural cascade, capped and uncapped (GPU)
python scripts/run_phase2.py                      # optional: runs the benchmark stages above as one resumable job
```

## 2. Reviewer-response checks (earlier rounds)

```bash
python scripts/run_clinc150_cv.py                 # results/clinc150_cv.jsonl (protocol check)
python scripts/run_ncap_sweep.py                  # results/ncap_sweep.jsonl
python scripts/diagnose_ncap_wos.py               # results/wos46985_ncap.jsonl
python scripts/focc_exposure_bias.py              # results/focc_exposure.jsonl
python scripts/verify_cascade_bound.py            # results/cascade_bound.jsonl (Proposition 1)
python scripts/run_finetune_study.py              # results/finetune_study.jsonl (MiniLM, 3 corpora)
python scripts/spot_check_trees.py                # results/spot_trees.jsonl (100 vs 200 trees)
```

## 3. Round-3 experiments

```bash
python scripts/verify_cascade_dependence.py --base nn   # results/cascade_dependence.jsonl (R6)
python scripts/verify_cascade_dependence.py --base rf
python scripts/focc_perlink_exposure.py                 # results/focc_perlink.jsonl (R9)
python scripts/run_ncap_adaptive.py --folds 5           # results/ncap_adaptive.jsonl (R8)
python scripts/run_weaklearner_sweep.py --learners stump,tree3,tree10 --features tfidf,emb   # R5
python scripts/run_weaklearner_sweep.py --learners treefull --features tfidf,emb           # R5, memory-hungry
python scripts/run_weaklearner_sweep.py --learners mlp,xgb --features emb,tfidf            # R5, GPU
python scripts/run_leak_check.py 20newsgroups ohsumed_23 wos46985 drug_reviews             # results/leak_check.jsonl (B6)
python scripts/probe_finetune_throughput.py             # results/probe_finetune_throughput.json (planning only)
python scripts/run_finetune_grid.py --repro-check       # must pass before the grid
python scripts/run_finetune_grid.py --smoke             # HoC fold 0, one loss
python scripts/run_finetune_grid.py                     # results/finetune_grid.jsonl (R7)
```

## 4. Tables

Every results table in the paper is produced from `results/*.jsonl` by a script. The generators
print LaTeX table bodies; differences are computed from unrounded means.

| Paper label | Generator |
|---|---|
| `tab:mainresults`, `tab:mainresultsmed` | `python scripts/latex_tables.py` (single-label panels) |
| `tab:focc` | `python scripts/latex_tables.py` (multi-label panels) |
| `tab:anova` | `python scripts/latex_tables.py` (factorial ANOVA) |
| `tab:foccablation` | `python scripts/latex_tables.py` (FOCC 2x2 ablation) |
| `tab:rq5` | `python scripts/latex_tables.py` (RQ5) |
| `tab:remediation` | `python scripts/latex_tables.py` (WOS46985 remediation sweep) |
| `tab:pilot` | `python scripts/latex_pilot_table.py` |
| `tab:ncap` | `python scripts/latex_round3_tables.py r8` (adaptive cap sweep, 8 corpora; gamma selected by `scripts/select_ncap_gamma.py --report`) |
| `tab:finetune` | `python scripts/latex_reviewer_tables.py finetune` |
| `tab:cascadecheck` | `python scripts/latex_reviewer_tables.py cascade` |
| `tab:perlinkorder` | `python scripts/latex_round3_tables.py r9order` |
| `tab:dependence` | `python scripts/latex_round3_tables.py r6` |
| `tab:capacity` | `python scripts/latex_round3_tables.py r5compact` (full ladder: `r5 --features tfidf\|emb`) |
| `tab:diagnosis` | `python scripts/latex_round3_tables.py diagnosis` (from `results/wos46985_diagnosis.jsonl`, written by `diagnose_wos.py`) |
| `tab:threshold` | `python scripts/threshold_flat_mlp.py` then `python scripts/analyze_threshold.py --latex` (plain run applies the pre-registered criterion) |

Tables with no generator:

- `tab:protocolcheck` was assembled from `clinc150.jsonl` and `clinc150_cv.jsonl` without a script.
  Its Random Forest rows match those files. Its RFOED-NN row reports the earlier fixed node cap in
  both columns, because the protocol re-run (2026-09-16) predates the adoption of the adaptive cap
  and the re-run of the main results under it (2026-09-18); the paper's caption states this. The
  two columns are therefore internally consistent with each other, not with Table S1a.
- `tab:datasets`, `tab:gapmatrix` and `tab:notation` are descriptive. `verify_datasets.py` prints
  the dataset statistics.

`python scripts/analyze.py --metric macro_f1 --ref RFOED-NN results/<dataset>.jsonl` prints the paired
significance tests; `python scripts/summarize.py` prints a Markdown summary of every results file.

## 5. Figures

```bash
python scripts/kbs_figures.py            # all of the below, or name keys to render a subset
python scripts/make_figures.py           # the numbered result figures
```

| Figure file | Generator key |
|---|---|
| `figure_landscape`, `figure_collapse_bars`, `figure_mainresults_sl`, `figure_mainresults_ml` | `kbs_figures.py landscape collapse mainresults_sl mainresults_ml` |
| `figure_factorial_heatmap`, `figure_remediation_sweep`, `figure_separability`, `figure_calibration` | `kbs_figures.py factorial remediation separability calibration` |
| `figure_exposure_bias`, `figure_perlink_exposure_<dataset>` | `kbs_figures.py exposure perlink` |
| `figure_neighbourhood`, `figure_failure_mechanisms`, `figure_decision_tree` | `kbs_figures.py neighbourhood failure_mechanisms decision_tree` |
| `figure_rfoed_flow`, `figure_focc_chain`, `figure_local_vs_global`, `figure_batched_nn` | `kbs_figures.py rfoed_flow focc_chain local_vs_global batched_nn` |
| `graphical_abstract` | `kbs_figures.py graphical_abstract` |
| `figure2_rank_frequency`, `figure3_cd_diagram`, `figure5_focc_ablation` | `make_figures.py rank_frequency cd_diagram focc_ablation` |
| `figure7_representation`, `figure8_scaling_curve` | `make_figures.py representation scaling_curve` |

`figure1_schematic.pdf` has no generator in this repository.

Notes. `threshold_flat_mlp.py` must run with the default BLAS thread count: with 1 or 4 threads the
refitted flat MLP differs from the stored rows by up to 6e-4 and the script's harness check aborts.
The TF-IDF leak check (`run_leak_check.py`) has been applied: `results/*.jsonl` hold the per-fold rows
and the pooled-vectoriser originals are in `results/_archive_global_tfidf/`, so `--compare` now reports
zero differences; compare against the archive instead.

## 6. Utilities

- `scripts/_shared.py`: all configuration (CV seed, ensemble sizes, encoders, the method registry).
- `scripts/export_results.py`: exports every result behind the paper as a self-describing bundle.
- `scripts/check_reproduction.py`: recomputes a stored CPU cell and a stored GPU cell before resuming
  on a new machine.
- `scripts/resume_status.py`: reports what is done and what is left in the round-3 campaign.
- `scripts/verify_datasets.py`: fetches every corpus and prints a summary.

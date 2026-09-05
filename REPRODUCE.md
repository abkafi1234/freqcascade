# Reproducing the paper

Every table and figure in *"Frequency-Ordered Local Decomposition for Imbalanced
Text Classification"* is produced by the scripts in `scripts/`, from the loaders
in `freqcascade.datasets` and the committed fold indices in `data/folds/`.

## Setup

```bash
pip install -e ".[all]"        # torch, imbalanced-learn, statsmodels, datasets, nltk
python -c "import nltk; nltk.download('reuters')"
```

Downloaded corpora and cached embeddings go under `~/.cache/freqcascade`
(override with `FREQCASCADE_DATA`).

## Datasets

| name (`freqcascade.datasets.load`) | track | source | notes |
|---|---|---|---|
| `clinc150` | single-label | HF `clinc_oos/imbalanced` | 150 intents, out-of-scope dropped, native fixed split |
| `20newsgroups` | single-label | sklearn `fetch_20newsgroups` | geometric imbalance injected to IR 50 |
| `wos46985` | single-label | HF `web_of_science/WOS46985` | 134 sub-field classes |
| `drug_reviews` | single-label | HF `lewtun/drug-reviews` | classify `condition`; `min_count` filters the tail |
| `ohsumed_23` | single-label | Moschitti `ohsumed-first-20000-docs` | 23 MeSH C14 categories (medical) |
| `reuters21578` | multi-label | NLTK `reuters` (ModApte) | R(90), train+test pooled |
| `hoc` | multi-label | HF Hallmarks of Cancer | 10 cancer-hallmark labels (medical) |
| `litcovid` | multi-label | HF LitCovid (BioCreative VII) | 7 COVID-19 topic labels (medical) |

## Run order

```bash
# 1. Fold indices (already committed; regenerates identically from CV_SEED)
python scripts/make_folds.py

# 2. Main benchmarks -> results/<dataset>.jsonl
python scripts/run_single_label_benchmark.py
python scripts/run_multilabel_benchmark.py

# 3. RQ5 -- swap the frozen encoder (see scripts/_shared.py: ENCODERS)
python scripts/run_single_label_benchmark.py --encoder pubmedbert drug_reviews ohsumed_23
python scripts/run_multilabel_benchmark.py  --encoder pubmedbert hoc litcovid

# 4. Significance tables
python scripts/analyze.py --metric macro_f1       --ref RFOED-NN results/clinc150.jsonl results/wos46985.jsonl ...
python scripts/analyze.py --metric label_macro_f1 --ref FOCC-NN  results/reuters21578.jsonl results/hoc.jsonl results/litcovid.jsonl
```

Config (CV seed, ensemble sizes, encoder list, the full method registry) lives
in `scripts/_shared.py`.

## Still to add (revision, in progress)

- `scripts/run_factorial_ablation.py` -- the 2x2x2 (ordering x base learner x
  rebalance) and 2x2 FOCC ablations.
- `scripts/diagnose_wos.py` -- the node-level own-node-miss / early-capture
  error decomposition.
- `scripts/make_figures.py` -- regenerate every figure.
- Committed `data/folds/*.npz` (generated in the data-ingest phase once every
  loader is verified end to end).

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

## Run order

```bash
python scripts/make_folds.py                     # regenerates data/folds/ identically

python scripts/run_single_label_benchmark.py     # -> results/<dataset>.jsonl
python scripts/run_multilabel_benchmark.py
python scripts/run_factorial_ablation.py clinc150 20newsgroups wos46985 drug_reviews ohsumed_23
python scripts/run_factorial_ablation.py --multilabel reuters21578 hoc litcovid
python scripts/run_representation_study.py        # RQ5, sweeps _shared.ENCODERS
python scripts/diagnose_wos.py

python scripts/analyze.py --metric macro_f1 --ref RFOED-NN results/clinc150.jsonl ...
python scripts/make_figures.py
```

All config -- CV seed, ensemble sizes, encoder list, the full method registry --
is in `scripts/_shared.py`.

## Not yet wired (revision, in progress)

- `make_figures.py`: `margin_vs_K`, `focc_ablation`, `per_class_recall`,
  `representation` (results-driven -- added after the first full run).
- The WOS46985 remediation sweep (cap / threshold / prior-correct) is driven
  ad hoc through `RFOEDClassifier`'s arguments; a dedicated sweep script is TODO.

# freqcascade

**Recursive frequency-ordered ensemble decomposition for imbalanced multiclass and multi-label classification.**

Most imbalance-handling techniques (SMOTE, ADASYN, EasyEnsemble, RUSBoost, ...) resample or resample-and-ensemble the *whole* K-class problem at once. `freqcascade` instead decomposes a K-class problem into a sequence of **binary** sub-problems, ordered by class/label frequency, and rebalances each sub-problem **locally**. No node ever poses a resampling problem larger than binary, regardless of K — which is exactly where global resampling degrades or collapses outright at high class count (see [Benchmark results](#benchmark-results) below).

Two structural variants, both built on the same mechanism (an ensemble base learner at every node/link, each member fit on its own class-balanced bootstrap draw):

- **`RFOEDClassifier`** — Recursive Frequency-Ordered Ensemble Decomposition, for single-label multiclass problems. Peels the current majority class off vs. "the rest," most-frequent class first, recursing on the remainder. A resolved example is removed from contention for every later node.
- **`FOCCClassifier`** — Frequency-Ordered Classifier Chains, for genuinely multi-label problems (one example, several labels). A *chain*, not a *peel*: no example is ever removed, and each link's classifier conditions on the labels resolved by earlier links.

Both accept a pluggable `base_learner_factory` — ship with a bagged-Random-Forest base learner and a bagged neural-network-ensemble base learner (batched on GPU via PyTorch), and a small TextCNN unit for raw-text input.

## Install

```bash
pip install freqcascade                 # core only (numpy/scipy/scikit-learn/pandas)
pip install freqcascade[torch]          # + neural-ensemble base learner, sentence embeddings
pip install freqcascade[imbalance]      # + SMOTE/ADASYN/EasyEnsemble/RUSBoost baseline wrappers
pip install freqcascade[stats]          # + paired significance testing, CD diagrams, factorial ANOVA
pip install freqcascade[all]            # everything
```

Heavy/optional dependencies are only imported inside the functions that need them, not at package import time — `import freqcascade` itself only needs the core four.

## Quickstart — single-label (`RFOEDClassifier`)

```python
from freqcascade import RFOEDClassifier, RFBaseLearner

clf = RFOEDClassifier(
    base_learner_factory=lambda node_idx: RFBaseLearner(
        n_estimators=200,
        rebalance=True,       # each tree's bootstrap is class-balanced at every node
        random_state=node_idx,
    ),
    order="frequency",        # peel the most frequent remaining class first
    decision="cascade",       # or "argmax_proba" for path-probability decoding
)
clf.fit(X_train, y_train)     # y_train: array of class labels, any hashable type
preds = clf.predict(X_test)
proba = clf.predict_proba(X_test)   # per-class path probabilities
```

Swap in the GPU-batched neural-ensemble base learner (`pip install freqcascade[torch]`):

```python
from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

clf = RFOEDClassifier(
    base_learner_factory=lambda node_idx: TorchNNEnsembleBaseLearner(
        n_members=50, hidden_size=128, max_epochs=250, rebalance=True, random_state=node_idx,
    ),
    order="frequency",
)
clf.fit(X_train_embeddings, y_train)  # dense features, e.g. sentence embeddings
```

If a dataset has many fine-grained classes that break naturally into coarse groups (e.g. a taxonomy), `HierarchicalRFOEDClassifier` peels by group first, then runs a short per-group cascade — bounding the worst-case chain length any example has to survive:

```python
from freqcascade import HierarchicalRFOEDClassifier

clf = HierarchicalRFOEDClassifier(base_learner_factory=..., order="frequency")
clf.fit(X_train, y_train, groups=coarse_group_labels)
```

## Quickstart — multi-label (`FOCCClassifier`)

```python
from freqcascade import FOCCClassifier
from freqcascade.base_learners import RFBaseLearner

def link_factory(rank, rebalance):
    return RFBaseLearner(n_estimators=200, rebalance=rebalance, random_state=rank)

clf = FOCCClassifier(base_learner_factory=link_factory, order="frequency", rebalance=True)
clf.fit(X_train, Y_train)      # Y_train: (n_samples, n_labels) binary indicator matrix
Y_pred = clf.predict(X_test)
```

Standard multi-label baselines (Binary Relevance, Classifier Chains, Ensemble of Classifier Chains) are provided as ready-made factories in `freqcascade.multilabel_baselines` for comparison — `make_br_rf`, `make_cc_rf`, `make_ecc_rf`, `make_balanced_br_rf`, and NN-based counterparts.

## Metrics

Raw accuracy is a poor headline metric under imbalance. `freqcascade.metrics.evaluate` / `freqcascade.multilabel_metrics.evaluate_multilabel` report macro-F1, macro-recall/G-mean, MCC, and **bottom-quartile-class recall** (mean recall over the rarest 25% of classes — directly measures what imbalance-handling is supposed to fix) instead:

```python
from freqcascade.metrics import evaluate
evaluate(y_test, preds, y_train)
# {'macro_f1': ..., 'macro_recall': ..., 'gmean': ..., 'mcc': ..., 'bottom_quartile_recall': ...}
```

## Standard-baseline wrappers (`pip install freqcascade[imbalance]`)

For comparing against established imbalance-correction techniques on the same features:

```python
from freqcascade.resampling_baselines import ResamplingBaseline, easy_ensemble_classifier

smote_rf = ResamplingBaseline(resampler_name="smote", n_estimators=200)
smote_rf.fit(X_train, y_train)
```

## Statistical testing (`pip install freqcascade[stats]`)

Paired Wilcoxon signed-rank + Holm-Bonferroni, Friedman + Nemenyi with critical-difference diagrams, and a three-way ANOVA / Scheirer-Ray-Hare fallback for factorial ablations — the machinery behind a properly-reported comparison, not just point estimates:

```python
from freqcascade import stats

result = stats.paired_comparison(scores_a, scores_b)   # matched-seed/fold Wilcoxon + t-test + effect sizes
friedman = stats.friedman_test(score_matrix)             # rows=datasets, cols=methods
cd = stats.nemenyi_critical_difference(k=len(methods), n=friedman.n_blocks)
```

## Cross-validation utilities

`freqcascade.cv` provides `repeated_stratified_kfold` (single-label) and `repeated_iterative_stratified_kfold` (multi-label, ported from Sechidis et al. 2011 — `scikit-multilearn`'s implementation has a broken `random_state` and hidden global-RNG coupling, so this is a from-scratch, fully-seeded port). Both return the same `FoldSplit` type and support `save_folds`/`load_folds` so every method under comparison sees identical folds — required for the paired significance tests above to be valid.

## Why this exists — benchmark results

This package grew out of a research project comparing recursive frequency-ordered decomposition against standard resampling techniques on real imbalanced text-classification benchmarks (CLINC150, 20 Newsgroups, WOS46985, Reuters-21578). Headline findings:

- On CLINC150 (150 intent classes, real imbalance), the neural-ensemble variant beat every standard baseline including SMOTE, with statistical significance (paired Wilcoxon, Holm-corrected, p < 0.05).
- **EasyEnsemble and RUSBoost — global instance-resampling ensembles — collapsed to near-zero macro-F1 at 134–150 classes**, while never breaking down at 20 classes. This is the clearest, most consistent finding across the whole study: binary-imbalance-designed resampling-ensemble methods don't scale to many-class problems, which is precisely the failure mode local, per-node rebalancing sidesteps by construction.
- On a dataset of fine-grained, semantically-overlapping classes (scientific sub-fields within a handful of broad domains), decomposition alone was not sufficient — the underlying classes were hard to separate regardless of how the imbalance was handled. Decomposition fixes an *imbalance-at-scale* problem; it does not fix a *class-separability* problem. Diagnosing this honestly (rather than only reporting favorable results) is part of what this package's test/experiment tooling is designed to support.

Frequency ordering itself (peel/chain the most frequent class or label first) was validated as a statistically significant, robust effect for the single-label peel structure across every dataset tested — but was found to be *neutral-to-harmful* for the multi-label chain structure, where rebalancing (not ordering) drove essentially all of the benefit. Both directions are reported because both are real.

## Development / CI

- `.github/workflows/test.yml` runs the full test suite (`pytest tests/`) on every push/PR across Python 3.10–3.12, plus a separate job that installs with **no** optional extras to verify the core-only import promise above actually holds.
- `.github/workflows/publish.yml` builds the sdist/wheel and publishes to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no long-lived API token stored in the repo) whenever a GitHub Release is published. It can also be run manually (`workflow_dispatch`) as a build-only dry run.

**One-time setup before the first release** (do this on pypi.org, not in this repo):
1. Create the `freqcascade` project on PyPI (or reserve the name).
2. On the project's PyPI page, go to *Publishing* → *Add a new publisher* → GitHub, and fill in this repo's owner/name, workflow filename `publish.yml`, and environment name `pypi` (matches the `environment:` block in the workflow).
3. In this GitHub repo's Settings → Environments, create an environment named `pypi` (optionally with required reviewers, for a manual approval gate before every publish).
4. Bump `version` in `pyproject.toml`, then cut a GitHub Release (tag `vX.Y.Z`) — the publish workflow takes it from there.

## License

MIT

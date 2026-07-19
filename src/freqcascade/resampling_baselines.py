"""Standard, established imbalance-correction baselines (RQ7/H5 in
EXPERIMENTAL_DESIGN.md §3): resample the *whole* K-class problem once,
then fit one flat classifier — the direct alternative to RFOED's
"decompose into binary sub-problems, rebalance each one locally"
approach. These exist so the paper can answer "why not just use SMOTE?"
with numbers instead of assertion.

All resamplers only ever see the training data (never val/test), and
all baselines here share whatever featurizer the rest of the comparison
uses, so any gap traces to the imbalance-handling strategy alone.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier


def _safe_k_neighbors(y: np.ndarray, requested: int = 5) -> int:
    """SMOTE/ADASYN require k_neighbors <= (smallest class count - 1).
    Auto-cap rather than crash, and this is exactly the kind of
    degradation-at-high-class-count H5 predicts — worth surfacing, not
    hiding, so the caller should log when capping actually occurs.

    A class with only 1 example has zero valid neighbors (SMOTE needs
    another same-class point to interpolate towards) — that case can't
    be capped to anything meaningful, so it's raised here with a clear,
    actionable message rather than left to surface ~3 layers down as
    imblearn/sklearn's much more opaque "Expected n_neighbors <=
    n_samples_fit" NearestNeighbors error."""
    _, counts = np.unique(y, return_counts=True)
    smallest = int(counts.min())
    if smallest < 2:
        raise ValueError(
            f"SMOTE-family resamplers need at least 2 examples in every class "
            f"to compute even a single neighbor, but the smallest class here "
            f"has only {smallest}. Use resampler_name='random_oversample' or "
            f"'random_undersample' instead, which have no such requirement."
        )
    return max(1, min(requested, smallest - 1))


def adasyn_floored_minority_strategy(y: np.ndarray) -> dict:
    """Alternate ADASYN `sampling_strategy` tried as part of the T2.4
    ADASYN retry (EXPERIMENTAL_DESIGN.md Table 2; TIER1_RESULTS.md ADASYN
    note). imblearn's default `sampling_strategy="auto"` for over-samplers
    is defined as, and verified empirically to behave identically to,
    `"not majority"` (every non-majority class raised to the majority
    count) — so retrying with the literal string `"not majority"` is a
    no-op and was not worth a separate code path.

    This instead is a genuinely different, gentler target: only classes
    *below the per-class-count median* are oversampled, and only up to
    the median (not the majority count). That shrinks both how many
    classes ADASYN must synthesize for and the per-class `n_samples`
    delta, which changes whether ADASYN's per-class rounding step
    (`n_samples_generate = round(ratio_nn * n_samples)`, in
    `imblearn.over_sampling._adasyn.ADASYN._fit_resample`) ever zeroes
    out entirely for every point in a class -- the proximate cause of the
    "No samples will be generated with the provided ratio settings."
    ValueError seen at the default `auto` target on CLINC150/WOS46985.

    Empirically (see TIER1_RESULTS.md): this recovers CLINC150 (succeeds)
    but not WOS46985 (still zeroes out on at least one class) -- i.e. a
    better-chosen sampling_strategy is *not* a full fix at the highest
    class count, only at the more moderate one."""
    counts = Counter(y)
    median = int(np.median(list(counts.values())))
    return {cls: median for cls, cnt in counts.items() if cnt < median}


class ResamplingBaseline:
    """Uniform fit/predict wrapper: `resampler_name` selects one of the
    standard techniques, applied once to the full training set, feeding
    a flat RandomForestClassifier (or a supplied classifier factory)."""

    RESAMPLER_NAMES = (
        "none",
        "random_undersample",
        "random_oversample",
        "smote",
        "adasyn",
        "adasyn_floored_minority",  # T2.4 alt-sampling_strategy ADASYN retry
        "smoteenn",
    )

    def __init__(
        self,
        resampler_name: str,
        classifier_factory=None,
        n_estimators: int = 200,
        n_jobs: int = 4,
        random_state: int | None = None,
    ):
        if resampler_name not in self.RESAMPLER_NAMES:
            raise ValueError(f"unknown resampler_name: {resampler_name}")
        self.resampler_name = resampler_name
        self.classifier_factory = classifier_factory or (
            lambda: RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state
            )
        )
        self.random_state = random_state
        self._model = None
        self.k_neighbors_used_: int | None = None
        self.capped_: bool = False

    def _build_resampler(self, y):
        if self.resampler_name == "none":
            return None
        if self.resampler_name == "random_undersample":
            return RandomUnderSampler(random_state=self.random_state)
        if self.resampler_name == "random_oversample":
            return RandomOverSampler(random_state=self.random_state)
        if self.resampler_name in ("smote", "adasyn", "adasyn_floored_minority", "smoteenn"):
            k = _safe_k_neighbors(y, requested=5)
            self.k_neighbors_used_ = k
            self.capped_ = k < 5
            if self.resampler_name == "smote":
                return SMOTE(k_neighbors=k, random_state=self.random_state)
            if self.resampler_name == "adasyn":
                return ADASYN(n_neighbors=k, random_state=self.random_state)
            if self.resampler_name == "adasyn_floored_minority":
                strategy = adasyn_floored_minority_strategy(y)
                return ADASYN(n_neighbors=k, sampling_strategy=strategy, random_state=self.random_state)
            return SMOTEENN(
                smote=SMOTE(k_neighbors=k, random_state=self.random_state),
                random_state=self.random_state,
            )
        raise AssertionError("unreachable")

    def fit(self, X, y) -> "ResamplingBaseline":
        y = np.asarray(y)
        resampler = self._build_resampler(y)
        if resampler is not None:
            X, y = resampler.fit_resample(X, y)
        self._model = self.classifier_factory()
        self._model.fit(X, y)
        return self

    def predict(self, X):
        assert self._model is not None, "call fit() first"
        return self._model.predict(X)

    def predict_proba(self, X):
        assert self._model is not None, "call fit() first"
        return self._model.predict_proba(X)


def easy_ensemble_classifier(n_estimators: int = 50, random_state: int | None = None):
    """Resampling-*ensemble* baseline (Liu, Wu & Zhou 2009): many bagged
    balanced subsets of the flat K-class problem — the closest existing
    analog to RFOED-RF's "many balanced views" idea, but balancing on the
    instance axis of a flat problem rather than RFOED's class-decomposition
    axis. Important comparator for isolating what RFOED's structure adds
    beyond "more balanced bagging.\""""
    return EasyEnsembleClassifier(n_estimators=n_estimators, random_state=random_state)


def rusboost_classifier(n_estimators: int = 50, random_state: int | None = None):
    """Boosting + random undersampling (Seiffert et al. 2010)."""
    return RUSBoostClassifier(n_estimators=n_estimators, random_state=random_state)

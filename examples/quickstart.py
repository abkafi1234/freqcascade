"""End-to-end demo: RFOED vs a SMOTE baseline on CLINC150, the exact API
from the paper. Replaces the old test.ipynb (which read a data file that
wasn't in the repo).

    pip install "freqcascade[datasets,imbalance]"
    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from freqcascade import RFOEDClassifier, RFBaseLearner
from freqcascade.datasets import load
from freqcascade.features import TfidfFeaturizer
from freqcascade.metrics import evaluate
from freqcascade.resampling_baselines import ResamplingBaseline


def main() -> None:
    ds = load("clinc150")
    print("dataset:", ds.summary())

    is_train = ds.split != "test"
    feat = TfidfFeaturizer(max_features=20000, ngram_range=(1, 2))
    feat.fit(list(ds.texts[is_train]))
    X = feat.transform(list(ds.texts))
    y = ds.target
    Xtr, ytr = X[is_train], y[is_train]
    Xte, yte = X[~is_train], y[~is_train]

    rfoed = RFOEDClassifier(
        base_learner_factory=lambda node: RFBaseLearner(
            n_estimators=200, rebalance=True, random_state=node
        ),
        order="frequency",
        decision="cascade",
    )
    rfoed.fit(Xtr, ytr)

    smote = ResamplingBaseline(resampler_name="smote", n_estimators=200, random_state=0)
    smote.fit(Xtr, ytr)

    print("\nmacro-F1 / bottom-quartile recall")
    for name, est in (("RFOED-RF", rfoed), ("Flat-RF+SMOTE", smote)):
        m = evaluate(yte, est.predict(Xte), ytr)
        print(f"  {name:<16} {m['macro_f1']:.3f}   {m['bottom_quartile_recall']:.3f}")


if __name__ == "__main__":
    main()

"""freqcascade: recursive frequency-ordered ensemble decomposition for
imbalanced multiclass and multi-label classification.

Core idea: instead of handling class imbalance globally (resampling the
whole K-class problem at once, the way SMOTE/ADASYN/EasyEnsemble do),
decompose the problem into a sequence of binary sub-problems ordered by
class/label frequency, and rebalance each sub-problem locally. Every
node/link never poses a resampling problem larger than binary,
regardless of how many classes the original problem has -- which is
where global resampling methods degrade or collapse outright at high
class count.

Two structural variants:

- :class:`~freqcascade.decomposition.RFOEDClassifier` (Recursive
  Frequency-Ordered Ensemble Decomposition) -- for single-label
  multiclass problems. Peels the current majority class off vs. "the
  rest," most-frequent first, recursing on the remainder. A resolved
  sample is removed from contention for every later node.
- :class:`~freqcascade.focc.FOCCClassifier` (Frequency-Ordered
  Classifier Chains) -- for multi-label problems, where the peel's
  removed-from-contention assumption doesn't hold (one example can carry
  several labels at once). Structured as a chain instead: no example is
  ever removed, and each link conditions on the labels resolved so far.

Both accept a pluggable `base_learner_factory` -- an ensemble base
learner (bagged Random Forest, or a bagged neural-network ensemble via
:class:`~freqcascade.torch_ensemble.TorchNNEnsembleBaseLearner`) is fit
independently at every node/link, each member on its own
(optionally class-balanced) bootstrap draw.

Heavy/optional dependencies (torch, sentence-transformers,
imbalanced-learn, statsmodels, matplotlib) are only imported inside the
functions that need them, not at package import time -- `import
freqcascade` itself only requires numpy/scipy/scikit-learn/pandas.
Install the relevant extra (`pip install freqcascade[torch]`, `[imbalance]`,
`[stats]`, or `[all]`) to use those specific features.
"""

from . import base_learners, decomposition, features, focc, metrics, multilabel_baselines, multilabel_metrics, rebalance
from .base_learners import NNEnsembleBaseLearner, RFBaseLearner
from .decomposition import HierarchicalRFOEDClassifier, RFOEDClassifier
from .focc import FOCCClassifier
from .metrics import evaluate
from .multilabel_metrics import evaluate_multilabel

__version__ = "0.1.0"

__all__ = [
    "RFOEDClassifier",
    "HierarchicalRFOEDClassifier",
    "FOCCClassifier",
    "RFBaseLearner",
    "NNEnsembleBaseLearner",
    "evaluate",
    "evaluate_multilabel",
    "base_learners",
    "decomposition",
    "features",
    "focc",
    "metrics",
    "multilabel_baselines",
    "multilabel_metrics",
    "rebalance",
]

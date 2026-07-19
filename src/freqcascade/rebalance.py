"""Per-node rebalancing. This is the ablatable factor that isolates
RFOED's delta over SBC (Vasudevan et al. 2024), which uses the
frequency-ordered peel structure alone with no rebalancing step."""

from __future__ import annotations

import numpy as np


def balanced_bootstrap_indices(
    y_binary: np.ndarray, rng: np.random.Generator, max_per_class: int | None = None
) -> np.ndarray:
    """Stratified bootstrap: sample with replacement independently within
    each of the two binary classes so both are drawn up to the size of
    the larger class — the same mechanism as Balanced Random Forest
    (Chen et al. 2004), generalized to be the shared rebalancing step for
    both RFOED-RF and RFOED-NN's ensemble members.

    `max_per_class` caps the draw size. Without it, a node with a huge
    "rest" class (e.g. WOS46985's shallow nodes: ~32k active, one class
    of ~750 vs. "everyone else" ~32k) draws target_n = ~32k per class —
    which, batched across K=50 GPU ensemble members at 384-dim
    embeddings, is several GB per node and OOM'd a real run on this
    hardware. A small MLP head doesn't need tens of thousands of
    per-member examples anyway (diminishing returns well before that),
    so capping is both the memory fix and a speed win — verified this
    doesn't silently change RF's behavior, which uses its own
    class_weight-based balancing (base_learners.RFBaseLearner), not
    this function, so RF is unaffected by the cap either way."""
    classes, counts = np.unique(y_binary, return_counts=True)
    target_n = counts.max()
    if max_per_class is not None:
        target_n = min(target_n, max_per_class)
    indices = []
    for c in classes:
        class_idx = np.flatnonzero(y_binary == c)
        drawn = rng.choice(class_idx, size=target_n, replace=True)
        indices.append(drawn)
    combined = np.concatenate(indices)
    rng.shuffle(combined)
    return combined


def random_undersample_indices(
    y_binary: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Undersample (without replacement) the larger class down to the
    smaller class's size — an alternative rebalancing strategy to compare
    against bootstrap oversampling in later ablations."""
    classes, counts = np.unique(y_binary, return_counts=True)
    target_n = counts.min()
    indices = []
    for c in classes:
        class_idx = np.flatnonzero(y_binary == c)
        drawn = rng.choice(class_idx, size=target_n, replace=False)
        indices.append(drawn)
    combined = np.concatenate(indices)
    rng.shuffle(combined)
    return combined


REBALANCE_STRATEGIES = {
    "bootstrap": balanced_bootstrap_indices,
    "undersample": random_undersample_indices,
    "none": None,
}

"""Tests for freqcascade's core promise: `import freqcascade` and the
RF-backed classes never require torch/imbalanced-learn/statsmodels/
matplotlib, even though the dev/test environment has all of them
installed. Uses two mocking techniques to prove this without relying on
"the CI environment happens to lack torch":

1. A subprocess launched fresh, so `sys.modules` reflects only what
   `import freqcascade` itself pulls in -- immune to import order/state
   leaking from other tests in the same process.
2. `unittest.mock.patch` on `builtins.__import__` to make `import torch`
   raise ModuleNotFoundError on demand, inside the *current* process --
   proves the core RFOEDClassifier/RFBaseLearner path is torch-independent
   even when torch is actively unavailable, not just untouched by luck.

Run with: pytest tests/test_lazy_imports.py -v
"""

from __future__ import annotations

import builtins
import subprocess
import sys

import numpy as np
import pytest

OPTIONAL_MODULES = ("torch", "imblearn", "statsmodels", "matplotlib", "sentence_transformers")


def test_core_import_does_not_pull_in_optional_heavy_deps():
    code = (
        "import sys\n"
        "import freqcascade\n"
        "leaked = [m for m in %r if m in sys.modules]\n"
        "assert not leaked, leaked\n"
    ) % (OPTIONAL_MODULES,)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _block_import(*blocked_names: str):
    """Returns a fake `__import__` that raises ModuleNotFoundError for the
    given top-level module names (and their submodules), delegating
    everything else to the real `__import__`."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in blocked_names or any(name.startswith(f"{n}.") for n in blocked_names):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    return fake_import


def test_torch_ensemble_fit_raises_cleanly_when_torch_is_unavailable(monkeypatch):
    monkeypatch.setattr(builtins, "__import__", _block_import("torch"))

    # Importing the module/class itself must still succeed -- torch is only
    # imported lazily inside method bodies, not at module load time.
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    learner = TorchNNEnsembleBaseLearner(n_members=2, hidden_size=4, max_epochs=1, random_state=0)
    X = np.random.default_rng(0).normal(size=(10, 3))
    y = np.array([0, 1] * 5)
    with pytest.raises(ModuleNotFoundError):
        learner.fit(X, y)


def test_rfoed_with_rf_base_learner_is_unaffected_by_torch_being_unavailable(monkeypatch):
    """The core single-label path (RFOEDClassifier + RFBaseLearner) must
    keep working end-to-end even while torch's import is actively blocked
    -- demonstrating the core/[torch] extra split is real, not incidental."""
    monkeypatch.setattr(builtins, "__import__", _block_import("torch"))

    from freqcascade import RFBaseLearner, RFOEDClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = np.array([0] * 30 + [1] * 20 + [2] * 10)

    clf = RFOEDClassifier(
        base_learner_factory=lambda i: RFBaseLearner(n_estimators=10, random_state=i, n_jobs=1),
        order="frequency", random_state=0,
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    assert len(pred) == len(y)


def test_resampling_baselines_import_fails_cleanly_without_imbalanced_learn(monkeypatch):
    """Unlike torch_ensemble.py, resampling_baselines.py imports imblearn
    eagerly at module scope (by design: it wraps imblearn resamplers
    directly, so there's no meaningful lazy-init to defer) -- so the
    failure surfaces at `import freqcascade.resampling_baselines` time,
    not at first use. Confirms that failure is a plain, legible
    ModuleNotFoundError rather than some harder-to-diagnose error."""
    monkeypatch.setattr(builtins, "__import__", _block_import("imblearn"))
    sys.modules.pop("freqcascade.resampling_baselines", None)

    with pytest.raises(ModuleNotFoundError):
        import freqcascade.resampling_baselines  # noqa: F401


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""Generate and commit the cross-validation fold indices.

Every method in the paper is evaluated on identical folds -- that identity
is what makes the paired significance tests valid. The fold indices are a
pure function of (dataset, CV_SEED), but we commit them to `data/folds/`
anyway so a reviewer reproduces the exact splits even if the fold-generation
code is later refactored.

    python scripts/make_folds.py                 # all datasets
    python scripts/make_folds.py reuters21578 hoc
"""

from __future__ import annotations

import sys

from _shared import CV_REPEATS, CV_SEED, CV_SPLITS, FOLDS_DIR, get_folds

from freqcascade.datasets import REGISTRY, load


def main(names: list[str]) -> None:
    names = names or list(REGISTRY)
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name == "clinc150":
            print("clinc150: fixed split, no CV folds -- skipped")
            continue
        ds = load(name)
        path = FOLDS_DIR / f"{ds.name}.npz"
        if path.exists():
            path.unlink()
        folds = get_folds(ds, path)
        print(f"{ds.name}: {len(folds)} folds "
              f"({CV_REPEATS}x{CV_SPLITS}, seed {CV_SEED}) -> {path.name}")


if __name__ == "__main__":
    main(sys.argv[1:])

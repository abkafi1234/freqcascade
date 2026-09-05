"""Precompute and cache frozen sentence embeddings for every dataset x
encoder, so the benchmark runs don't re-encode. Cache lives at
data/cache/embeddings/<dataset>.<encoder>.npy (git-ignored).

    python scripts/precompute_embeddings.py                       # all x all
    python scripts/precompute_embeddings.py --encoders minilm     # one encoder
    python scripts/precompute_embeddings.py hoc litcovid          # a subset
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from _shared import ENCODERS, EMB_CACHE, embed

from freqcascade.datasets import REGISTRY, load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(REGISTRY))
    ap.add_argument("--encoders", default=",".join(ENCODERS))
    args = ap.parse_args()
    encoders = [e for e in args.encoders.split(",") if e in ENCODERS]

    for name in (args.datasets or REGISTRY):
        ds = load(name)
        for enc in encoders:
            cache = EMB_CACHE / f"{ds.name}.{enc}.npy"
            if cache.exists() and len(np.load(cache, mmap_mode="r")) == ds.n_samples:
                print(f"  {ds.name:22} {enc:11} cached ({cache.stat().st_size // 1024} KB)")
                continue
            t0 = time.perf_counter()
            arr = embed(ds.texts, enc, ds.name)
            print(f"  {ds.name:22} {enc:11} {arr.shape} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()

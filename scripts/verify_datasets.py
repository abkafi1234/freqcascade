"""Fetch every dataset and print a summary, so the data layer can be checked
end to end outside CI (the loaders hit the network / large downloads).

    python scripts/verify_datasets.py                 # all
    python scripts/verify_datasets.py hoc litcovid
"""

from __future__ import annotations

import sys
import time
import traceback

from freqcascade.datasets import REGISTRY, load

EXPECTED = {  # rough sanity bounds, from the paper's dataset table
    "clinc150": dict(n_classes=150, multilabel=False),
    "20newsgroups": dict(n_classes=20, multilabel=False),
    "wos46985": dict(n_classes=134, multilabel=False),
    "ohsumed_23": dict(n_classes=23, multilabel=False),
    "reuters21578": dict(n_classes=90, multilabel=True),
    "hoc": dict(n_classes=10, multilabel=True),
    "litcovid": dict(n_classes=7, multilabel=True),
}


def main(names: list[str]) -> int:
    names = names or list(REGISTRY)
    bad = 0
    for name in names:
        print(f"\n{'=' * 64}\n{name}\n{'=' * 64}", flush=True)
        t0 = time.perf_counter()
        try:
            kw = dict(min_count=20, subsample=40000) if name == "drug_reviews" else {}
            ds = load(name, **kw)
            s = ds.summary()
            print(f"OK ({time.perf_counter() - t0:.0f}s)  {s}")
            print(f"   notes: {ds.notes}")
            print(f"   labels: {ds.label_names[:6]}{' ...' if len(ds.label_names) > 6 else ''}")
            print(f"   top frequencies: {ds.class_frequencies[:5].tolist()}")
            print(f"   text[0]: {str(ds.texts[0])[:140]!r}")
            exp = EXPECTED.get(name)
            if exp and (s["n_classes"] != exp["n_classes"] or s["multilabel"] != exp["multilabel"]):
                print(f"   !! MISMATCH vs expected {exp}")
                bad += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAILED ({time.perf_counter() - t0:.0f}s): {type(e).__name__}: {e}")
            traceback.print_exc()
            bad += 1
    print(f"\n{'ok' if not bad else str(bad) + ' problem(s)'}")
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

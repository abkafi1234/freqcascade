"""Emit the LaTeX body of the base-learner pilot table (paper Table 8) from
results/unit_pilot.jsonl, so the table is generated from data rather than
transcribed by hand.

    python scripts/latex_pilot_table.py            # LaTeX body
    python scripts/latex_pilot_table.py --summary  # human-readable summary
"""

from __future__ import annotations

import json
import sys

from _shared import RESULTS_DIR

SRC = RESULTS_DIR / "unit_pilot.jsonl"
PRETTY = {"clinc150": "CLINC150", "20newsgroups_ir50": "20 Newsgroups",
          "wos46985": "WOS46985"}
UNIT_PRETTY = {"frozen_embed_mlp": "Frozen embedding + MLP", "textcnn": "TextCNN"}
UNIT_ORDER = ["frozen_embed_mlp", "textcnn"]
DS_ORDER = ["clinc150", "20newsgroups_ir50", "wos46985"]
SWEEP_DS = ["clinc150", "wos46985"]


def rows():
    out = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "macro_f1" in r:
                out.append(r)
    return out


def main() -> None:
    R = rows()
    unit = {(r["dataset"], r["unit"]): r for r in R if r["arm"] == "unit"}
    sweep = {(r["dataset"], r["n_members"]): r for r in R if r["arm"] == "sweep"}
    # The unit arm's frozen-embedding cell at M=10 and the sweep's M=10 cell are
    # the same computation on the same split. Report one measurement for both so
    # the two halves of the table cannot disagree (the flaw in the original).
    for ds in SWEEP_DS:
        if (ds, 10) in sweep:
            unit[(ds, "frozen_embed_mlp")] = sweep[(ds, 10)]
    Ms = sorted({r["n_members"] for r in R if r["arm"] == "sweep"})

    if "--summary" in sys.argv:
        print("UNIT COMPARISON (M=10, validation split)")
        for ds in DS_ORDER:
            for u in UNIT_ORDER:
                r = unit.get((ds, u))
                if r:
                    print(f"  {PRETTY[ds]:15s} {UNIT_PRETTY[u]:24s} "
                          f"macro-F1 {r['macro_f1']:.3f}  fit {r['fit_s']:7.1f}s  "
                          f"{r['ms_per_member']:7.1f} ms/member  infer {r['infer_s']:.2f}s")
        print("\nENSEMBLE-SIZE SWEEP (frozen embedding + MLP)")
        for M in Ms:
            cells = []
            for ds in SWEEP_DS:
                r = sweep.get((ds, M))
                cells.append(f"{PRETTY[ds]} {r['macro_f1']:.3f}/{r['fit_s']:.1f}s" if r else f"{PRETTY[ds]} --")
            print(f"  M={M:3d}  " + "   ".join(cells))
        print("\nCONSISTENCY CHECK (unit arm vs sweep at M=10, same split, same code path)")
        for ds in SWEEP_DS:
            a, b = unit.get((ds, "frozen_embed_mlp")), sweep.get((ds, 10))
            if a and b:
                ok = abs(a["macro_f1"] - b["macro_f1"]) < 1e-9
                print(f"  {PRETTY[ds]:15s} unit {a['macro_f1']:.4f}  sweep {b['macro_f1']:.4f}  "
                      f"{'IDENTICAL' if ok else 'DIFFER'}")
        return

    # --- LaTeX body ---
    for ds in DS_ORDER:
        for u in UNIT_ORDER:
            r = unit.get((ds, u))
            if not r:
                continue
            best = max(unit[(ds, v)]["macro_f1"] for v in UNIT_ORDER if (ds, v) in unit)
            f1 = f"\\textbf{{{r['macro_f1']:.3f}}}" if r["macro_f1"] == best else f"{r['macro_f1']:.3f}"
            print(f"{UNIT_PRETTY[u]} & {PRETTY[ds]} & {f1} & {r['fit_s']:.1f}s & "
                  f"{r['ms_per_member']:.1f}ms & {r['infer_s']:.2f}s \\\\")
    print("\\midrule")
    setstr = ",".join(str(m) for m in Ms)
    print(f"\\multicolumn{{6}}{{l}}{{\\emph{{Ensemble-size sweep, RFOED-NN (frozen embedding + MLP), "
          f"$M\\in\\{{{setstr}\\}}$ (macro-F1 / fit time, CLINC150 -- WOS46985):}}}} \\\\")
    for M in Ms:
        c = []
        for ds in SWEEP_DS:
            r = sweep.get((ds, M))
            c.append(f"{r['macro_f1']:.3f} / {r['fit_s']:.1f}s" if r else "---")
        print(f"\\multicolumn{{2}}{{l}}{{$M$={M}}} & \\multicolumn{{2}}{{l}}{{{c[0]}}} & "
              f"\\multicolumn{{2}}{{l}}{{{c[1]}}} \\\\")


if __name__ == "__main__":
    main()

"""What is done and what is left in the round-3 experiment campaign.

Reads results/*.jsonl and reports completion against each experiment's full
target, so a resumed session (on any machine) can see the state without
re-deriving it. Read-only: runs nothing.

    python scripts/resume_status.py
"""

from __future__ import annotations

import collections
import json

from _shared import RESULTS_DIR, SINGLE_LABEL_METHODS

MAIN_SL = {"20newsgroups_ir50": "20newsgroups", "ohsumed_23": "ohsumed_23",
           "wos46985": "wos46985", "drug_reviews": "drug_reviews"}


def rows(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def line(tag, done, target, note=""):
    pct = 100.0 * done / target if target else 0.0
    state = "DONE" if done >= target else ("not started" if done == 0 else "partial")
    print("  %-34s %5d / %-5d %5.1f%%  %-11s %s" % (tag, done, target, pct, state, note))


def main():
    print("Round-3 campaign status (from %s)\n" % RESULTS_DIR)

    d = rows("cascade_dependence.jsonl")
    cfg = {(r["dataset"], r["base"]) for r in d if r.get("kind") == "summary"}
    line("R6 dependence (configs)", len(cfg), 6, "done: " + ", ".join("%s/%s" % c for c in sorted(cfg)))

    p = rows("focc_perlink.jsonl")
    g = collections.defaultdict(list)
    for r in p:
        g[(r["dataset"], r["order"], r["fold"])].append(r)
    complete = [k for k, v in g.items() if len(v) == v[0]["n_links"]]
    line("R9 per-link exposure (folds)", len(complete), 60, "3 corpora x 2 orders x 10 folds")

    w = rows("weaklearner_sweep.jsonl")
    sweep = [r for r in w if r["protocol"] == "class-count sweep"]
    trees = [r for r in sweep if r["learner"] in ("stump", "tree3", "tree10", "treefull")]
    nets = [r for r in sweep if r["learner"] in ("mlp", "xgb")]
    line("R5 sweep, tree learners (cells)", len(trees), 4 * 3 * 2 * 6 * 3)
    line("R5 sweep, MLP/XGBoost (cells)", len(nets), 2 * 3 * 2 * 6 * 3)
    native = [r for r in w if r["protocol"].startswith("native")]
    print("  %-34s %5d rows  (design decided after the sweep; plan section 5 R5)" % ("R5 native-K check", len(native)))

    n = rows("ncap_adaptive.jsonl")
    line("R8 adaptive n_cap (cells)", len(n), 8 * 5 * 6, "8 corpora x 5 folds x 6 arms")

    lk = rows("leak_check.jsonl")
    have = collections.defaultdict(set)
    for r in lk:
        have[r["dataset"]].add(r["method"])
    rf_names = {m.name for m in SINGLE_LABEL_METHODS().values() if m.kind == "rf"}
    tgt = done = 0
    notes = []
    for ds in MAIN_SL:
        main = {r.get("method") for r in rows(ds + ".jsonl")} & rf_names
        tgt += len(main)
        done += len(main & have[ds])
        notes.append("%s %d/%d" % (ds, len(main & have[ds]), len(main)))
    line("B6 TF-IDF leak check (blocks)", done, tgt, "; ".join(notes))

    ft = rows("finetune_grid.jsonl")
    line("R7 DeBERTa fine-tuning grid", len(ft), 1,
         "script scripts/run_finetune_grid.py still to be written (plan section 5 R7)" if not ft else "")
    print("\nNot experiments, still to do: D2 decision after R8, then the writing phase (plan sections 6-8).")


if __name__ == "__main__":
    main()

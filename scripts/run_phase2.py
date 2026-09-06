"""Orchestrate the whole benchmark campaign as one resumable job, one heavy
process at a time (so RF_NJOBS can use the whole box without oversubscription).

    FREQCASCADE_RF_NJOBS=20 python scripts/run_phase2.py
    python scripts/run_phase2.py --from nn_sl        # resume from a stage
    python scripts/run_phase2.py --only rq5          # a single stage
    python scripts/run_phase2.py --list

Completed stages are recorded in results/.phase2_done; a finished stage is
skipped on a re-run unless --force.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DONE = HERE.parent / "results" / ".phase2_done"

SL = ["clinc150", "20newsgroups", "wos46985", "ohsumed_23", "drug_reviews"]
ML = ["reuters21578", "hoc", "litcovid"]
RFOED_RF_SL = ["20newsgroups", "ohsumed_23", "clinc150", "wos46985"]  # not drug_reviews (355 nodes)

# Full baseline set for the small / low-K datasets that match the paper's
# main table; the lean set (the essential imbalance comparison + the
# "collapses at high K" pair) for the large / high-K ones, where the O(K)
# and oversample-blowup baselines cost hours for little marginal insight.
FLAT_FULL = ("Flat-RF,Flat-RF+Undersample,Flat-RF+Oversample,Flat-RF+SMOTE,"
             "Flat-RF+ADASYN,Flat-RF+SMOTE+ENN,Flat Balanced-RF,OVR-RF,EasyEnsemble,RUSBoost")
FLAT_LEAN = "Flat-RF,Flat-RF+SMOTE,Flat Balanced-RF,OVR-RF,EasyEnsemble,RUSBoost"
# drug_reviews at K=356: drop the O(K) serial baselines (OVR-RF = 356 forests,
# RUSBoost = single-threaded AdaBoost). Keep the imbalance comparison + EasyEnsemble.
FLAT_DRUG = "Flat-RF,Flat-RF+SMOTE,Flat Balanced-RF,EasyEnsemble"
FLAT_FULL_SL = ["clinc150", "20newsgroups"]
FLAT_LEAN_SL = ["wos46985", "ohsumed_23"]

STAGES: list[tuple[str, list[str]]] = [
    ("embeddings",   ["precompute_embeddings.py"]),
    ("flat_sl_full", ["run_single_label_benchmark.py", "--methods", FLAT_FULL, *FLAT_FULL_SL]),
    ("flat_sl_lean", ["run_single_label_benchmark.py", "--methods", FLAT_LEAN, *FLAT_LEAN_SL]),
    ("flat_sl_drug", ["run_single_label_benchmark.py", "--methods", FLAT_DRUG, "drug_reviews"]),
    ("rf_ml",        ["run_multilabel_benchmark.py", "--kind", "rf", *ML]),
    ("rfoed_rf",     ["run_single_label_benchmark.py", "--methods", "RFOED-RF", *RFOED_RF_SL]),
    ("nn_sl",        ["run_single_label_benchmark.py", "--kind", "nn", *SL]),
    ("nn_ml",        ["run_multilabel_benchmark.py", "--kind", "nn", *ML]),
    # the 3-way ANOVA reproduces the paper's original single-label set; the
    # multi-label 2x2 is the new n=3 RQ3 test.
    ("factorial_sl", ["run_factorial_ablation.py", "20newsgroups", "clinc150", "wos46985"]),
    ("factorial_ml", ["run_factorial_ablation.py", "--multilabel", *ML]),
    ("rq5",          ["run_representation_study.py"]),
    ("wos_remed",    ["run_wos_remediation.py"]),
    ("diagnose_wos", ["diagnose_wos.py"]),
    ("perclass",     ["dump_per_class_recall.py", "clinc150", "drug_reviews"]),
    ("figures",      ["make_figures.py"]),
]


def done_set() -> set[str]:
    return set(DONE.read_text().split()) if DONE.exists() else set()


def mark(stage: str) -> None:
    DONE.parent.mkdir(parents=True, exist_ok=True)
    with DONE.open("a") as fh:
        fh.write(stage + "\n")


def run_stage(name: str, cmd: list[str]) -> bool:
    print(f"\n{'#' * 70}\n# {name}   {time.strftime('%H:%M:%S')}\n{'#' * 70}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-u", str(HERE / cmd[0]), *cmd[1:]], cwd=HERE)
    dt = time.perf_counter() - t0
    ok = r.returncode == 0
    print(f"# {name}: {'OK' if ok else 'FAILED (rc=%d)' % r.returncode} in {dt / 60:.1f} min", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        d = done_set()
        for n, _ in STAGES:
            print(f"  [{'x' if n in d else ' '}] {n}")
        return

    stages = STAGES
    if args.only:
        stages = [s for s in STAGES if s[0] == args.only]
    elif args.start:
        names = [n for n, _ in STAGES]
        stages = STAGES[names.index(args.start):]

    d = done_set()
    for name, cmd in stages:
        if name in d and not args.force and not args.only:
            print(f"skip {name} (done)")
            continue
        if not run_stage(name, cmd):
            print(f"\nSTOPPED at {name}. Fix, then: python scripts/run_phase2.py --from {name}")
            sys.exit(1)
        mark(name)
    print("\nall stages complete")


if __name__ == "__main__":
    main()

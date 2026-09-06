"""Shared configuration for every benchmark script: the exact method set,
featurizers, encoder registry, and the fit/predict/score loop, so all
scripts agree on what "RFOED-NN" or "Flat-RF+SMOTE" means and every method
in a comparison sees identical folds and features.

This module is the machine-readable version of the paper's Experimental
Setup section.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FOLDS_DIR = REPO_ROOT / "data" / "folds"
EMB_CACHE = REPO_ROOT / "data" / "cache" / "embeddings"

CV_SEED = 20260905          # one seed drives every fold split in the paper
CV_REPEATS, CV_SPLITS = 5, 2
N_ESTIMATORS = 200          # RF trees / node
# The base-learner pilot (paper Table 8) shows macro-F1 saturates by K~=10-25
# members while cost keeps growing near-linearly, so the revision runs the
# neural ensemble at 25 members / 150 epochs (was 50 / 250). Override with
# FREQCASCADE_NN_MEMBERS / _NN_EPOCHS to reproduce the original setting.
NN_MEMBERS = int(os.environ.get("FREQCASCADE_NN_MEMBERS", "25"))
NN_EPOCHS = int(os.environ.get("FREQCASCADE_NN_EPOCHS", "150"))
RF_NJOBS = int(os.environ.get("FREQCASCADE_RF_NJOBS", "12"))  # RFOED fits nodes serially, so spend cores here

# --- RQ5: frozen sentence encoders -------------------------------------------
ENCODERS = {
    "minilm": dict(model_name="all-MiniLM-L6-v2", chunk_words=200),
    "pubmedbert": dict(model_name="pritamdeka/S-PubMedBert-MS-MARCO", chunk_words=200),
    "mpnet": dict(model_name="all-mpnet-base-v2", chunk_words=200),   # general-domain capacity control
}
DEFAULT_ENCODER = "minilm"


# --------------------------------------------------------------------------- #
# Featurization
# --------------------------------------------------------------------------- #

# 10k features (was 20k): a Random Forest on text TF-IDF plateaus well below
# this, and the sparse matrix is half the size -- material at 40-50k rows.
TFIDF_MAX_FEATURES = int(os.environ.get("FREQCASCADE_TFIDF_FEATURES", "10000"))
# Over-samplers (SMOTE / ADASYN / random oversample) raise minority classes
# only to min(majority, OVERSAMPLE_CAP_MULT * median). No-op on near-balanced
# data; on a long tail it is a gentler rebalancing and keeps synthesis from
# generating millions of rows at K in the hundreds.
OVERSAMPLE_CAP_MULT = float(os.environ.get("FREQCASCADE_OVERSAMPLE_CAP", "4.0"))


def make_tfidf(texts_train, texts_all):
    """Sparse TF-IDF, fit on the training rows only."""
    from freqcascade.features import TfidfFeaturizer

    f = TfidfFeaturizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    f.fit(list(texts_train))
    return f.transform(list(texts_all))


def embed(texts, encoder: str, dataset_name: str):
    """Frozen sentence embeddings, cached to disk per (dataset, encoder).
    Embeddings don't depend on the fold split, so they're computed once."""
    from freqcascade.features import SentenceEmbeddingFeaturizer

    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    cache = EMB_CACHE / f"{dataset_name}.{encoder}.npy"
    if cache.exists():
        arr = np.load(cache)
        if len(arr) == len(texts):
            return arr
    f = SentenceEmbeddingFeaturizer(**ENCODERS[encoder])
    arr = np.asarray(f.fit_transform(list(texts)), dtype=np.float32)
    np.save(cache, arr)
    return arr


# --------------------------------------------------------------------------- #
# Method registry
# --------------------------------------------------------------------------- #

@dataclass
class Method:
    name: str
    kind: str                       # "rf" (TF-IDF) or "nn" (embeddings)
    build: object                   # (seed) -> estimator with fit/predict[/predict_proba]
    multilabel: bool = False
    tags: tuple = ()


def _rf_node_factory(seed, rebalance=True):
    from freqcascade.base_learners import RFBaseLearner
    return lambda node: RFBaseLearner(
        n_estimators=N_ESTIMATORS, rebalance=rebalance, random_state=seed * 1000 + node, n_jobs=RF_NJOBS
    )


def _nn_node_factory(seed, rebalance=True):
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner
    return lambda node: TorchNNEnsembleBaseLearner(
        n_members=NN_MEMBERS, rebalance=rebalance, hidden_size=128,
        max_epochs=NN_EPOCHS, random_state=seed * 1000 + node,
    )


def _single_label_methods() -> dict[str, Method]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier

    from freqcascade.decomposition import RFOEDClassifier
    from freqcascade.resampling_baselines import (
        ResamplingBaseline,
        easy_ensemble_classifier,
        rusboost_classifier,
    )

    m: dict[str, Method] = {}

    m["RFOED-RF"] = Method("RFOED-RF", "rf", lambda s: RFOEDClassifier(
        base_learner_factory=_rf_node_factory(s), order="frequency", random_state=s), tags=("proposed",))
    m["RFOED-NN"] = Method("RFOED-NN", "nn", lambda s: RFOEDClassifier(
        base_learner_factory=_nn_node_factory(s), order="frequency", random_state=s), tags=("proposed",))

    for key, rs in [("Flat-RF", "none"), ("Flat-RF+Undersample", "random_undersample"),
                    ("Flat-RF+Oversample", "random_oversample"), ("Flat-RF+SMOTE", "smote"),
                    ("Flat-RF+ADASYN", "adasyn_floored_minority"), ("Flat-RF+SMOTE+ENN", "smoteenn")]:
        m[key] = Method(key, "rf", lambda s, rs=rs: ResamplingBaseline(
            resampler_name=rs, n_estimators=N_ESTIMATORS, n_jobs=RF_NJOBS, random_state=s,
            oversample_cap_mult=OVERSAMPLE_CAP_MULT))

    m["Flat Balanced-RF"] = Method("Flat Balanced-RF", "rf", lambda s: ResamplingBaseline(
        resampler_name="none",
        classifier_factory=lambda: RandomForestClassifier(
            n_estimators=N_ESTIMATORS, class_weight="balanced_subsample", n_jobs=RF_NJOBS, random_state=s),
    ))
    m["OVR-RF"] = Method("OVR-RF", "rf", lambda s: OneVsRestClassifier(
        RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=2, random_state=s), n_jobs=RF_NJOBS))
    m["EasyEnsemble"] = Method("EasyEnsemble", "rf", lambda s: easy_ensemble_classifier(random_state=s, n_jobs=RF_NJOBS))
    m["RUSBoost"] = Method("RUSBoost", "rf", lambda s: rusboost_classifier(n_estimators=30, random_state=s))
    return m


def _multi_label_methods() -> dict[str, Method]:
    from freqcascade.focc import make_focc_nn, make_focc_rf
    from freqcascade.multilabel_baselines import (
        make_balanced_br_rf,
        make_br_rf,
        make_cc_rf,
        make_ecc_rf,
    )

    m: dict[str, Method] = {}
    m["FOCC-RF"] = Method("FOCC-RF", "rf", lambda s: make_focc_rf(
        order="frequency", rebalance=True, random_state=s, n_estimators=N_ESTIMATORS, n_jobs=RF_NJOBS), multilabel=True, tags=("proposed",))
    m["FOCC-NN"] = Method("FOCC-NN", "nn", lambda s: make_focc_nn(
        order="frequency", rebalance=True, random_state=s, n_members=NN_MEMBERS,
        hidden_size=128, max_epochs=NN_EPOCHS), multilabel=True, tags=("proposed",))
    m["Binary Relevance-RF"] = Method("Binary Relevance-RF", "rf",
        lambda s: make_br_rf(random_state=s, n_estimators=N_ESTIMATORS, n_jobs=RF_NJOBS), multilabel=True)
    m["Balanced BR-RF"] = Method("Balanced BR-RF", "rf",
        lambda s: make_balanced_br_rf(random_state=s, n_estimators=N_ESTIMATORS, n_jobs=RF_NJOBS), multilabel=True)
    m["Classifier Chains-RF"] = Method("Classifier Chains-RF", "rf",
        lambda s: make_cc_rf(random_state=s, n_estimators=N_ESTIMATORS, n_jobs=RF_NJOBS), multilabel=True)
    m["Ensemble of Chains-RF"] = Method("Ensemble of Chains-RF", "rf",
        lambda s: make_ecc_rf(random_state=s, n_estimators=N_ESTIMATORS, n_jobs=RF_NJOBS), multilabel=True)
    return m


SINGLE_LABEL_METHODS = _single_label_methods
MULTI_LABEL_METHODS = _multi_label_methods


# --------------------------------------------------------------------------- #
# Fold cache + run loop
# --------------------------------------------------------------------------- #

def get_folds(ds, path: Path | None = None):
    """Load committed folds if present, else derive them from the seed and
    (optionally) save. Guarantees every method sees identical folds."""
    from freqcascade import cv

    path = path or (FOLDS_DIR / f"{ds.name}.npz")
    if path.exists():
        return cv.load_folds(path)
    if ds.is_multilabel:
        folds = cv.repeated_iterative_stratified_kfold(
            ds.target, n_repeats=CV_REPEATS, n_splits=CV_SPLITS, seed=CV_SEED)
    else:
        folds = cv.repeated_stratified_kfold(
            ds.target, n_repeats=CV_REPEATS, n_splits=CV_SPLITS, seed=CV_SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.save_folds(folds, path)
    return folds


def score(method: Method, X, y, folds, fixed_split=None):
    """Fit + evaluate `method` over every fold (or the fixed split, repeated
    over seeds for CLINC150). Returns a list of per-fold metric dicts."""
    from freqcascade.metrics import evaluate
    from freqcascade.multilabel_metrics import evaluate_multilabel

    rows = []
    iterator = (
        [(s, np.flatnonzero(fixed_split != "test"), np.flatnonzero(fixed_split == "test"))
         for s in range(10)]
        if fixed_split is not None
        else [(i, f.train_idx, f.test_idx) for i, f in enumerate(folds)]
    )
    for seed, tr, te in iterator:
        est = method.build(seed)
        t0 = time.perf_counter()
        est.fit(X[tr], y[tr])
        pred = est.predict(X[te])
        dt = time.perf_counter() - t0
        if method.multilabel:
            met = evaluate_multilabel(y[te], pred, y[tr])
        else:
            met = evaluate(y[te], pred, y[tr])
        met.update(method=method.name, fold=seed, fit_predict_s=round(dt, 2))
        rows.append(met)
    return rows


def already_done(name: str, method_name: str, min_rows: int = 6, encoder: str | None = None) -> bool:
    """True if results/<name>.jsonl already holds >= min_rows non-error rows for
    this (method, encoder) -- lets a re-run skip completed cells."""
    out = RESULTS_DIR / f"{name}.jsonl"
    if not out.exists():
        return False
    k = 0
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("method") == method_name and "error" not in r and r.get("encoder") == encoder:
            k += 1
    return k >= min_rows


def write_results(rows: list[dict], name: str):
    """Merge into results/<name>.jsonl: rows for any (method, encoder) present
    in `rows` replace what's on disk, others are kept. Lets the benchmark be
    run in pieces (--kind rf, then --kind nn, then a re-run of one method)
    without clobbering earlier output."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.jsonl"

    def key(r):
        return (r.get("method"), r.get("encoder"))

    incoming_keys = {key(r) for r in rows}
    kept = []
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if key(r) not in incoming_keys:
                    kept.append(r)
    merged = kept + rows
    with out.open("w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows ({len(merged)} total) -> {out}")
    return out

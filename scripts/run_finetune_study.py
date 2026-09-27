"""Does the decomposition result survive a *dynamic* representation? (R1)

The paper's benchmark runs every method on a frozen representation (TF-IDF or a
frozen sentence encoder), on the argument that freezing the features isolates
imbalance handling from representation learning. The obvious objection is that
modern text classification fine-tunes the encoder, so a result obtained on
frozen vectors may be an artifact of the freezing rather than a fact about
imbalance.

This script tests that directly. Per dataset, on one deterministic split (the
cross-validation test partitions are never touched):

  1. Fine-tune all-MiniLM-L6-v2 end-to-end on the K-class problem, with plain
     and with class-weighted cross-entropy. These are the modern baselines the
     frozen pipeline is accused of ignoring, and they set the absolute ceiling.
  2. Freeze the fine-tuned encoder, mean-pool its hidden states, and run on
     those features the same method set the paper runs on frozen features.

Arm 2 is the actual test. The paper's claim is about how the rebalancing
sub-problem is *posed*, which should be orthogonal to the representation
underneath. If the claim is right, the pattern (resampling ensembles collapse
at high K; local decomposition stays well-posed) reappears on the fine-tuned
features. If it holds only on frozen vectors, the claim is narrower than the
paper states and the paper has to say so.

    python scripts/run_finetune_study.py --probe clinc150
    python scripts/run_finetune_study.py

Writes results/finetune_study.jsonl (resumable per dataset/arm/method).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from _shared import (
    DEFAULT_ENCODER,
    N_ESTIMATORS,
    NN_EPOCHS,
    NN_MEMBERS,
    RESULTS_DIR,
    RF_NJOBS,
    embed,
)

from freqcascade.datasets import load
from freqcascade.decomposition import RFOEDClassifier
from freqcascade.metrics import evaluate

SPLIT_SEED = 20260914          # same split seed as the base-learner pilot
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASETS = ["clinc150", "20newsgroups", "wos46985"]
MAX_LEN = {"clinc150": 64, "20newsgroups": 256, "wos46985": 256}
# The fine-tuned arm is the strong modern baseline this study is accused of
# ignoring, so it has to actually be strong: an undertrained transformer would
# rig the comparison in the frozen pipeline's favour. A first pass at 4 epochs /
# 3e-5 left CLINC150 at training loss 3.06 and macro-F1 0.667, below the frozen
# encoder -- clearly unconverged for a 22M-parameter model over 150 classes.
# These settings train to convergence instead; run_finetune_study reports the
# per-epoch loss so the reader can check that it has flattened.
FT_EPOCHS = 12
FT_LR = 5e-5
FT_BATCH = 32
OUT = RESULTS_DIR / "finetune_study.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def split_of(ds):
    """CLINC150's native validation partition; stratified 80/20 elsewhere."""
    y = np.asarray(ds.target)
    if ds.split is not None and "val" in set(np.asarray(ds.split).tolist()):
        s = np.asarray(ds.split)
        return np.flatnonzero(s == "train"), np.flatnonzero(s == "val"), "native val split"
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=SPLIT_SEED, stratify=y
    )
    return tr, te, "stratified 80/20 @ %d" % SPLIT_SEED


# --------------------------------------------------------------------------- #
# Arm 1: end-to-end fine-tuning
# --------------------------------------------------------------------------- #

def finetune(texts, y_tr, n_classes, max_len, class_weighted, seed):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=n_classes
    ).to(DEV)

    enc = tok(list(texts), truncation=True, max_length=max_len,
              padding="max_length", return_tensors="pt")
    tds = torch.utils.data.TensorDataset(
        enc["input_ids"], enc["attention_mask"], torch.tensor(y_tr, dtype=torch.long)
    )
    dl = torch.utils.data.DataLoader(tds, batch_size=FT_BATCH, shuffle=True)

    if class_weighted:
        cnt = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
        w = np.where(cnt > 0, len(y_tr) / (n_classes * np.maximum(cnt, 1)), 0.0)
        lossf = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=DEV))
    else:
        lossf = torch.nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=FT_LR, total_steps=FT_EPOCHS * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))

    model.train()
    for ep in range(FT_EPOCHS):
        tot = 0.0
        for ids, am, yb in dl:
            ids, am, yb = ids.to(DEV), am.to(DEV), yb.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
                loss = lossf(model(input_ids=ids, attention_mask=am).logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(yb)
        print("      epoch %d/%d  loss=%.4f" % (ep + 1, FT_EPOCHS, tot / len(y_tr)), flush=True)
    return model, tok


@torch.no_grad()
def predict_flat(model, tok, texts, max_len, batch=128):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        e = tok(list(texts[i:i + batch]), truncation=True, max_length=max_len,
                padding=True, return_tensors="pt").to(DEV)
        with torch.amp.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            out.append(model(**e).logits.float().argmax(-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_with(model, tok, texts, max_len, batch=128):
    """Mean-pooled hidden states of the fine-tuned encoder body: the same
    pooling the frozen pipeline uses, so the two feature sets differ only in
    whether the encoder has seen the task."""
    model.eval()
    body = model.base_model
    out = []
    for i in range(0, len(texts), batch):
        e = tok(list(texts[i:i + batch]), truncation=True, max_length=max_len,
                padding=True, return_tensors="pt").to(DEV)
        with torch.amp.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            h = body(**e).last_hidden_state.float()
        m = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# --------------------------------------------------------------------------- #
# Arm 2: the paper's method set, on whichever features it is handed
# --------------------------------------------------------------------------- #

def downstream_methods():
    from sklearn.ensemble import RandomForestClassifier

    from freqcascade.resampling_baselines import (
        ResamplingBaseline,
        easy_ensemble_classifier,
        rusboost_classifier,
    )
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    def nn_fac(seed):
        return lambda node: TorchNNEnsembleBaseLearner(
            n_members=NN_MEMBERS, rebalance=True, hidden_size=128,
            max_epochs=NN_EPOCHS, random_state=seed * 1000 + node)

    return {
        "RFOED-NN": lambda s: RFOEDClassifier(
            base_learner_factory=nn_fac(s), order="frequency", random_state=s),
        "Flat-RF": lambda s: ResamplingBaseline(
            resampler_name="none", n_estimators=N_ESTIMATORS,
            n_jobs=RF_NJOBS, random_state=s),
        "Flat-RF+SMOTE": lambda s: ResamplingBaseline(
            resampler_name="smote", n_estimators=N_ESTIMATORS, n_jobs=RF_NJOBS,
            random_state=s, oversample_cap_mult=4.0),
        "Flat Balanced-RF": lambda s: ResamplingBaseline(
            resampler_name="none",
            classifier_factory=lambda: RandomForestClassifier(
                n_estimators=N_ESTIMATORS, class_weight="balanced_subsample",
                n_jobs=RF_NJOBS, random_state=s)),
        "EasyEnsemble": lambda s: easy_ensemble_classifier(
            random_state=s, n_jobs=RF_NJOBS),
        "RUSBoost": lambda s: rusboost_classifier(n_estimators=30, random_state=s),
    }


def done():
    if not OUT.exists():
        return set()
    rows = (json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip())
    return {(r["dataset"], r["arm"], r["method"]) for r in rows}


def emit(row):
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def run(names, probe=False, seed=0):
    seen = done()
    for name in names:
        ds = load(name)
        tr, te, how = split_of(ds)
        y = np.asarray(ds.target)
        classes, y_ix = np.unique(y, return_inverse=True)
        K = int(len(classes))
        print("\n=== %s === K=%d  n_train=%d  n_test=%d  (%s)"
              % (name, K, len(tr), len(te), how), flush=True)
        mx = MAX_LEN.get(name, 256)

        # ---- arm 1: end-to-end fine-tuning --------------------------------
        ft_model = ft_tok = None
        for cw, arm in [(False, "ft-endtoend"), (True, "ft-endtoend-classweighted")]:
            if (ds.name, arm, "FineTuned-MiniLM") in seen:
                continue
            print("    fine-tuning (class_weighted=%s)..." % cw, flush=True)
            t0 = time.perf_counter()
            model, tok = finetune(ds.texts[tr], y_ix[tr], K, mx, cw, seed)
            pred_ix = predict_flat(model, tok, ds.texts[te], mx)
            dt = time.perf_counter() - t0
            met = evaluate(y[te], classes[pred_ix], y[tr])
            met.update(dataset=ds.name, arm=arm, method="FineTuned-MiniLM", fold=seed,
                       fit_predict_s=round(dt, 1), K=K, max_len=mx,
                       epochs=FT_EPOCHS, split=how)
            emit(met)
            print("      -> macroF1=%.3f  (%.0fs)" % (met["macro_f1"], dt), flush=True)
            if cw:
                del model
            else:
                ft_model, ft_tok = model, tok    # reuse the plain one for features
            torch.cuda.empty_cache()

        # ---- features: frozen vs fine-tuned -------------------------------
        feats = {"frozen": embed(ds.texts, DEFAULT_ENCODER, ds.name)}
        need_ft = any((ds.name, "finetuned-features", m) not in seen
                      for m in downstream_methods())
        if need_ft:
            if ft_model is None:
                print("    re-fine-tuning to extract features...", flush=True)
                ft_model, ft_tok = finetune(ds.texts[tr], y_ix[tr], K, mx, False, seed)
            feats["finetuned"] = encode_with(ft_model, ft_tok, ds.texts, mx)
        ft_model = ft_tok = None
        torch.cuda.empty_cache()

        # ---- arm 2: the same method set on each feature set ----------------
        for featname, X in feats.items():
            arm = "frozen-features" if featname == "frozen" else "finetuned-features"
            for mname, build in downstream_methods().items():
                if (ds.name, arm, mname) in seen:
                    continue
                t0 = time.perf_counter()
                try:
                    est = build(seed)
                    est.fit(X[tr], y[tr])
                    met = evaluate(y[te], est.predict(X[te]), y[tr])
                except Exception as e:
                    met = dict(error="%s: %s" % (type(e).__name__, e),
                               macro_f1=float("nan"))
                met.update(dataset=ds.name, arm=arm, method=mname, fold=seed,
                           fit_predict_s=round(time.perf_counter() - t0, 1),
                           K=K, split=how)
                emit(met)
                shown = ("%.3f" % met["macro_f1"]) if met["macro_f1"] == met["macro_f1"] \
                    else str(met.get("error", "nan"))[:60]
                print("    %-20s %-18s macroF1=%s" % (arm, mname, shown), flush=True)
                if probe:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run(a.datasets or DATASETS, probe=a.probe)


if __name__ == "__main__":
    main()

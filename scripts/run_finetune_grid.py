"""End-to-end fine-tuning grid across all eight corpora (round-3 R7).

The reviewer's objection is that the paper's fine-tuning claim rests on three
corpora, one split and one small encoder, while the abstract states it as a
general result. `run_finetune_study.py` is that smaller study; this script is
the confirmatory version of it:

  encoder   microsoft/deberta-v3-base, fine-tuned end-to-end on the task
  corpora   all eight
  folds     0, 1, 2 of the committed 5x2 CV (data/folds/<corpus>.npz)
  losses    single-label  cross-entropy | class-weighted CE | focal (gamma=2)
            multi-label   BCE | positive-weighted BCE | focal BCE

Per (corpus, loss, fold) it records two arms, exactly as the smaller study does:

  ft-endtoend         the fine-tuned encoder used directly as a classifier,
                      which is what modern practice achieves;
  finetuned-features  the paper's own method set re-run on mean-pooled hidden
                      states of that same encoder. This is the actual test: the
                      study's claim concerns how the rebalancing sub-problem is
                      *posed*, which ought to be orthogonal to the
                      representation underneath it.

Early stopping is on macro-F1 of a 10% validation slice carved out of the
*training* fold, so no test partition is ever seen during model selection.

    python scripts/run_finetune_grid.py --repro-check     # rule 4: reproduce a stored cell first
    python scripts/run_finetune_grid.py --smoke           # HoC fold 0, one loss
    python scripts/run_finetune_grid.py                   # the grid
    python scripts/run_finetune_grid.py clinc150 --folds 0,1,2

Writes results/finetune_grid.jsonl, one row per (corpus, loss, fold, arm,
method), resumable at that granularity: a cell already on disk is skipped and an
exception is recorded as an `error` field rather than ending the run. Mean-pooled
features are cached to data/cache/finetuned/<corpus>.<loss>.fold<k>.npy (several
GB over the whole grid; --discard-features deletes each one once its downstream
methods are done).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from _shared import (
    MULTI_LABEL_METHODS,
    RESULTS_DIR,
    SINGLE_LABEL_METHODS,
    get_folds,
)

from freqcascade.datasets import load
from freqcascade.metrics import evaluate
from freqcascade.multilabel_metrics import evaluate_multilabel

MODEL = "microsoft/deberta-v3-base"
CORPORA = ["clinc150", "20newsgroups", "ohsumed_23", "wos46985", "drug_reviews",
           "reuters21578", "hoc", "litcovid"]
FOLDS = (0, 1, 2)
MAX_LEN = {"clinc150": 64}          # 256 everywhere else
DEFAULT_MAX_LEN = 256
BATCH = 16
LR = 2e-5
MAX_EPOCHS = 5
PATIENCE = 1
# Minimum optimiser steps before early stopping may trigger, and the epoch budget
# is stretched to reach it. The agreed 5 epochs give WOS46985 ~6,500 steps but HoC
# only ~230, and the HoC smoke test showed a multi-label head still predicting no
# positives after 5 epochs (validation macro-F1 0 while the loss kept falling).
# Large corpora are unaffected: their 5 epochs already exceed the minimum.
MIN_STEPS = 1500
# Seconds between progress lines inside the training loop. The loop's only output
# used to be one line per epoch, so when a CUDA call stalled inside it on 2026-09-22
# (main thread parked in loss.item(), 0% GPU utilisation, one core spinning) the run
# looked identical to a slow epoch and sat dead for eight hours. run_r7.sh's stall
# watchdog keys its tight timeout on these lines.
HEARTBEAT_S = 60
WARMUP_FRAC = 0.1
VAL_FRAC = 0.1
FOCAL_GAMMA = 2.0
SINGLE_LOSSES = ("ce", "ce_weighted", "focal")
MULTI_LOSSES = ("bce", "bce_posweight", "focal_bce")

# The method set the grid re-runs on the fine-tuned features. Names index the
# registries in _shared.py, so "RFOED-NN" here means exactly what it means in
# the main benchmark. EasyEnsemble and RUSBoost are multiclass estimators and
# have no multi-label counterpart, hence the shorter multi-label set.
# Flat-MLP baselines added 2026-09-18: on frozen embeddings a plain (class-weighted)
# MLP beat RFOED-NN, so omitting them here would repeat the representation confound.
SL_DOWNSTREAM = ["RFOED-NN", "Flat-MLP", "Flat-MLP (class-weighted)", "Flat-RF+SMOTE", "Flat Balanced-RF",
                 "EasyEnsemble", "RUSBoost"]
ML_DOWNSTREAM = ["FOCC-NN", "Flat-MLP", "Binary Relevance-RF", "Balanced BR-RF", "Ensemble of Chains-RF"]

OUT = RESULTS_DIR / "finetune_grid.jsonl"
ATTEMPTS = RESULTS_DIR / "finetune_attempts.json"
FEAT_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "finetuned"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Result file: resume, append
# --------------------------------------------------------------------------- #

def done_cells():
    if not OUT.exists():
        return set()
    rows = (json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip())
    return {(r["dataset"], r["loss"], r["fold"], r["arm"], r["method"]) for r in rows}


def emit(row):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


# Only the downstream loop catches per-cell errors; a crash or a stall during
# fine-tuning leaves the cell with no row at all, so done_cells() cannot skip it and
# run_r7.sh restarts straight back into it. A deterministic failure (an OOM on a
# large corpus, the 2026-09-22 CUDA stall) therefore livelocks: one cell consumes
# every restart and the corpora after it are never reached. The count is written
# *before* the attempt so it survives the SIGKILL the stall watchdog uses. Abandoned
# cells become ordinary error rows; delete those rows and this file's entry to retry.
MAX_CELL_ATTEMPTS = 3


def attempt_count(key, rows_done, bump=False):
    """Consecutive attempts at `key` that finished no further row. `rows_done` is how
    many of the cell's rows already exist, so an attempt that got some of them written
    before dying resets the budget rather than spending it: only a cell that is making
    no progress at all is abandoned."""
    try:
        d = json.loads(ATTEMPTS.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    k = "%s|%s|%d" % key
    tries, seen_rows = d.get(k, [0, -1])
    if bump:
        tries = 1 if rows_done > seen_rows else tries + 1
        d[k] = [tries, rows_done]
        ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
        ATTEMPTS.write_text(json.dumps(d, indent=0, sort_keys=True), encoding="utf-8")
    return tries


# --------------------------------------------------------------------------- #
# Batching: dynamic padding with length bucketing (see probe_finetune_throughput)
# --------------------------------------------------------------------------- #

def length_bucketed_batches(enc_ids, idx, batch, rng):
    """Shuffle, then sort by length inside large chunks so each batch is nearly
    uniform in length, then shuffle the batch order. Padding is to the longest
    sequence in the batch rather than to max_len, which is where the speed-up
    measured by the throughput probe comes from."""
    idx = np.array(idx)
    rng.shuffle(idx)
    chunk = batch * 50
    idx = np.concatenate([sorted(idx[i:i + chunk], key=lambda j: len(enc_ids[j]))
                          for i in range(0, len(idx), chunk)])
    bl = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    rng.shuffle(bl)
    return bl


def collate(enc_ids, rows, pad_id):
    L = max(len(enc_ids[j]) for j in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(rows), L), dtype=torch.long)
    for i, j in enumerate(rows):
        x = enc_ids[j]
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        am[i, :len(x)] = 1
    return ids, am


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def make_loss(kind, y_train, n_out, multi):
    """Return loss_fn(logits, target). Class weights and positive weights are
    computed from the training fold only."""
    if multi:
        pos = np.asarray(y_train).sum(0).astype(np.float64)
        neg = len(y_train) - pos
        if kind == "bce":
            return lambda lg, t: torch.nn.functional.binary_cross_entropy_with_logits(lg, t)
        if kind == "bce_posweight":
            w = torch.tensor(np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0),
                             dtype=torch.float32, device=DEV).clamp(max=100.0)
            return lambda lg, t: torch.nn.functional.binary_cross_entropy_with_logits(lg, t, pos_weight=w)
        if kind == "focal_bce":
            def focal_bce(lg, t):
                bce = torch.nn.functional.binary_cross_entropy_with_logits(lg, t, reduction="none")
                p_t = torch.exp(-bce)
                return ((1 - p_t) ** FOCAL_GAMMA * bce).mean()
            return focal_bce
        raise ValueError(kind)

    cnt = np.bincount(np.asarray(y_train), minlength=n_out).astype(np.float64)
    if kind == "ce":
        return lambda lg, t: torch.nn.functional.cross_entropy(lg, t)
    if kind == "ce_weighted":
        w = np.where(cnt > 0, len(y_train) / (n_out * np.maximum(cnt, 1.0)), 0.0)
        wt = torch.tensor(w, dtype=torch.float32, device=DEV)
        return lambda lg, t: torch.nn.functional.cross_entropy(lg, t, weight=wt)
    if kind == "focal":
        # alpha from class frequency, as specified: rarer class -> larger alpha.
        a = np.where(cnt > 0, len(y_train) / (n_out * np.maximum(cnt, 1.0)), 0.0)
        at = torch.tensor(a / max(a.max(), 1e-12), dtype=torch.float32, device=DEV)

        def focal(lg, t):
            ce = torch.nn.functional.cross_entropy(lg, t, reduction="none")
            p_t = torch.exp(-ce)
            return (at[t] * (1 - p_t) ** FOCAL_GAMMA * ce).mean()
        return focal
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Validation slice carved out of the training fold
# --------------------------------------------------------------------------- #

def train_val_split(y, train_idx, multi, seed):
    """Stratified 10% validation slice of the training fold. Single-label
    stratification falls back to a plain random slice when some class has too
    few examples to stratify on; multi-label slices at random, since exact
    multi-label stratification of a 10% slice is not worth the dependency here
    (the slice only drives early stopping)."""
    from sklearn.model_selection import train_test_split

    if not multi:
        yt = np.asarray(y)[train_idx]
        _, counts = np.unique(yt, return_counts=True)
        if counts.min() >= 2:
            tr, va = train_test_split(train_idx, test_size=VAL_FRAC, random_state=seed, stratify=yt)
            return np.asarray(tr), np.asarray(va)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    k = max(1, int(round(VAL_FRAC * len(perm))))
    return np.asarray(perm[k:]), np.asarray(perm[:k])


def val_score(logits, y_true, multi):
    from sklearn.metrics import f1_score

    if multi:
        pred = (logits > 0).astype(int)
        return float(f1_score(y_true, pred, average="macro", zero_division=0))
    return float(f1_score(y_true, logits.argmax(1), average="macro", zero_division=0))


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #

def finetune(texts, y, tr, va, n_out, multi, loss_kind, max_len, seed, batch, log):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # dtype=float32 is required, not cosmetic. transformers 5 loads a checkpoint in
    # its stored dtype by default, and deberta-v3-base is stored in float16: AdamW
    # then updates fp16 weights with no fp32 master copy and the loss goes to NaN
    # in the first epoch (caught by the HoC smoke test on 2026-09-18). fp32 master
    # weights with bf16 autocast for the forward pass is the intended setup.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=n_out, dtype=torch.float32,
        problem_type="multi_label_classification" if multi else "single_label_classification",
    ).to(DEV)

    enc = tok(list(texts), truncation=True, max_length=max_len)["input_ids"]
    target = (torch.tensor(np.asarray(y), dtype=torch.float32) if multi
              else torch.tensor(np.asarray(y), dtype=torch.long))
    loss_fn = make_loss(loss_kind, np.asarray(y)[tr], n_out, multi)

    rng = np.random.default_rng(seed)
    steps_per_epoch = int(np.ceil(len(tr) / batch))
    n_epochs = max(MAX_EPOCHS, int(np.ceil(MIN_STEPS / steps_per_epoch)))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total = steps_per_epoch * n_epochs
    warm = max(1, int(WARMUP_FRAC * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    best = (-1.0, None, 0)      # score, state_dict on cpu, epoch
    history = []
    step, last_beat = 0, time.perf_counter()
    for ep in range(1, n_epochs + 1):
        model.train()
        tot = 0.0
        for rows in length_bucketed_batches(enc, tr, batch, rng):
            ids, am = collate(enc, rows, tok.pad_token_id)
            ids, am = ids.to(DEV), am.to(DEV)
            t = target[rows].to(DEV)
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                logits = model(input_ids=ids, attention_mask=am).logits
            loss = loss_fn(logits.float(), t)
            if not torch.isfinite(loss):
                # a diverged encoder yields garbage features that every downstream
                # method would then score near zero without complaint
                raise FloatingPointError("non-finite training loss at epoch %d" % ep)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss.item()) * len(rows)
            step += 1
            if time.perf_counter() - last_beat >= HEARTBEAT_S:
                last_beat = time.perf_counter()
                log("        step %d/%d  (epoch %d)" % (step, total, ep))
        vs = val_score(infer_logits(model, tok, enc, va, batch), np.asarray(y)[va], multi)
        history.append(dict(epoch=ep, train_loss=round(tot / max(1, len(tr)), 4), val_macro_f1=round(vs, 4)))
        log("      epoch %d/%d  loss=%.4f  val_macroF1=%.4f" % (ep, n_epochs, tot / max(1, len(tr)), vs))
        if vs > best[0]:
            best = (vs, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        elif best[0] > 0 and ep * steps_per_epoch >= MIN_STEPS and ep - best[2] >= PATIENCE:
            # Patience only runs once the model predicts anything. Early in training a
            # multi-label head predicts no positives, validation macro-F1 sits at 0,
            # and a strict 0 -> 0 comparison stopped HoC at epoch 2 with an encoder
            # worse than the frozen one (smoke test, 2026-09-18).
            log("      early stop: no improvement for %d epoch(s), best was epoch %d" % (PATIENCE, best[2]))
            break
    if best[0] <= 0:
        log("      WARNING: validation macro-F1 never rose above 0 in %d epochs" % n_epochs)
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, tok, enc, history, best[2]


@torch.no_grad()
def infer_logits(model, tok, enc, rows, batch=64):
    model.eval()
    out = []
    rows = np.asarray(rows)
    order = np.argsort([len(enc[j]) for j in rows])      # bucket for speed, restore order after
    for i in range(0, len(order), batch):
        sel = rows[order[i:i + batch]]
        ids, am = collate(enc, sel, tok.pad_token_id)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            lg = model(input_ids=ids.to(DEV), attention_mask=am.to(DEV)).logits
        out.append(lg.float().cpu().numpy())
    res = np.concatenate(out)
    back = np.empty_like(res)
    back[order] = res
    return back


@torch.no_grad()
def mean_pooled(model, tok, enc, batch=64):
    """Mean-pooled hidden states of the fine-tuned encoder body: the identical
    pooling the frozen pipeline uses, so the feature sets differ only in whether
    the encoder has seen the task."""
    model.eval()
    body = model.base_model
    n = len(enc)
    order = np.argsort([len(x) for x in enc])
    out = None
    for i in range(0, n, batch):
        sel = order[i:i + batch]
        ids, am = collate(enc, sel, tok.pad_token_id)
        ids, am = ids.to(DEV), am.to(DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            h = body(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).float()
        pooled = ((h.float() * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy().astype(np.float32)
        if out is None:
            out = np.empty((n, pooled.shape[1]), dtype=np.float32)
        out[sel] = pooled
    return out


# --------------------------------------------------------------------------- #
# One (corpus, loss, fold) cell
# --------------------------------------------------------------------------- #

def downstream_registry(multi):
    reg = MULTI_LABEL_METHODS() if multi else SINGLE_LABEL_METHODS()
    names = ML_DOWNSTREAM if multi else SL_DOWNSTREAM
    return [(n, reg[n]) for n in names if n in reg]


def run_cell(ds, loss_kind, fold, tr_all, te, seen, batch, discard_features, log):
    multi = bool(ds.is_multilabel)
    y = np.asarray(ds.target)
    n_out = int(y.shape[1]) if multi else int(len(np.unique(y)))
    max_len = MAX_LEN.get(ds.name, DEFAULT_MAX_LEN)
    key = (ds.name, loss_kind, fold)
    methods = downstream_registry(multi)

    need_ft_arm = (key + ("ft-endtoend", "FineTuned-DeBERTa")) not in seen
    missing = [n for n, _ in methods if key + ("finetuned-features", n) not in seen]
    cache = FEAT_CACHE / ("%s.%s.fold%d.npy" % (ds.name, loss_kind, fold))
    have_cache = cache.exists()
    if not need_ft_arm and not missing:
        return

    # Counted for the whole cell, not just fine-tuning: the downstream arm runs
    # RFOED-NN / FOCC-NN on the GPU too, so it can stall in the same way.
    tries = attempt_count(key, (0 if need_ft_arm else 1) + len(methods) - len(missing), bump=True)
    if tries > MAX_CELL_ATTEMPTS:
        for arm, mname in ([("ft-endtoend", "FineTuned-DeBERTa")] if need_ft_arm else []) \
                          + [("finetuned-features", n) for n in missing]:
            emit(dict(error="abandoned: crashed or stalled on %d attempts" % (tries - 1),
                      score=None, dataset=ds.name, loss=loss_kind, fold=fold, arm=arm,
                      method=mname, multilabel=multi, K=n_out, encoder=MODEL))
        log("    ABANDONED %s | %s | fold %d after %d attempts -- error rows written, "
            "continuing with the rest of the grid" % (ds.name, loss_kind, fold, tries - 1))
        return

    if not need_ft_arm and missing and have_cache:
        X = np.load(cache)                      # downstream only; no fine-tuning needed
    else:
        classes = None if multi else np.unique(y)
        y_ix = y if multi else np.searchsorted(classes, y)
        tr, va = train_val_split(y_ix, tr_all, multi, seed=fold)
        log("    fine-tuning %s | %s | fold %d  (n_train=%d, n_val=%d, n_test=%d, max_len=%d)"
            % (ds.name, loss_kind, fold, len(tr), len(va), len(te), max_len))
        t0 = time.perf_counter()
        if DEV == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model, tok, enc, history, best_ep = finetune(
            ds.texts, y_ix, tr, va, n_out, multi, loss_kind, max_len, fold, batch, log)
        ft_secs = time.perf_counter() - t0

        if need_ft_arm:
            log("      scoring the test fold ...")
            lg = infer_logits(model, tok, enc, te, batch=64)
            if multi:
                met = evaluate_multilabel(y[te], (lg > 0).astype(int), y[tr_all])
                met["score"] = met.get("label_macro_f1")
            else:
                met = evaluate(y[te], classes[lg.argmax(1)], y[tr_all])
                met["score"] = met.get("macro_f1")
            met.update(dataset=ds.name, loss=loss_kind, fold=fold, arm="ft-endtoend",
                       method="FineTuned-DeBERTa", multilabel=multi, K=n_out, encoder=MODEL,
                       max_len=max_len, batch=batch, lr=LR, best_epoch=best_ep, history=history,
                       max_epochs=MAX_EPOCHS, patience=PATIENCE, min_steps=MIN_STEPS,
                       fit_predict_s=round(ft_secs, 1),
                       peak_gpu_mem_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                        if DEV == "cuda" else None))
            emit(met)
            log("      -> end-to-end %s macroF1=%.4f  (%.0fs, best epoch %d)"
                % (ds.name, met["score"], ft_secs, best_ep))

        log("      mean-pooling encoder features ...")
        X = mean_pooled(model, tok, enc)
        FEAT_CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()

    for name, method in methods:
        if key + ("finetuned-features", name) in seen:
            continue
        log("      running %s ..." % name)
        t0 = time.perf_counter()
        try:
            est = method.build(fold)
            est.fit(X[tr_all], y[tr_all])
            pred = est.predict(X[te])
            met = evaluate_multilabel(y[te], pred, y[tr_all]) if multi else evaluate(y[te], pred, y[tr_all])
            met["score"] = met.get("label_macro_f1") if multi else met.get("macro_f1")
        except Exception as e:                                   # a failed cell must not end the grid
            met = dict(error="%s: %s" % (type(e).__name__, str(e)[:200]), score=None)
        met.update(dataset=ds.name, loss=loss_kind, fold=fold, arm="finetuned-features",
                   method=name, multilabel=multi, K=n_out, encoder=MODEL,
                   fit_predict_s=round(time.perf_counter() - t0, 1))
        emit(met)
        log("      %-24s %s" % (name, ("%.4f" % met["score"]) if met["score"] is not None
                                else met["error"][:60]))
    if discard_features and cache.exists():
        cache.unlink()


# --------------------------------------------------------------------------- #
# Reproduction check (rule 4) and entry point
# --------------------------------------------------------------------------- #

def repro_check():
    """Everything downstream of the encoder is shared with the main benchmark,
    so it can be checked against a stored number before the grid runs: rebuild
    one main-benchmark cell (HoC, FOCC-NN, fold 0, frozen MiniLM features) with
    this script's own downstream path and compare."""
    from _shared import embed

    ds = load("hoc")
    y = np.asarray(ds.target)
    folds = get_folds(ds)
    tr, te = folds[0].train_idx, folds[0].test_idx
    X = embed(ds.texts, "minilm", ds.name)
    method = MULTI_LABEL_METHODS()["FOCC-NN"]
    est = method.build(0)
    est.fit(X[tr], y[tr])
    got = evaluate_multilabel(y[te], est.predict(X[te]), y[tr])["label_macro_f1"]

    stored = None
    p = RESULTS_DIR / "hoc.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("method") == "FOCC-NN" and r.get("fold") == 0 and "error" not in r:
                stored = r.get("label_macro_f1")
    print("HoC FOCC-NN fold 0 label-macro-F1: recomputed %.7f | stored %s" % (got, stored))
    if stored is None:
        print("REPRO: no stored row found")
        return 1
    d = abs(got - stored)
    print("REPRO: %s (|difference| = %.7f)" % ("EXACT" if d == 0 else ("within 0.002" if d <= 0.002
                                                                       else "MISMATCH"), d))
    return 0 if d <= 0.002 else 1


def main():
    global MAX_EPOCHS, PATIENCE, MIN_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--folds", default=",".join(str(f) for f in FOLDS))
    ap.add_argument("--losses", default=None, help="comma-separated subset of the loss names")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--smoke", action="store_true", help="HoC fold 0, one loss, then stop")
    ap.add_argument("--repro-check", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--min-steps", type=int, default=MIN_STEPS)
    ap.add_argument("--discard-features", action="store_true",
                    help="delete each cached feature matrix once its downstream methods are done")
    a = ap.parse_args()

    if a.repro_check:
        raise SystemExit(repro_check())
    MAX_EPOCHS, PATIENCE, MIN_STEPS = a.max_epochs, a.patience, a.min_steps

    names = a.datasets or CORPORA
    folds = [int(x) for x in a.folds.split(",") if x.strip() != ""]
    if a.smoke:
        names, folds = ["hoc"], [0]

    seen = done_cells()
    for name in names:
        ds = load(name)
        multi = bool(ds.is_multilabel)
        losses = list(MULTI_LOSSES if multi else SINGLE_LOSSES)
        if a.losses:
            losses = [x for x in a.losses.split(",") if x in losses]
        if a.smoke:
            losses = losses[:1]
        cv_folds = get_folds(ds)
        print("\n=== %s === multilabel=%s  folds=%s  losses=%s"
              % (ds.name, multi, folds, losses), flush=True)
        for loss_kind in losses:
            for fold in folds:
                f = cv_folds[fold]
                run_cell(ds, loss_kind, fold, np.asarray(f.train_idx), np.asarray(f.test_idx),
                         seen, a.batch, a.discard_features,
                         log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()

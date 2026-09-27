"""Throughput probe for the round-3 fine-tuning grid (plan R7, decision D1).

Measures training throughput (samples/s, real tokens/s) and peak GPU memory for
candidate encoders on real WOS46985 text, under two batching schemes:

  fixed    every sequence padded to max_len (what run_finetune_study.py does)
  dynamic  length-bucketed batches padded only to the longest in the batch

Then counts real tokens per corpus to turn tokens/s into a wall-clock estimate
for the whole grid. Nothing here touches the CV test folds or writes results
used by the paper; output goes to results/probe_finetune_throughput.json.

    python scripts/probe_finetune_throughput.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from _shared import RESULTS_DIR

from freqcascade.datasets import load

MAX_LEN = 256
BATCH = 16
WARMUP, STEPS = 8, 40
MODELS = ["microsoft/deberta-v3-base", "FacebookAI/roberta-base", "answerdotai/ModernBERT-base"]
CORPORA = ["clinc150", "20newsgroups", "wos46985", "ohsumed_23", "drug_reviews",
           "reuters21578", "hoc", "litcovid"]
DEV = "cuda"


def batches(enc_ids, dynamic, rng):
    idx = np.arange(len(enc_ids))
    if dynamic:
        # sort by length inside shuffled chunks of 50 batches: near-uniform
        # lengths per batch without a fully deterministic order
        rng.shuffle(idx)
        chunk = BATCH * 50
        idx = np.concatenate([sorted(idx[i:i + chunk], key=lambda j: len(enc_ids[j]))
                              for i in range(0, len(idx), chunk)])
        bl = [idx[i:i + BATCH] for i in range(0, len(idx), BATCH)]
        rng.shuffle(bl)
        return bl
    rng.shuffle(idx)
    return [idx[i:i + BATCH] for i in range(0, len(idx), BATCH)]


def collate(ids_list, pad_id, fixed):
    L = MAX_LEN if fixed else max(len(x) for x in ids_list)
    ids = torch.full((len(ids_list), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x)
        am[i, :len(x)] = 1
    return ids, am


def probe(model_name, texts, labels, K):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"]
    out = {}
    for scheme in ("fixed", "dynamic"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        # fp32 master weights, as in run_finetune_grid.py; some checkpoints are stored
        # in fp16, and transformers 5 would otherwise load and train them in fp16
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=K, dtype=torch.float32).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        rng = np.random.default_rng(0)
        bl = batches(enc, scheme == "dynamic", rng)
        model.train()
        n_samp = n_tok = 0
        t0 = None
        for s, b in enumerate(bl[:WARMUP + STEPS]):
            if s == WARMUP:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                n_samp = n_tok = 0
            ids, am = collate([enc[j] for j in b], tok.pad_token_id, scheme == "fixed")
            ids, am = ids.to(DEV), am.to(DEV)
            y = torch.tensor(labels[b], dtype=torch.long, device=DEV)
            with torch.autocast(DEV, dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=am).logits.float(), y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            n_samp += len(b)
            n_tok += int(am.sum())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out[scheme] = dict(samples_per_s=round(n_samp / dt, 1), real_tokens_per_s=round(n_tok / dt),
                           peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                           loss_finite=bool(torch.isfinite(loss).item()))
        print("  %-32s %-8s %7.1f samples/s  %7d tok/s  peak %.2f GB"
              % (model_name, scheme, out[scheme]["samples_per_s"], out[scheme]["real_tokens_per_s"],
                 out[scheme]["peak_mem_gb"]), flush=True)
        del model, opt
    return out


def corpus_tokens(tok_name):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_name)
    res = {}
    for name in CORPORA:
        ds = load(name)
        L = 64 if name == "clinc150" else MAX_LEN
        lens = [min(len(x), L) for x in tok(list(ds.texts), truncation=True, max_length=L)["input_ids"]]
        res[ds.name] = dict(n_docs=len(lens), tokens=int(np.sum(lens)),
                            mean_len=round(float(np.mean(lens)), 1),
                            frac_truncated=round(float(np.mean(np.array(lens) >= L)), 3), max_len=L)
        print("  %-18s docs=%6d  mean_len=%6.1f  truncated=%.1f%%  total_tokens=%d"
              % (ds.name, len(lens), res[ds.name]["mean_len"], 100 * res[ds.name]["frac_truncated"],
                 res[ds.name]["tokens"]), flush=True)
    return res


def main():
    ds = load("wos46985")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ds.texts), size=BATCH * (WARMUP + STEPS) * 2, replace=False)
    texts = np.asarray(ds.texts)[pick]
    classes, y = np.unique(np.asarray(ds.target), return_inverse=True)
    labels = y[pick]
    report = {"setup": dict(max_len=MAX_LEN, batch=BATCH, steps=STEPS, dtype="bf16", corpus="wos46985",
                            gpu=torch.cuda.get_device_name(0))}
    print("== throughput on WOS46985 text ==", flush=True)
    for m in MODELS:
        try:
            report[m] = probe(m, texts, labels, len(classes))
        except Exception as e:
            report[m] = dict(error="%s: %s" % (type(e).__name__, e))
            print("  %s failed: %s" % (m, report[m]["error"][:150]), flush=True)
    print("== real tokens per corpus (DeBERTa-v3 tokenizer) ==", flush=True)
    report["corpus_tokens"] = corpus_tokens("microsoft/deberta-v3-base")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "probe_finetune_throughput.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote results/probe_finetune_throughput.json")


if __name__ == "__main__":
    main()

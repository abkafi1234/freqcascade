"""Does Proposition 1 depend on node decisions being independent? (round-3 R6)

The reviewer's objection: nodes in a sequential peel are not independent, so a
product form R_j = (1-alpha_j) prod_{k<j} (1-beta_kj) built from *marginal*
rates is empirically false. This script measures the dependence directly and
checks the three dependence-free statements the proposition is restated as.

For each class c_j, let the failure events be E_k = {node k fires} for k<j and
E_j = {node j does not fire}. From the full node-fire matrix (every node run on
every test document of class c_j):

  union bound          R_j >= 1 - sum_i P(E_i)                      (any dependence)
  second-order bound   R_j <= 1 - sum_i P(E_i) + sum_{a<b} P(E_a & E_b)  (Bonferroni)
  chain-rule identity  R_j  = prod_{k<j} (1 - beta~_kj) * (1 - alpha~_j)
                       with beta~_kj = P(node k fires | y=c_j, no earlier node fired)
                       (exact under any dependence)
  marginal product     R_j ~= prod_{k<j} (1 - beta_kj) * (1 - alpha_j)
                       (exact only under conditional independence; its error
                       is reported, not assumed away)

and the reviewer's requested dependence measure: for adjacent nodes, the
covariance and phi coefficient of the fire indicators given y=c_j, plus the
total pairwise co-firing excess sum_{a<b} [P(E_a & E_b) - P(E_a)P(E_b)].

R_j is taken two ways: from the fire matrix under the first-fire rule, and
from the fitted cascade's own predict(); their agreement is reported so the
identity check is not merely self-consistent.

    python scripts/verify_cascade_dependence.py --probe clinc150
    python scripts/verify_cascade_dependence.py

Writes results/cascade_dependence.jsonl (per-class rows + one summary per config).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from _shared import DEFAULT_ENCODER, N_ESTIMATORS, NN_EPOCHS, NN_MEMBERS, RESULTS_DIR, RF_NJOBS, embed, make_tfidf
from verify_cascade_bound import fire_matrix, split_of

from freqcascade.datasets import load
from freqcascade.decomposition import RFOEDClassifier

CONFIGS = [("clinc150", "nn"), ("clinc150", "rf"), ("20newsgroups", "nn"), ("20newsgroups", "rf"),
           ("wos46985", "nn"), ("wos46985", "rf")]
OUT = RESULTS_DIR / "cascade_dependence.jsonl"


def per_class(F, j, mask):
    """All quantities for the class at peel position j. F: nodes x docs (bool)."""
    n_nodes = F.shape[0]
    Fj = F[:, mask]
    n = Fj.shape[1]
    has_own = j < n_nodes
    fires_before = Fj[:j]                                  # E_k, k<j
    events = [fires_before[k] for k in range(j)]
    if has_own:
        events.append(~Fj[j])                              # E_j
    E = np.asarray(events, dtype=float).reshape(len(events), n)
    p = E.mean(axis=1) if len(events) else np.zeros(0)

    # R_j from the fire matrix under the first-fire rule
    if j:
        none_before = ~fires_before.any(axis=0)
    else:
        none_before = np.ones(n, dtype=bool)
    correct = none_before & (Fj[j] if has_own else np.ones(n, dtype=bool))
    r_firstfire = float(correct.mean())

    # chain-rule identity with survival-conditional rates
    alive = np.ones(n, dtype=bool)
    chain = 1.0
    beta_surv = []
    for k in range(j):
        if alive.any():
            b = float(Fj[k][alive].mean())
            beta_surv.append(b)
            chain *= (1.0 - b)
            alive &= ~Fj[k]
        else:
            beta_surv.append(float("nan"))
            chain *= 1.0
    if has_own:
        a_surv = float((~Fj[j][alive]).mean()) if alive.any() else 0.0
        chain *= (1.0 - a_surv)
    else:
        a_surv = 0.0
    if not alive.any() and j:
        chain = 0.0

    # marginal product (independence form)
    alpha = float(1.0 - Fj[j].mean()) if has_own else 0.0
    betas = fires_before.mean(axis=1) if j else np.zeros(0)
    indep = (1.0 - alpha) * float(np.prod(1.0 - betas))

    # Bonferroni first and second order
    s1 = float(p.sum())
    G = (E @ E.T) / n if len(events) else np.zeros((0, 0))
    pair_joint = float((G.sum() - np.trace(G)) / 2.0) if len(events) > 1 else 0.0
    pair_prod = float((np.outer(p, p).sum() - (p ** 2).sum()) / 2.0) if len(events) > 1 else 0.0
    lower = 1.0 - s1
    upper2 = 1.0 - s1 + pair_joint

    # adjacent-node dependence among the earlier nodes (the reviewer's Cov(h_k, h_k+1 | y=c_j))
    covs, phis = [], []
    for k in range(max(j - 1, 0)):
        a, b = fires_before[k].astype(float), fires_before[k + 1].astype(float)
        pa, pb = a.mean(), b.mean()
        c = float((a * b).mean() - pa * pb)
        covs.append(c)
        den = pa * (1 - pa) * pb * (1 - pb)
        if den > 0:
            phis.append(c / np.sqrt(den))

    return dict(position=j, n_test=int(n), alpha=round(alpha, 5), alpha_surv=round(a_surv, 5),
                beta_sum=round(float(betas.sum()), 5),
                r_firstfire=round(r_firstfire, 6), r_chain=round(chain, 6), r_indep=round(indep, 6),
                lower_union=round(lower, 6), upper_bonf2=round(upper2, 6),
                pair_cofire_joint=round(pair_joint, 6), pair_cofire_excess=round(pair_joint - pair_prod, 6),
                adj_pairs=len(covs), adj_pairs_phi_defined=len(phis),
                adj_cov_mean=round(float(np.mean(covs)), 7) if covs else None,
                adj_phi_mean=round(float(np.mean(phis)), 4) if phis else None,
                adj_phi_values=[round(float(x), 4) for x in phis])


def fit(ds, base, tr, seed=0):
    from freqcascade.base_learners import RFBaseLearner
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    y = np.asarray(ds.target)
    if base == "nn":
        X = embed(ds.texts, DEFAULT_ENCODER, ds.name)
        fac = lambda node: TorchNNEnsembleBaseLearner(  # noqa: E731
            n_members=NN_MEMBERS, rebalance=True, hidden_size=128,
            max_epochs=NN_EPOCHS, random_state=seed * 1000 + node)
    else:
        mask = (ds.split != "test") if ds.split is not None else np.ones(ds.n_samples, bool)
        X = make_tfidf(ds.texts[mask], ds.texts)
        fac = lambda node: RFBaseLearner(  # noqa: E731
            n_estimators=N_ESTIMATORS, rebalance=True, random_state=seed * 1000 + node, n_jobs=RF_NJOBS)
    est = RFOEDClassifier(base_learner_factory=fac, order="frequency", random_state=seed)
    est.fit(X[tr], y[tr])
    return est, X


def done():
    if not OUT.exists():
        return set()
    return {(r["dataset"], r["base"]) for r in
            (json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip())
            if r.get("kind") == "summary"}


def run(configs, probe=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seen = done()
    for name, base in configs:
        ds = load(name)
        if (ds.name, base) in seen:
            continue
        tr, te, how = split_of(ds)
        y = np.asarray(ds.target)
        t0 = time.perf_counter()
        est, X = fit(ds, base, tr)
        F = fire_matrix(est, X[te])
        pred = est.predict(X[te])
        rows = []
        for j, cls in enumerate(est.class_order_):
            mask = y[te] == cls
            if not mask.any():
                continue
            r = per_class(F, j, mask)
            r["r_observed"] = round(float((pred[mask] == cls).mean()), 6)
            r.update(dataset=ds.name, base=base, cls=str(cls), split=how)
            rows.append(r)

        obs = np.array([r["r_observed"] for r in rows])
        ff = np.array([r["r_firstfire"] for r in rows])
        ch = np.array([r["r_chain"] for r in rows])
        ind = np.array([r["r_indep"] for r in rows])
        lo = np.array([r["lower_union"] for r in rows])
        up = np.array([r["upper_bonf2"] for r in rows])
        phis = np.array([p for r in rows for p in r["adj_phi_values"]])
        summ = dict(
            dataset=ds.name, base=base, kind="summary", K=len(rows), split=how,
            fit_s=round(time.perf_counter() - t0, 1),
            identity_max_abs_err_vs_firstfire=float(np.max(np.abs(ch - ff))),
            firstfire_vs_predict_agreement=float(np.mean(np.isclose(ff, obs, atol=1e-9))),
            firstfire_vs_predict_max_abs_diff=float(np.max(np.abs(ff - obs))),
            within_union_and_bonf2=float(np.mean((obs >= lo - 1e-9) & (obs <= up + 1e-9))),
            bonf2_interval_width_mean=float(np.mean(np.clip(up, 0, 1) - np.clip(lo, 0, 1))),
            indep_form_mae=float(np.mean(np.abs(ind - obs))),
            indep_form_max_abs_err=float(np.max(np.abs(ind - obs))),
            pair_cofire_excess_mean=float(np.mean([r["pair_cofire_excess"] for r in rows])),
            adj_pairs_phi_defined=int(len(phis)),
            adj_phi_median=float(np.median(phis)) if len(phis) else None,
            adj_phi_q25=float(np.quantile(phis, 0.25)) if len(phis) else None,
            adj_phi_q75=float(np.quantile(phis, 0.75)) if len(phis) else None,
            adj_phi_share_positive=float(np.mean(phis > 0)) if len(phis) else None,
            adj_phi_share_abs_gt_0_1=float(np.mean(np.abs(phis) > 0.1)) if len(phis) else None,
        )
        with OUT.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.write(json.dumps(summ) + "\n")
        print("  %-18s %s K=%d  identity max|err|=%.2e  predict agree=%.3f  in [union,bonf2]=%.3f  "
              "indep MAE=%.4f  adj phi median=%s (n=%d)"
              % (ds.name, base, summ["K"], summ["identity_max_abs_err_vs_firstfire"],
                 summ["firstfire_vs_predict_agreement"], summ["within_union_and_bonf2"],
                 summ["indep_form_mae"], summ["adj_phi_median"], summ["adj_pairs_phi_defined"]), flush=True)
        if probe:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--base", default="nn,rf")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    bases = a.base.split(",")
    configs = [(d, b) for d, b in CONFIGS if b in bases and (not a.datasets or d in a.datasets)]
    run(configs, probe=a.probe)


if __name__ == "__main__":
    main()

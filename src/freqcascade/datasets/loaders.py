"""Dataset loaders. Each returns a :class:`~freqcascade.datasets._base.TextDataset`
(raw text + targets); the benchmark scripts featurize and cross-validate.

Sources, exactly:

single-label
  clinc150            oos-eval ``data_imbalanced.json`` (Larson et al. 2019) --
                      150 intents, the paper's exact native train/val/test split.
  twenty_newsgroups   sklearn ``fetch_20newsgroups`` (headers/footers/quotes
                      removed), then geometric imbalance injected to a target IR.
  wos46985            HF ``bakirgrbic/web-of-science`` (Kowsari HDLTeX release,
                      X/Y/YL1 txt) -- 134 sub-field classes under 7 parent domains.
  drug_reviews        HF ``lewtun/drug-reviews`` (UCI Drugs.com). Classify the
                      ``condition`` field from the review text; junk conditions
                      dropped, rare conditions filtered by ``min_count``.
  ohsumed_23          Moschitti's ``ohsumed-first-20000-docs`` distribution --
                      23 MeSH C14 (cardiovascular disease) categories, one per
                      document. A fine-grained medical single-label benchmark.

multi-label
  reuters21578        NLTK ``reuters`` corpus (ModApte), train+test pooled, the
                      90 topic labels with support in both.
  hoc                 ``sb895/Hallmarks-of-Cancer`` GitHub -- 10 cancer-hallmark
                      labels over ~1.6k PubMed abstracts (document-level rollup
                      of the sentence-level annotations).
  litcovid            HF ``KushT/LitCovid_BioCreative`` -- 7 COVID-19 topic
                      labels, genuinely multi-label, strong per-label skew.

Heavy deps (``datasets``, ``nltk``, ``huggingface_hub``) are imported inside
each loader -- ``import freqcascade.datasets`` needs none of them.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np

from ._base import (
    DatasetUnavailable,
    TextDataset,
    cache_dir,
    clean_text,
    inject_geometric_imbalance,
    indicator_matrix,
)

# --- junk values seen in the Drugs.com ``condition`` field ---------------------
_DRUG_CONDITION_JUNK = ("users found this comment helpful", "</span>")

_UA = {"User-Agent": "Mozilla/5.0 (freqcascade dataset loader)"}


def _download(url: str, dest_name: str):
    """Fetch ``url`` to the cache dir once, return the local Path."""
    path = cache_dir() / dest_name
    if not path.exists():
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as fh:  # noqa: S310
                fh.write(r.read())
        except Exception as e:
            raise DatasetUnavailable(
                f"Could not download {url}: {e}. Save it to {path} manually and re-run."
            ) from e
    return path


def _hf_load(path: str, name: str | None = None, **kw):
    """``datasets.load_dataset`` (parquet/data-file datasets only -- script
    datasets are no longer supported by ``datasets`` >= 4)."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - trivial
        raise DatasetUnavailable(
            "pip install 'freqcascade[datasets]' for this loader"
        ) from e
    try:
        return load_dataset(path, name, **kw)
    except Exception as e:
        raise DatasetUnavailable(
            f"Could not load '{path}'" + (f" ({name})" if name else "") + f": {e}"
        ) from e


# --------------------------------------------------------------------------- #
# Single-label
# --------------------------------------------------------------------------- #

_CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_imbalanced.json"


def load_clinc150() -> TextDataset:
    """CLINC150, the paper's exact source: the ``data_imbalanced.json`` release
    from Larson et al. (2019). Uses the ``train``/``val``/``test`` keys (150
    in-scope intents); the separate ``oos_*`` keys are ignored."""
    path = _download(_CLINC_URL, "clinc_data_imbalanced.json")
    raw = json.loads(open(path, encoding="utf-8").read())
    texts, labels, split = [], [], []
    for split_name, key in (("train", "train"), ("val", "val"), ("test", "test")):
        for text, intent in raw[key]:
            texts.append(clean_text(text))
            labels.append(intent)
            split.append(split_name)
    return TextDataset(
        name="clinc150",
        texts=np.array(texts, dtype=object),
        target=np.array(labels, dtype=object),
        label_names=sorted(set(labels)),
        is_multilabel=False,
        split=np.array(split, dtype=object),
        notes="oos-eval data_imbalanced.json (train/val/test = 10525/3000/4500)",
    )


def load_twenty_newsgroups(imbalance_ratio: float = 50.5, seed: int = 0) -> TextDataset:
    """20 Newsgroups with a geometric imbalance injected to ``imbalance_ratio``
    (default 50.5, the paper's IR). The original run's exact per-class counts
    weren't checked in, so this reproduces the target IR and the geometric
    construction, not necessarily the first submission's row count."""
    from sklearn.datasets import fetch_20newsgroups

    bunch = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes"), random_state=seed
    )
    texts = np.array([clean_text(t) for t in bunch.data], dtype=object)
    labels = np.array(bunch.target)
    if imbalance_ratio and imbalance_ratio > 1:
        idx = inject_geometric_imbalance(labels, imbalance_ratio, seed)
        texts, labels = texts[idx], labels[idx]
    names = list(bunch.target_names)
    return TextDataset(
        name=f"20newsgroups_ir{int(imbalance_ratio)}",
        texts=texts,
        target=np.array([names[i] for i in labels], dtype=object),
        label_names=names,
        is_multilabel=False,
        notes=f"sklearn fetch_20newsgroups, geometric imbalance IR={imbalance_ratio}, seed={seed}",
    )


def load_wos46985() -> TextDataset:
    """Web of Science (Kowsari HDLTeX), 46,985 abstracts across 134 sub-field
    classes nested under 7 parent domains. Sourced from the ``bakirgrbic/
    web-of-science`` mirror of the original X/Y/YL1 text files."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise DatasetUnavailable("pip install 'freqcascade[datasets]' for load_wos46985") from e
    try:
        x_path = hf_hub_download("bakirgrbic/web-of-science", "X.txt", repo_type="dataset")
        y_path = hf_hub_download("bakirgrbic/web-of-science", "Y.txt", repo_type="dataset")
    except Exception as e:
        raise DatasetUnavailable(f"Could not fetch bakirgrbic/web-of-science: {e}") from e

    texts = np.array(
        [clean_text(t) for t in open(x_path, encoding="utf-8", errors="replace").read().splitlines()],
        dtype=object,
    )
    y = [f"C{int(v):03d}" for v in open(y_path, encoding="utf-8").read().split()]
    return TextDataset(
        name="wos46985",
        texts=texts,
        target=np.array(y, dtype=object),
        label_names=sorted(set(y)),
        is_multilabel=False,
        notes="bakirgrbic/web-of-science (HDLTeX WOS46985), 134 classes",
    )


def load_drug_reviews(min_count: int = 20, subsample: int | None = 50_000, seed: int = 0) -> TextDataset:
    """UCI Drugs.com reviews; classify ``condition`` from review text. A
    high-cardinality, extreme-long-tail medical single-label benchmark
    (~350 conditions, IR in the thousands).

    ``min_count`` drops conditions with fewer than that many reviews (below it
    the fold splits stop being meaningful); set it to 1 to keep the full tail.
    ``subsample`` caps the total row count (stratified, seeded) -- default
    50k for a tractable compute budget; pass ``None`` for the full ~150k.
    """
    ds = _hf_load("lewtun/drug-reviews")
    texts, conds = [], []
    for key in ds:
        part = ds[key]
        for review, condition in zip(part["review"], part["condition"]):
            if not condition or not isinstance(condition, str):
                continue
            low = condition.lower()
            if any(j in low for j in _DRUG_CONDITION_JUNK):
                continue
            texts.append(clean_text(review))
            conds.append(condition.strip())
    texts = np.array(texts, dtype=object)
    conds = np.array(conds, dtype=object)

    classes, counts = np.unique(conds, return_counts=True)
    keep_classes = set(classes[counts >= min_count])
    mask = np.array([c in keep_classes for c in conds])
    texts, conds = texts[mask], conds[mask]

    if subsample and subsample < len(texts):
        rng = np.random.default_rng(seed)
        # stratified: proportional draw per remaining class, >= 1 each
        idx_parts = []
        for cls in np.unique(conds):
            ci = np.flatnonzero(conds == cls)
            k = max(1, round(len(ci) * subsample / len(texts)))
            idx_parts.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
        sel = np.sort(np.concatenate(idx_parts))
        texts, conds = texts[sel], conds[sel]

    return TextDataset(
        name="drug_reviews",
        texts=texts,
        target=conds,
        label_names=sorted(set(conds.tolist())),
        is_multilabel=False,
        notes=f"HF lewtun/drug-reviews, min_count={min_count}, subsample={subsample}",
    )


def load_ohsumed_23() -> TextDataset:
    """OHSUMED, Moschitti's ``ohsumed-first-20000-docs`` distribution: 23 MeSH
    C14 (cardiovascular disease) categories, one category per document,
    ``training/`` and ``test/`` pooled here for repeated CV."""
    import io
    import tarfile
    import urllib.request

    from ._base import cache_dir

    url = "http://disi.unitn.it/moschitti/corpora/ohsumed-first-20000-docs.tar.gz"
    local = cache_dir() / "ohsumed-first-20000-docs.tar.gz"
    if not local.exists():
        try:
            urllib.request.urlretrieve(url, local)  # noqa: S310 - documented academic source
        except Exception as e:
            raise DatasetUnavailable(
                f"Could not download OHSUMED from {url}: {e}. "
                f"Download it manually to {local} and re-run."
            ) from e

    texts, labels = [], []
    with tarfile.open(local, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 3 or parts[-3] not in ("training", "test"):
                continue
            category = parts[-2]
            fh = tar.extractfile(member)
            if fh is None:
                continue
            texts.append(clean_text(io.TextIOWrapper(fh, encoding="latin-1").read()))
            labels.append(category)
    if not texts:
        raise DatasetUnavailable("OHSUMED archive parsed to zero documents; archive layout may have changed.")
    return TextDataset(
        name="ohsumed_23",
        texts=np.array(texts, dtype=object),
        target=np.array(labels, dtype=object),
        label_names=sorted(set(labels)),
        is_multilabel=False,
        notes="Moschitti ohsumed-first-20000-docs, training+test pooled",
    )


# --------------------------------------------------------------------------- #
# Multi-label
# --------------------------------------------------------------------------- #

def load_reuters21578() -> TextDataset:
    """NLTK ``reuters`` (ModApte). Train+test pooled; labels are the 90 topics
    that have at least one training and one test document (the standard R(90)
    set). Documents with no topic are dropped."""
    try:
        import nltk
        from nltk.corpus import reuters
    except ImportError as e:  # pragma: no cover
        raise DatasetUnavailable("pip install nltk to use load_reuters21578") from e
    try:
        reuters.fileids()
    except LookupError:
        nltk.download("reuters", quiet=True)

    train_cats, test_cats = set(), set()
    for fid in reuters.fileids():
        (train_cats if fid.startswith("training/") else test_cats).update(reuters.categories(fid))
    labels_90 = sorted(train_cats & test_cats)

    texts, label_lists = [], []
    for fid in reuters.fileids():
        cats = [c for c in reuters.categories(fid) if c in labels_90]
        if not cats:
            continue
        texts.append(clean_text(reuters.raw(fid)))
        label_lists.append(cats)
    return TextDataset(
        name="reuters21578",
        texts=np.array(texts, dtype=object),
        target=indicator_matrix(label_lists, labels_90),
        label_names=labels_90,
        is_multilabel=True,
        notes="NLTK reuters ModApte, R(90), train+test pooled",
    )


_HOC_HALLMARKS = [
    "Sustaining proliferative signaling",
    "Evading growth suppressors",
    "Resisting cell death",
    "Enabling replicative immortality",
    "Inducing angiogenesis",
    "Activating invasion and metastasis",
    "Genomic instability and mutation",
    "Tumor promoting inflammation",
    "Cellular energetics",
    "Avoiding immune destruction",
]
# spelling / phrasing variants seen in the annotation files -> canonical name
_HOC_ALIASES = {
    "genome instability and mutation": "Genomic instability and mutation",
    "tumor-promoting inflammation": "Tumor promoting inflammation",
    "deregulating cellular energetics": "Cellular energetics",
    "avoiding immune destruction": "Avoiding immune destruction",
}


def load_hoc() -> TextDataset:
    """Hallmarks of Cancer: 10 cancer-hallmark labels over ~1.85k PubMed
    abstracts. The source (``sb895/Hallmarks-of-Cancer``) annotates individual
    sentences; a document gets a hallmark if any of its sentences carries it."""
    import io
    import zipfile

    zip_path = _download(
        "https://github.com/sb895/Hallmarks-of-Cancer/archive/refs/heads/master.zip",
        "hallmarks_of_cancer.zip",
    )
    texts, label_lists = [], []
    with zipfile.ZipFile(zip_path) as z:
        text_files = sorted(n for n in z.namelist() if "/text/" in n and n.endswith(".txt"))
        for tf in text_files:
            pmid = tf.rsplit("/", 1)[-1]
            lf = tf.replace("/text/", "/labels/")
            try:
                raw_label = z.read(lf).decode("utf-8", "replace").lower()
            except KeyError:
                continue
            labs = {c for c in _HOC_HALLMARKS if c.lower() in raw_label}
            labs |= {v for k, v in _HOC_ALIASES.items() if k in raw_label}
            if not labs:
                continue  # ~270 abstracts carry no hallmark annotation
            texts.append(clean_text(io.TextIOWrapper(io.BytesIO(z.read(tf)), encoding="utf-8", errors="replace").read()))
            label_lists.append(sorted(labs))
    return TextDataset(
        name="hoc",
        texts=np.array(texts, dtype=object),
        target=indicator_matrix(label_lists, _HOC_HALLMARKS),
        label_names=_HOC_HALLMARKS,
        is_multilabel=True,
        notes="sb895/Hallmarks-of-Cancer, sentence labels rolled up to document level, unlabeled abstracts dropped",
    )


# BioCreative VII LitCovid track label order (the 7-vector in the `label` column)
_LITCOVID_TOPICS = [
    "Treatment", "Diagnosis", "Prevention", "Mechanism",
    "Transmission", "Epidemic Forecasting", "Case Report",
]


def load_litcovid() -> TextDataset:
    """LitCovid (BioCreative VII track 5): 7 COVID-19 topic labels over ~33k
    PubMed abstracts, multi-label, strong per-label skew."""
    import ast

    ds = _hf_load("KushT/LitCovid_BioCreative")
    texts, Y = [], []
    for key in ds:
        part = ds[key]
        for title, abstract, label in zip(part["title"], part["abstract"], part["label"]):
            vec = ast.literal_eval(label) if isinstance(label, str) else list(label)
            if len(vec) != len(_LITCOVID_TOPICS):
                continue
            texts.append(clean_text(f"{title}. {abstract}"))
            Y.append(vec)
    return TextDataset(
        name="litcovid",
        texts=np.array(texts, dtype=object),
        target=np.asarray(Y, dtype=np.int8),
        label_names=_LITCOVID_TOPICS,
        is_multilabel=True,
        notes="KushT/LitCovid_BioCreative, train+val+test pooled",
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

SINGLE_LABEL = {
    "clinc150": load_clinc150,
    "20newsgroups": load_twenty_newsgroups,
    "wos46985": load_wos46985,
    "drug_reviews": load_drug_reviews,
    "ohsumed_23": load_ohsumed_23,
}

MULTI_LABEL = {
    "reuters21578": load_reuters21578,
    "hoc": load_hoc,
    "litcovid": load_litcovid,
}

REGISTRY = {**SINGLE_LABEL, **MULTI_LABEL}


def load(name: str, **kwargs) -> TextDataset:
    """Load a dataset by registry name (see ``REGISTRY``)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)

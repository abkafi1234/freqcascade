"""Dataset loaders. Each returns a :class:`~freqcascade.datasets._base.TextDataset`
(raw text + targets); the benchmark scripts featurize and cross-validate.

Sources, exactly:

single-label
  clinc150            HF ``clinc_oos`` config ``imbalanced``; the out-of-scope
                      class dropped -> 150 intents, native train/val/test split.
  twenty_newsgroups   sklearn ``fetch_20newsgroups`` (headers/footers/quotes
                      removed), then geometric imbalance injected to a target IR.
  wos46985            HF ``web_of_science`` config ``WOS46985`` -- 134 sub-field
                      classes under 7 parent domains.
  drug_reviews        HF ``lewtun/drug-reviews`` (UCI Drugs.com). Classify the
                      ``condition`` field from the review text; junk conditions
                      dropped, rare conditions filtered by ``min_count``.
  ohsumed_23          Moschitti's ``ohsumed-first-20000-docs`` distribution --
                      23 MeSH C14 (cardiovascular disease) categories, one per
                      document. A fine-grained medical single-label benchmark.

multi-label
  reuters21578        NLTK ``reuters`` corpus (ModApte), train+test pooled, the
                      90 topic labels with support in both.
  hoc                 HF Hallmarks of Cancer -- 10 cancer-hallmark labels over
                      ~1.6k PubMed abstracts (document-level aggregation).
  litcovid            HF LitCovid (BioCreative VII) -- 7 COVID-19 topic labels,
                      genuinely multi-label, strong per-label skew.

Heavy deps (``datasets``, ``nltk``) are imported inside each loader, matching
the rest of the package -- ``import freqcascade.datasets`` needs neither.
"""

from __future__ import annotations

import numpy as np

from ._base import (
    DatasetUnavailable,
    TextDataset,
    clean_text,
    inject_geometric_imbalance,
    indicator_matrix,
)

# --- junk values seen in the Drugs.com ``condition`` field ---------------------
_DRUG_CONDITION_JUNK = ("users found this comment helpful", "</span>")


def _hf_load(path: str, name: str | None = None, **kw):
    """``datasets.load_dataset`` with a uniform, actionable failure."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - trivial
        raise DatasetUnavailable(
            "The 'datasets' package is required for this loader: pip install 'freqcascade[datasets]'"
        ) from e
    try:
        return load_dataset(path, name, **kw)
    except Exception as e:
        raise DatasetUnavailable(
            f"Could not load '{path}'"
            + (f" ({name})" if name else "")
            + f" from the Hugging Face Hub: {e}. "
            "Check network access, or set FREQCASCADE_DATA to a cached copy."
        ) from e


# --------------------------------------------------------------------------- #
# Single-label
# --------------------------------------------------------------------------- #

def load_clinc150(drop_oos: bool = True) -> TextDataset:
    ds = _hf_load("clinc_oos", "imbalanced")
    parts, texts, labels, split = [], [], [], []
    names = ds["train"].features["intent"].names
    for split_name, hf_key in (("train", "train"), ("val", "validation"), ("test", "test")):
        part = ds[hf_key]
        for text, intent in zip(part["text"], part["intent"]):
            label = names[intent]
            if drop_oos and label == "oos":
                continue
            texts.append(clean_text(text))
            labels.append(label)
            split.append(split_name)
    kept = sorted(set(labels))
    return TextDataset(
        name="clinc150",
        texts=np.array(texts, dtype=object),
        target=np.array(labels, dtype=object),
        label_names=kept,
        is_multilabel=False,
        split=np.array(split, dtype=object),
        notes="HF clinc_oos/imbalanced, oos dropped" if drop_oos else "HF clinc_oos/imbalanced",
    )


def load_twenty_newsgroups(imbalance_ratio: float = 50.0, seed: int = 0) -> TextDataset:
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
    ds = _hf_load("web_of_science", "WOS46985")
    part = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    cols = part.column_names
    text_col = next((c for c in ("input_data", "text", "abstract", "Abstract") if c in cols), None)
    label_col = next((c for c in ("label", "Y", "labels") if c in cols), None)
    if text_col is None or label_col is None:
        raise DatasetUnavailable(f"web_of_science schema unexpected: columns were {cols}")
    texts = np.array([clean_text(t) for t in part[text_col]], dtype=object)
    raw = part[label_col]
    feat = part.features[label_col]
    names_attr = getattr(feat, "names", None)
    if names_attr:
        target = np.array([names_attr[i] for i in raw], dtype=object)
        label_names = list(names_attr)
    else:
        target = np.array([str(v) for v in raw], dtype=object)
        label_names = sorted(set(target))
    return TextDataset(
        name="wos46985",
        texts=texts,
        target=target,
        label_names=label_names,
        is_multilabel=False,
        notes="HF web_of_science/WOS46985",
    )


def load_drug_reviews(min_count: int = 20, subsample: int | None = None, seed: int = 0) -> TextDataset:
    """UCI Drugs.com reviews; classify ``condition`` from review text.

    ``min_count`` drops conditions with fewer than that many reviews (below it
    the fold splits stop being meaningful); set it to 1 to keep the full tail.
    ``subsample`` optionally caps the total row count (stratified) for compute.
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


def load_hoc() -> TextDataset:
    """Hallmarks of Cancer: 10 cancer-hallmark labels over ~1.6k PubMed
    abstracts. Sentence-level labels in the source are aggregated to one
    binary vector per abstract."""
    last_err = None
    for path, name in (("qanastek/HoC", None), ("bigbio/hallmarks_of_cancer", "hallmarks_of_cancer_bigbio_text")):
        try:
            ds = _hf_load(path, name)
        except DatasetUnavailable as e:
            last_err = e
            continue
        rows: dict[str, list[str]] = {}
        for key in ds:
            part = ds[key]
            cols = part.column_names
            id_col = next((c for c in ("document_id", "doc_id", "id", "pmid") if c in cols), None)
            text_col = next((c for c in ("text", "abstract", "sentence") if c in cols), None)
            label_col = next((c for c in ("labels", "label", "hallmarks") if c in cols), None)
            if not (text_col and label_col):
                continue
            ids = part[id_col] if id_col else range(len(part[text_col]))
            for doc_id, text, labs in zip(ids, part[text_col], part[label_col]):
                key_id = str(doc_id)
                rec = rows.setdefault(key_id, [text, []])
                if isinstance(labs, str):
                    labs = [labs]
                rec[1].extend(l for l in (labs or []) if l)
        if rows:
            texts = np.array([clean_text(v[0]) for v in rows.values()], dtype=object)
            label_lists = [sorted(set(v[1])) for v in rows.values()]
            names = sorted({l for ll in label_lists for l in ll})
            return TextDataset(
                name="hoc",
                texts=texts,
                target=indicator_matrix(label_lists, names),
                label_names=names,
                is_multilabel=True,
                notes=f"HF {path}, document-level aggregation",
            )
    raise DatasetUnavailable(f"No usable Hallmarks of Cancer source found. Last error: {last_err}")


def load_litcovid() -> TextDataset:
    """LitCovid (BioCreative VII): 7 COVID-19 topic labels, multi-label, with
    strong per-label frequency skew -- a second multi-label medical benchmark."""
    last_err = None
    for path in ("bio-datasets/litcovid", "Yijia-Xiao/LitCovid", "mwong/litcovid"):
        try:
            ds = _hf_load(path)
        except DatasetUnavailable as e:
            last_err = e
            continue
        texts, label_lists = [], []
        for key in ds:
            part = ds[key]
            cols = part.column_names
            text_col = next((c for c in ("abstract", "text", "title_abstract") if c in cols), None)
            label_col = next((c for c in ("label", "labels", "topics", "category") if c in cols), None)
            if not (text_col and label_col):
                continue
            for text, labs in zip(part[text_col], part[label_col]):
                if isinstance(labs, str):
                    labs = [x.strip() for x in labs.replace(";", ",").split(",") if x.strip()]
                texts.append(clean_text(text))
                label_lists.append(sorted(set(labs or [])))
        if texts:
            names = sorted({l for ll in label_lists for l in ll})
            return TextDataset(
                name="litcovid",
                texts=np.array(texts, dtype=object),
                target=indicator_matrix(label_lists, names),
                label_names=names,
                is_multilabel=True,
                notes=f"HF {path}",
            )
    raise DatasetUnavailable(f"No usable LitCovid source found. Last error: {last_err}")


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

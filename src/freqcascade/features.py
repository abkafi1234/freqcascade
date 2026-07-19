"""Featurizers shared identically across every method being compared, so
that differences in results trace to the modeling choice, not the input
representation (§4 of EXPERIMENTAL_DESIGN.md)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class Featurizer(Protocol):
    def fit(self, texts: list[str]) -> "Featurizer": ...
    def transform(self, texts: list[str]) -> np.ndarray: ...

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


class TfidfFeaturizer:
    """Fast, dependency-light featurizer for correctness smoke tests and
    as the RF base learner's feature space (RF traditionally pairs with
    sparse bag-of-words/TF-IDF rather than dense embeddings)."""

    def __init__(self, max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2)):
        self._vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, sublinear_tf=True
        )

    def fit(self, texts: list[str]) -> "TfidfFeaturizer":
        self._vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self._vectorizer.fit_transform(texts)


class TfidfSvdFeaturizer:
    """Dense, low-dimensional stand-in for the real §3a sentence-embedding
    featurizer, used only for fast local smoke testing. NN classifiers
    (MLPClassifier, and neural nets generally) densify sparse input
    internally on every fit — feeding them a raw 20k-dim sparse TF-IDF
    matrix silently blows up memory (this is what caused the first
    smoke-test run to swap the machine to a halt). Projecting to a small
    dense space up front avoids that entirely and is a closer proxy for
    what real embeddings look like (dense, few hundred dims) than raw
    TF-IDF ever was."""

    def __init__(self, max_features: int = 20000, n_components: int = 100):
        self._tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
        self._svd = TruncatedSVD(n_components=n_components, random_state=0)

    def fit(self, texts: list[str]) -> "TfidfSvdFeaturizer":
        sparse = self._tfidf.fit_transform(texts)
        self._svd.fit(sparse)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        sparse = self._tfidf.transform(texts)
        return self._svd.transform(sparse)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        sparse = self._tfidf.fit_transform(texts)
        return self._svd.fit_transform(sparse)


class SentenceEmbeddingFeaturizer:
    """§3a candidate-unit-1 featurizer: frozen sentence embeddings
    (default `all-MiniLM-L6-v2`). Embeddings are computed once and cached
    by the caller (see scripts/) rather than recomputed per bootstrap
    member, per the compute-budget plan (§4a).

    `all-MiniLM-L6-v2` silently truncates input at 256 subword tokens.
    CLINC150's utterances average 8 words, so this never mattered there
    — but it was found (via a real Tier-1 benchmark regression: RFOED-NN
    scored 0.701 macro-F1 on 20 Newsgroups, *worse* than flat RF's 0.962,
    the opposite of the CLINC150 pattern) that 20 Newsgroups documents
    average 231 words and WOS46985 abstracts average 200 — both already
    past the token budget on average once subword tokenization inflates
    the count, meaning most of each document's content was being
    discarded while TF-IDF-based baselines saw the whole thing. When
    `chunk_words` is set, documents are split into word-chunks of that
    size, each chunk is embedded, and the per-document embedding is the
    mean over its chunks — so no content is silently dropped."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        chunk_words: int | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunk_words = chunk_words
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> "SentenceEmbeddingFeaturizer":
        self._load()
        return self

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        n = self.chunk_words
        return [" ".join(words[i:i + n]) for i in range(0, len(words), n)] or [text]

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.chunk_words:
            return self._transform_chunked(texts)
        model = self._load()
        return np.asarray(
            model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        )

    def _transform_chunked(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        chunk_lists = [self._chunk(t) for t in texts]
        flat_chunks = [c for chunks in chunk_lists for c in chunks]
        flat_emb = np.asarray(
            model.encode(flat_chunks, batch_size=self.batch_size, show_progress_bar=False)
        )
        out = np.empty((len(texts), flat_emb.shape[1]), dtype=flat_emb.dtype)
        pos = 0
        for i, chunks in enumerate(chunk_lists):
            n = len(chunks)
            out[i] = flat_emb[pos:pos + n].mean(axis=0)
            pos += n
        return out

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

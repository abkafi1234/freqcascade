"""§3a candidate unit 2: small TextCNN (Kim 2014 style) over *trainable*
embeddings, for the base-learner pilot study (T2.3). Unlike
`TorchNNEnsembleBaseLearner` (unit 1: frozen MiniLM sentence embeddings
+ MLP), this unit builds its own vocabulary and trainable embedding
table per node, so it consumes raw text directly rather than
precomputed features -- `X` here is a 1-D numpy object array of strings
(RFOEDClassifier's boolean/integer indexing works fine on that), not a
TF-IDF/embedding matrix.

Same ensemble-per-node + per-node-rebalancing mechanism as every other
base learner in this project (K members, each on its own
`balanced_bootstrap_indices` draw), but members are trained
sequentially (not batched via `torch.bmm` like `TorchNNEnsembleBaseLearner`)
since each member needs its own vocabulary/embedding table sized to its
own bootstrap sample, which don't share a common tensor shape the way
MLP-over-fixed-dim-embeddings members do -- a deliberate simplicity
tradeoff appropriate for a pilot study (small K, small text volumes),
not the full-scale production path.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .rebalance import balanced_bootstrap_indices

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _TextCNNModule:
    """Thin wrapper around a torch nn.Module + its own vocab, built
    lazily so `torch` is only imported when this unit is actually used."""

    def __init__(self, vocab_size, embed_dim, n_filters, kernel_sizes, max_len, device):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.convs = nn.ModuleList([
                    nn.Conv1d(embed_dim, n_filters, k, padding=k // 2) for k in kernel_sizes
                ])
                self.dropout = nn.Dropout(0.3)
                self.fc = nn.Linear(n_filters * len(kernel_sizes), 2)

            def forward(self, x):
                e = self.embed(x).transpose(1, 2)  # (B, embed_dim, L)
                pooled = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]
                h = self.dropout(torch.cat(pooled, dim=1))
                return self.fc(h)

        self.net = _Net().to(device)
        self.max_len = max_len
        self.device = device


class TextCNNEnsembleBaseLearner:
    def __init__(
        self,
        n_members: int = 10,
        rebalance: bool = True,
        embed_dim: int = 64,
        n_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        max_vocab: int = 5000,
        max_len: int = 64,
        max_epochs: int = 15,
        lr: float = 1e-3,
        device: str | None = None,
        random_state: int | None = None,
    ):
        self.n_members = n_members
        self.rebalance = rebalance
        self.embed_dim = embed_dim
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.max_epochs = max_epochs
        self.lr = lr
        self.random_state = random_state
        self._device_arg = device
        self._members = []  # list of (module, vocab) or a stub

    def _resolve_device(self):
        import torch

        if self._device_arg is not None:
            return torch.device(self._device_arg)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_vocab(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(_tokenize(t))
        vocab = {"<pad>": 0, "<unk>": 1}
        for word, _ in counts.most_common(self.max_vocab - 2):
            vocab[word] = len(vocab)
        return vocab

    def _encode(self, texts, vocab):
        import torch

        out = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [vocab.get(tok, 1) for tok in _tokenize(t)[: self.max_len]]
            out[i, : len(ids)] = ids
        return torch.from_numpy(out)

    def fit(self, X, y: np.ndarray) -> "TextCNNEnsembleBaseLearner":
        import torch
        import torch.nn.functional as F

        device = self._resolve_device()
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y)
        X = np.asarray(X, dtype=object)
        n = len(y)

        self._members = []
        for k in range(self.n_members):
            idx = balanced_bootstrap_indices(y, rng, max_per_class=1000) if self.rebalance else rng.integers(0, n, size=n)
            X_k, y_k = X[idx], y[idx]
            if len(np.unique(y_k)) < 2:
                self._members.append(("stub", int(y_k[0])))
                continue

            vocab = self._build_vocab(X_k)
            module = _TextCNNModule(len(vocab), self.embed_dim, self.n_filters, self.kernel_sizes, self.max_len, device)
            X_enc = self._encode(X_k, vocab).to(device)
            y_t = torch.from_numpy(y_k.astype(np.int64)).to(device)

            opt = torch.optim.Adam(module.net.parameters(), lr=self.lr)
            module.net.train()
            batch_size = 64
            for epoch in range(self.max_epochs):
                perm = torch.randperm(len(y_k))
                for b in range(0, len(y_k), batch_size):
                    b_idx = perm[b:b + batch_size]
                    opt.zero_grad()
                    logits = module.net(X_enc[b_idx])
                    loss = F.cross_entropy(logits, y_t[b_idx])
                    loss.backward()
                    opt.step()
            module.net.eval()
            self._members.append(("model", module, vocab))
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        assert self._members, "call fit() first"
        X = np.asarray(X, dtype=object)
        n = len(X)
        probas = []
        for member in self._members:
            if member[0] == "stub":
                out = np.zeros((n, 2))
                out[:, member[1]] = 1.0
                probas.append(out)
                continue
            _, module, vocab = member
            X_enc = self._encode(X, vocab).to(module.device)
            with torch.no_grad():
                logits = module.net(X_enc)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()
            probas.append(proba)
        return np.mean(probas, axis=0)

"""GPU-accelerated RFOED-NN base learner. Same mechanism as
base_learners.NNEnsembleBaseLearner (K members, each on its own
bootstrap draw, rebalance flag controls stratified vs. plain bootstrap)
but all K members are trained as ONE batched computation via
`torch.bmm` over a leading K dimension, rather than K sequential fits —
these per-node MLPs are small enough that sequential per-member GPU
calls would be dominated by kernel-launch/transfer overhead and could
easily be *slower* than CPU; batching across K is what actually makes
spending a GPU on this worthwhile. Import is lazy (torch is optional)
so the rest of the package works without it installed.
"""

from __future__ import annotations

import numpy as np

from .rebalance import balanced_bootstrap_indices


class TorchNNEnsembleBaseLearner:
    def __init__(
        self,
        n_members: int = 25,
        rebalance: bool = True,
        hidden_size: int = 64,
        max_epochs: int = 150,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        device: str | None = None,
        random_state: int | None = None,
        max_bootstrap_per_class: int = 2000,
    ):
        self.n_members = n_members
        self.rebalance = rebalance
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self._device_arg = device
        self._params = None
        self._device_used = None
        # Uncapped, a node with a huge "rest" class (WOS46985's shallow
        # nodes: ~750 vs ~32k) draws ~32k per class; batched across K=50
        # members at 384 dims that's several GB and OOM'd a real 8GB-VRAM
        # run. A small MLP head has no use for tens of thousands of
        # per-member examples anyway, so this cap is both the fix and a
        # speed win (see rebalance.balanced_bootstrap_indices).
        self.max_bootstrap_per_class = max_bootstrap_per_class

    def _resolve_device(self):
        import torch

        if self._device_arg is not None:
            return torch.device(self._device_arg)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y: np.ndarray) -> "TorchNNEnsembleBaseLearner":
        import torch

        device = self._resolve_device()
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y)
        X_np = np.asarray(X, dtype=np.float32)
        n, d = X_np.shape
        K = self.n_members

        # Plain bootstrap is capped to the same budget as the balanced
        # case (2 * max_bootstrap_per_class) so both branches produce a
        # fixed-length draw regardless of dataset size — required for a
        # rectangular (K, boot_len) index array, and the memory-safety
        # reason this cap exists in the first place.
        plain_boot_len = min(n, 2 * self.max_bootstrap_per_class)
        idx_list = []
        for _ in range(K):
            if self.rebalance:
                idx = balanced_bootstrap_indices(y, rng, max_per_class=self.max_bootstrap_per_class)
            else:
                idx = rng.integers(0, n, size=plain_boot_len)
            idx_list.append(idx)
        idx_arr = np.stack(idx_list)

        Xb = X_np[idx_arr]
        yb = y[idx_arr].astype(np.int64)
        Xb_t = torch.from_numpy(Xb).to(device)
        yb_t = torch.from_numpy(yb).to(device)

        torch.manual_seed(0 if self.random_state is None else self.random_state)
        H = self.hidden_size
        W1 = (torch.randn(K, d, H, device=device) / np.sqrt(d)).requires_grad_(True)
        b1 = torch.zeros(K, 1, H, device=device, requires_grad=True)
        W2 = (torch.randn(K, H, 2, device=device) / np.sqrt(H)).requires_grad_(True)
        b2 = torch.zeros(K, 1, 2, device=device, requires_grad=True)

        opt = torch.optim.Adam([W1, b1, W2, b2], lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.max_epochs):
            opt.zero_grad()
            h = torch.relu(torch.bmm(Xb_t, W1) + b1)
            logits = torch.bmm(h, W2) + b2
            # Per-row mean-reduction over the flattened (K*boot_len) batch
            # divides *each member's* gradient by an extra factor of K,
            # since a member's parameters only ever receive gradient from
            # its own boot_len rows — at fixed (epochs, lr) this silently
            # undertrains every member more as K grows. Mean within each
            # member first, then SUM across members, so gradient
            # magnitude per member is independent of K (found by
            # comparing K=50 vs K=10 on real data: K=50 collapsed under
            # both the cascade and the argmax decision rule despite
            # identical per-member capacity, which is what this bug predicts
            # and a cascade-specific brittleness would not).
            per_row_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 2), yb_t.reshape(-1), reduction="none"
            ).reshape(K, -1)
            loss = per_row_loss.mean(dim=1).sum()
            loss.backward()
            opt.step()

        self._params = tuple(p.detach().clone() for p in (W1, b1, W2, b2))
        self._device_used = device

        # Explicitly drop the large bootstrap tensors and optimizer state
        # (which holds Adam momentum buffers the same size as the
        # params) before returning, rather than relying on them going
        # out of scope — decomposition.py fits one of these per cascade
        # node (up to ~150), and on an 8GB-VRAM card the caching
        # allocator's fragmentation across that many sequential fits is
        # cheap insurance against, even though max_bootstrap_per_class is
        # the actual fix for the OOM this was written after (a single
        # node's uncapped balanced bootstrap on WOS46985 needed ~5GB by
        # itself).
        del Xb_t, yb_t, W1, b1, W2, b2, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        assert self._params is not None, "call fit() first"
        W1, b1, W2, b2 = self._params
        device = self._device_used
        X_np = np.asarray(X, dtype=np.float32)
        n = X_np.shape[0]
        K = self.n_members

        X_t = torch.from_numpy(X_np).to(device)
        X_exp = X_t.unsqueeze(0).expand(K, n, X_t.shape[1])
        with torch.no_grad():
            h = torch.relu(torch.bmm(X_exp, W1) + b1)
            logits = torch.bmm(h, W2) + b2
            proba = torch.softmax(logits, dim=-1)
            mean_proba = proba.mean(dim=0)
        return mean_proba.cpu().numpy()

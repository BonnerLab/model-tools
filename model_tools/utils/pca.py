from typing import Optional, Tuple

import numpy as np
import torch


class IncrementalPCAPytorch:
    """
    Based on sklearn's IncrementalPCA class:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
    """

    def __init__(self, n_components: Optional[int] = None,
                 whiten: bool = False, device: Optional[str] = None):
        self.n_components = n_components
        self.whiten = whiten
        self.device = device

        self._initialized = False
        self._batch_size = None
        self._n_feats = None

        self.mean = None
        self.var = None
        self.n_samples_seen = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.singular_values = None
        self.noise_variance = None


    def fit_partial(self, X: torch.Tensor) -> None:
        X = X.to(self.device)
        if not self._initialized:
            self._initialize_from(X)
        n_feats = X.size(1)

        new_mean, new_var, new_n_samples_seen = self._incremental_mean_and_var(X)

        # Whitening
        if self.n_samples_seen == 0:
            # If this is the first step, simply whitten X
            X -= new_mean
        else:
            batch_mean = X.mean(dim=0)
            X -= batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = np.sqrt(
                (self.n_samples_seen / new_n_samples_seen) * X.size(0)
            ) * (self.mean - batch_mean)
            X = torch.vstack([
                self.singular_values.reshape(-1, 1) * self.components,
                X,
                mean_correction
            ])

        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        U, Vt = svd_flip(U, Vt, u_based_decision=False)
        S_squared = S ** 2
        explained_variance = S_squared / (new_n_samples_seen - 1)
        explained_variance_ratio = S_squared / torch.sum(new_var * new_n_samples_seen)

        self.mean = new_mean
        self.var = new_var
        self.n_samples_seen = new_n_samples_seen

        self.components = Vt[:self.n_components]
        self.singular_values = S[:self.n_components]
        self.explained_variance = explained_variance[:self.n_components]
        self.explained_variance_ratio = explained_variance_ratio[:self.n_components]
        if self.n_components < n_feats:
            self.noise_variance = explained_variance[self.n_components:].mean()
        else:
            self.noise_variance = 0.0


    def _initialize_from(self, X: torch.Tensor):
        assert X.ndim == 2

        batch_size, n_feats = X.shape
        if self.n_components is None:
            self.n_components = min(batch_size, n_feats)
        else:
            assert 1 <= self.n_components <= min(batch_size, n_feats)
        self._batch_size, self._n_feats = n_feats, batch_size

        self.mean = torch.zeros(n_feats).to(self.device)
        self.var = torch.zeros(n_feats).to(self.device)
        self.n_samples_seen = 0

        self.singular_values = torch.zeros(self.n_components).to(self.device)
        self.components = torch.zeros(self.n_components, n_feats).to(self.device)

        self._initialized = True


    def _incremental_mean_and_var(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        X_sum = X.sum(dim=0)
        X_n_samples = X.size(0)
        last_sum = self.mean * self.n_samples_seen

        new_n_samples_seen = self.n_samples_seen + X_n_samples
        new_mean = (last_sum + X_sum) / new_n_samples_seen

        temp = X - X_sum / X_n_samples
        correction = temp.sum(dim=0)
        X_unnormalized_var = (temp ** 2).sum(dim=0)
        X_unnormalized_var -= correction ** 2 / X_n_samples
        last_unnormalized_var = self.var * self.n_samples_seen
        last_over_new_n_samples = max(self.n_samples_seen / X_n_samples, 1e-7)
        new_unnormalized_var = (
            last_unnormalized_var
            + X_unnormalized_var
            + last_over_new_n_samples
            / new_n_samples_seen
            * (last_sum / last_over_new_n_samples - X_sum) ** 2
        )
        new_var = new_unnormalized_var / new_n_samples_seen

        return new_mean, new_var, new_n_samples_seen


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD."""
    if u_based_decision:
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs.unsqueeze(dim=1)
    else:
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs.unsqueeze(dim=1)
    return u, v

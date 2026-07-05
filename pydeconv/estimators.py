"""Custom estimators for PyDeconv.

This module hosts estimator classes compatible with scikit-learn's estimator
API so they can be plugged into ``DeconvolutionModel(estimator=...)``.
Later we may add more estimators like the one that startfrom the precalculated XtX
gram matrix which promise to outperform the current Tridge estimator in terms of speed and memory usage.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

try:
    import torch
except ImportError as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class Tridge(BaseEstimator, RegressorMixin):
    """Torch-based ridge regressor.

    Parameters
    ----------
    alpha : float
        L2 regularization strength.
    use_gpu : bool
        If ``True`` and CUDA is available, run on GPU.
    """

    def __init__(self, alpha: float = 1.0, use_gpu: bool = True):
        if torch is None:
            raise ImportError(
                "Tridge requires PyTorch. Install torch to use this estimator."
            ) from _TORCH_IMPORT_ERROR

        self.alpha = alpha
        self.use_gpu = use_gpu
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.coefs_ = None
        self.coef_ = None
        self.y_predicted_ = None

    def fit(self, X, y):
        """Fit ridge coefficients."""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(self.device)

        x_tx = X.T @ X
        ridge_term = self.alpha * torch.eye(X.shape[1], device=self.device)
        x_tx_reg = x_tx + ridge_term

        x_ty = X.T @ y
        betas = torch.linalg.solve(x_tx_reg, x_ty)

        self.coefs_ = betas
        self.coef_ = betas.detach().cpu().numpy()
        self.y_predicted_ = X @ betas
        return self

    def predict(self, X):
        """Generate predictions for ``X``."""
        if self.coefs_ is None:
            raise RuntimeError("Tridge has not been fitted yet.")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        y_hat = X @ self.coefs_
        return y_hat.detach().cpu().numpy()

    def predict_some_columns(self, X, columns_idxs: list[int]):
        """Predict using only a subset of columns."""
        if self.coefs_ is None:
            raise RuntimeError("Tridge has not been fitted yet.")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        reduced_coefs = self.coefs_[columns_idxs]
        reduced_x = X[:, columns_idxs]
        return (reduced_x @ reduced_coefs).detach().cpu().numpy()

    def get_coefficients(self) -> np.ndarray:
        """Return fitted coefficients as a NumPy array."""
        if self.coefs_ is None:
            raise RuntimeError("Tridge has not been fitted yet.")
        return self.coefs_.detach().cpu().numpy()

    def get_predictions(self) -> np.ndarray:
        """Return fitted predictions as a NumPy array."""
        if self.y_predicted_ is None:
            raise RuntimeError("Tridge has not been fitted yet.")
        return self.y_predicted_.detach().cpu().numpy()

    def score(self, X, y) -> float:
        """Compute R^2 score."""
        y_pred = self.predict(X)
        y_true = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        return r2_score(y_true, y_pred)

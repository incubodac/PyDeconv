import numpy as np
import pytest
from sklearn.base import is_regressor

from pydeconv.estimators import Tridge


try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch is required for Tridge")
def test_tridge_is_sklearn_regressor():
    est = Tridge(alpha=1.0, use_gpu=False)
    assert is_regressor(est)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch is required for Tridge")
def test_tridge_fit_predict_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8))
    y = rng.standard_normal(50)

    est = Tridge(alpha=1.0, use_gpu=False)
    est.fit(X, y)
    y_pred = est.predict(X)

    assert est.coef_.shape == (8,)
    assert y_pred.shape == (50,)

import pytest
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

from model_tools.utils.pca import IncrementalPCAPytorch


@pytest.mark.parametrize('X_shape', [(1000, 20)])
@pytest.mark.parametrize('n_components', [None, 9])
@pytest.mark.parametrize(['batch_size', 'whitten'], [(10, True), (100, False)])
def test_incremental_pca_pytorch(X_shape, n_components, batch_size, whitten):
    np.random.seed(27)
    torch.random.manual_seed(27)

    cov = np.random.rand(X_shape[1], X_shape[1]).astype(np.float32)
    cov = cov @ cov.T
    X = np.random.multivariate_normal(mean=np.random.rand(X_shape[1]), cov=cov, size=X_shape[0])
    X_torch = torch.from_numpy(X)

    pca_sklearn = IncrementalPCA(n_components=n_components, whiten=whitten)
    pca_pytorch = IncrementalPCAPytorch(n_components=n_components, whiten=whitten)

    for i in range(0, X_shape[0], batch_size):
        pca_sklearn.partial_fit(X[i:i + batch_size])
        pca_pytorch.fit_partial(X_torch[i:i + batch_size])

    assert np.allclose(pca_sklearn.components_, pca_pytorch.components.numpy())
    assert np.allclose(pca_sklearn.explained_variance_, pca_pytorch.explained_variance.numpy())
    assert np.allclose(pca_sklearn.explained_variance_ratio_, pca_pytorch.explained_variance_ratio.numpy())
    assert np.allclose(pca_sklearn.singular_values_, pca_pytorch.singular_values.numpy())
    assert np.allclose(pca_sklearn.noise_variance_, pca_pytorch.noise_variance)

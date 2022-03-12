from typing import Optional

import numpy as np
import xarray as xr
import torch
from sklearn.linear_model import LinearRegression

from brainio.assemblies import NeuroidAssembly
from result_caching import store_xarray
from . import ModelActivationsAnalysis


class ModelEigspecAnalysis(ModelActivationsAnalysis):

    def analysis_func(self,
                      assembly: NeuroidAssembly,
                      batch_size: Optional[int] = None,
                      device=None) -> Any:
        return get_eigspec(assembly, batch_size, device)

    @property
    def results(self) -> xr.DataArray:
        layer_results = xr.concat(self._layer_results.values(), dim='identifier')
        layer_results['identifier'] = [self.identifier] * layer_results.sizes['identifier']
        layer_results['layer'] = ('identifier', list(layer_results.keys()))
        for name, data in self.metadata:
            layer_results[name] = ('identifier', [data] * layer_results.sizes['identifier'])
        return layer_results


def get_eigspec(assembly: NeuroidAssembly,
                batch_size: Optional[int] = None,
                device=None) -> xr.DataArray:
    if batch_size is None:
        assembly = torch.from_numpy(assembly.values).to(device)
        S = torch.linalg.svdvals(assembly - assembly.mean(dim=0))
        eigspec = S ** 2 / (assembly.shape[0] - 1)
    else:
        # todo: use an incremental PCA approach
        pass

    eigspec = eigspec.cpu().numpy()
    ed = effective_dimensionalities(eigspec)
    eighty_percent_var = x_percent_var(eigspec, x=0.8)
    alpha = powerlaw_exponent(eigspec)

    eigspec = xr.DataArray(eigspec.reshape(1, -1),
                           dims=['identifier', 'eigval_index'],
                           coords={'effective_dimensionality': ('identifier', [ed]),
                                   'eighty_percent_var': ('identifier', [eighty_percent_var]),
                                   'powerlaw_decay_exponent': ('identifier', [alpha]),
                                   'eigval_index': np.arange(1, len(eigspec) + 1)})
    return eigspec


@store_xarray(identifier_ignore=['assembly', 'batch_size', 'device'])
def get_eigspec_stored(identifier: str,
                       assembly: NeuroidAssembly,
                       batch_size: Optional[int] = None,
                       device=None) -> xr.DataArray:
    eigspec = get_eigspec(assembly=assembly, batch_size=batch_size, device=device)
    eigspec['identifier'] = [identifier]
    return eigspec


############################################################
#################### Helper functions ######################
############################################################


def effective_dimensionalities(eigspec: np.ndarray) -> float:
    return eigspec.sum() ** 2 / (eigspec ** 2).sum()


def x_percent_var(eigspec: np.ndarray, x: float) -> float:
    assert 0 < x < 1
    i_varx = None
    pvar = eigspec.cumsum() / eigspec.sum()
    for i in range(len(pvar)):
        if pvar[i] >= x:
            i_varx = i + 1
            break
    return i_varx


def powerlaw_exponent(eigspec: np.ndarray) -> float:
    start, end = 0, np.log10(len(eigspec))
    eignum = np.logspace(start, end, num=50).round().astype(int)
    eigspec = eigspec[eignum - 1]
    logeignum = np.log10(eignum)
    logeigspec = np.log10(eigspec)
    linear_fit = LinearRegression().fit(logeignum.reshape(-1, 1), logeigspec)
    alpha = -linear_fit.coef_.item()
    return alpha

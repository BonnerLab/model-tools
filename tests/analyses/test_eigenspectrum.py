from pathlib import Path
import os; os.environ['RESULTCACHING_HOME'] = str(Path(__file__).parent / 'result_caching')

import pytest
import numpy as np
import torch

from brainio.assemblies import NeuroidAssembly
from result_caching import get_function_identifier
from model_tools.analyses import ModelActivationsAnalysis
from model_tools.analyses.eigenspectrum import get_eigspec, get_eigspec_stored, ModelEigspecAnalysis
from tests.activations.test___init__ import pytorch_custom, _build_stimulus_set


@pytest.mark.parametrize('assembly_shape', [(1000, 100)])
@pytest.mark.parametrize('batch_size', [None, 200, 50])
def test_get_eigspec(assembly_shape, batch_size):
    np.random.seed(27)
    torch.random.manual_seed(27)

    cov = np.random.rand(assembly_shape[1], assembly_shape[1]).astype(np.float32)
    cov = cov @ cov.T
    assembly = np.random.multivariate_normal(mean=np.random.rand(assembly_shape[1]), cov=cov, size=assembly_shape[0])
    assembly = NeuroidAssembly(assembly, dims=['presentation', 'neuroid'])

    results = get_eigspec(assembly, batch_size)

    # Verify shapes and coordinates
    if batch_size is None:
        batch_size = assembly_shape[1]
    assert results.shape == (1, min(batch_size, assembly_shape[1]))
    assert results.dims == ('identifier', 'eigval_index')
    assert (results.eigval_index.values == np.arange(1, min(batch_size, assembly_shape[1]) + 1)).all()

    # Verify estimated eigenvalues close enough to true ones (i.e. absolute difference < 2% of total variance)
    true_eigspec = np.linalg.eigvalsh(cov)
    true_eigspec = true_eigspec[::-1]
    true_eigspec = true_eigspec[:batch_size]
    assert np.abs(true_eigspec - results.values).sum() / true_eigspec.sum() < 0.02


@pytest.mark.parametrize(['assembly_shape', 'batch_size'], [((1000, 100), 200)])
def test_get_eigspec_stored(assembly_shape, batch_size):
    assembly = NeuroidAssembly(np.random.rand(*assembly_shape), dims=['presentation', 'neuroid'])

    results = get_eigspec_stored('test_assembly', assembly, batch_size)

    # Verify shapes and coordinates
    if batch_size is None:
        batch_size = assembly_shape[1]
    assert results.shape == (1, min(batch_size, assembly_shape[1]))
    assert results.dims == ('identifier', 'eigval_index')
    assert results.identifier.values[0] == 'test_assembly'
    assert (results.eigval_index.values == np.arange(1, min(batch_size, assembly_shape[1]) + 1)).all()

    # Verify saved file exists
    wrapped_func = get_eigspec_stored.__closure__[0].cell_contents
    assert os.path.exists(os.environ['RESULTCACHING_HOME'] + '/' +
                          get_function_identifier(wrapped_func, call_args={'identifier': 'test_assembly'}) + '.pkl')


class TestModelEigspecAnalysis:

    @pytest.mark.parametrize('stimuli_identifier', ['test_stimuli_paths', None, False])
    def test_from_paths(self, stimuli_identifier):
        activations_extractor, layers, stimuli, batch_size = self.get_args()

        eigspec_analysis = ModelEigspecAnalysis(activations_extractor)
        results = eigspec_analysis(stimuli=stimuli,
                                   layers=layers,
                                   stimuli_identifier=stimuli_identifier,
                                   batch_size=batch_size)

        # Verify shapes and coordinates
        self.verify_dims_and_coords(results, activations_extractor, batch_size)

        # Verify saved file paths
        self.verify_saved_paths(eigspec_analysis, activations_extractor, stimuli_identifier, batch_size)

    @pytest.mark.parametrize('stimuli_identifier', ['test_stimulus_set', None, False])
    def test_from_stimulus_set(self, stimuli_identifier):
        activations_extractor, layers, stimuli, batch_size = self.get_args()
        stimuli = _build_stimulus_set(stimuli)
        stimuli.identifier = 'test_stimulus_set_default_identifier'

        eigspec_analysis = ModelEigspecAnalysis(activations_extractor)
        results = eigspec_analysis(stimuli=stimuli,
                                   layers=layers,
                                   stimuli_identifier=stimuli_identifier,
                                   batch_size=batch_size)

        # Verify shapes and coordinates
        self.verify_dims_and_coords(results, activations_extractor, batch_size)

        # Verify saved file paths
        if stimuli_identifier is None:
            stimuli_identifier = stimuli.identifier
        self.verify_saved_paths(eigspec_analysis, activations_extractor, stimuli_identifier, batch_size)

    def get_args(self):
        activations_extractor = pytorch_custom()
        layers = ['relu1', 'relu2']
        stimuli = ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png']
        batch_size = 2
        return activations_extractor, layers, stimuli, batch_size

    def verify_dims_and_coords(self, results, activations_extractor, batch_size):
        assert results.shape == (2, batch_size)
        assert results.dims == ('identifier', 'eigval_index')
        assert (results.identifier.values ==
                np.array([activations_extractor.identifier] *
                         results.sizes['identifier'])
                ).all()

    def verify_saved_paths(self, eigspec_analysis, activations_extractor, stimuli_identifier, batch_size):
        wrapped_func = ModelActivationsAnalysis._run_analysis_stored.__closure__[0].cell_contents
        function_identifier = get_function_identifier(wrapped_func,
                                                      call_args={'self': eigspec_analysis,
                                                                 'identifier': activations_extractor.identifier,
                                                                 'stimuli_identifier': stimuli_identifier,
                                                                 'analysis_kwargs': {'batch_size': batch_size}})
        save_path = os.path.join(os.environ['RESULTCACHING_HOME'], function_identifier + '.pkl')
        if stimuli_identifier:
            assert os.path.exists(save_path)
        else:
            assert not os.path.exists(save_path)

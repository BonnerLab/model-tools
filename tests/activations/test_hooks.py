from pathlib import Path
import os; os.environ['RESULTCACHING_HOME'] = str(Path(__file__).parent / 'result_caching')

import pytest

from result_caching import get_function_identifier
from model_tools.activations.hooks import LayerGlobalMaxPool2d, LayerRandomProjection, LayerPCA
from tests.activations.test___init__ import pytorch_custom, _build_stimulus_set


def test_add_remove_hook():
    extractor, _, _, _ = get_args()

    handle = LayerGlobalMaxPool2d.hook(extractor)
    assert LayerGlobalMaxPool2d.is_hooked(extractor)

    handle.remove()
    assert not LayerGlobalMaxPool2d.is_hooked(extractor)


class TestLayerGlobalMaxPool2d:

    def test_output_shape(self):
        extractor, layers, stimuli, _ = get_args()

        LayerGlobalMaxPool2d.hook(extractor)
        activations = extractor(stimuli, layers=layers)

        assert activations.sel(layer=layers[0]).sizes['neuroid'] == 2      # 2 channels after max-pool
        assert activations.sel(layer=layers[1]).sizes['neuroid'] == 1000   # 1000 features in linear layer

    @pytest.mark.parametrize('hook_identifier', ['test-maxpool', '', None])
    def test_save_path(self, hook_identifier):
        extractor, layers, stimuli, _ = get_args(as_stimulus_set=True)

        LayerGlobalMaxPool2d.hook(extractor, identifier=hook_identifier)
        _ = extractor(stimuli, layers=layers)

        expected_model_identifier = extractor.identifier
        if hook_identifier is None:
            expected_model_identifier += '-maxpool'
        elif hook_identifier != '':
            expected_model_identifier += f'-{hook_identifier}'
        expected_path = extractor._extractor._saver.activations_save_path(identifier=expected_model_identifier,
                                                                          stimuli_identifier=stimuli.identifier)
        assert os.path.exists(expected_path)


class TestLayerRandomProjection:

    @pytest.mark.parametrize('force', [False, True])
    def test_output_shape(self, force):
        extractor, layers, stimuli, _ = get_args()
        n_components = 1001

        LayerRandomProjection.hook(extractor, n_components=n_components, force=force)
        activations = extractor(stimuli, layers=layers)

        assert activations.sel(layer=layers[0]).sizes['neuroid'] == n_components
        if force:
            assert activations.sel(layer=layers[1]).sizes['neuroid'] == n_components   # Random projection
        else:
            assert activations.sel(layer=layers[1]).sizes['neuroid'] == 1000   # Original 1000 features kept

    @pytest.mark.parametrize('hook_identifier', ['test-randproj', '', None])
    def test_save_path(self, hook_identifier):
        extractor, layers, stimuli, _ = get_args(as_stimulus_set=True)
        n_components, force = 3, False

        LayerRandomProjection.hook(extractor, identifier=hook_identifier, n_components=n_components, force=force)
        _ = extractor(stimuli, layers=layers)

        expected_model_identifier = extractor.identifier
        if hook_identifier is None:
            expected_model_identifier += f'-randproj_ncomponents={n_components}_force={force}'
        elif hook_identifier != '':
            expected_model_identifier += f'-{hook_identifier}'
        expected_path = extractor._extractor._saver.activations_save_path(identifier=expected_model_identifier,
                                                                          stimuli_identifier=stimuli.identifier)
        assert os.path.exists(expected_path)


class TestLayerPCA:

    @pytest.mark.parametrize('batch_size', [None, 2])
    def test_output_shape(self, batch_size):
        extractor, layers, stimuli, _ = get_args()
        n_components = 2

        handle = LayerPCA.hook(extractor,
                               n_components=n_components,
                               stimuli=stimuli,
                               stimuli_identifier=f'test-pca-stimuli-batch_size={batch_size}')
        activations = extractor(stimuli, layers=layers)

        assert activations.sel(layer=layers[0]).sizes['neuroid'] == n_components
        assert activations.sel(layer=layers[1]).sizes['neuroid'] == n_components

    @pytest.mark.parametrize('hook_identifier', ['test-pca', '', None])
    @pytest.mark.parametrize('as_stimulus_set', [True, False])
    def test_save_path(self, hook_identifier, as_stimulus_set):
        extractor, layers, stimuli, _ = get_args(as_stimulus_set=True)
        n_components, force = 3, False
        stimuli_identifier_arg = None if as_stimulus_set else 'test-neural-stimuli-paths'
        stimuli_identifier = stimuli.identifier if as_stimulus_set else 'test-neural-stimuli-paths'

        handle = LayerPCA.hook(extractor, identifier=hook_identifier,
                               n_components=n_components, force=force,
                               stimuli=stimuli, stimuli_identifier=stimuli_identifier_arg)
        hook = list(handle.hook_dict.values())[0]
        _ = extractor(stimuli, layers=layers, stimuli_identifier=stimuli_identifier_arg)

        expected_model_identifier = extractor.identifier
        if hook_identifier is None:
            expected_model_identifier += f'-pca_ncomponents={n_components}_force={force}' \
                                         f'_stimuli_identifier={stimuli_identifier}'
        elif hook_identifier != '':
            expected_model_identifier += f'-{hook_identifier}'
        expected_path = extractor._extractor._saver.activations_save_path(identifier=expected_model_identifier,
                                                                          stimuli_identifier=stimuli_identifier)
        assert os.path.exists(expected_path)

        # Make sure cached PCA pickle files exist
        self.verify_pca_path(hook, extractor, n_components, force, stimuli_identifier)

    def verify_pca_path(self, hook, activations_extractor, n_components, force, stimuli_identifier):
        wrapped_func = LayerPCA._pcas.__closure__[0].cell_contents
        function_identifier = get_function_identifier(wrapped_func,
                                                      call_args={'self': hook,
                                                                 'identifier': activations_extractor.identifier,
                                                                 'n_components': n_components,
                                                                 'force': force,
                                                                 'stimuli_identifier': stimuli_identifier})
        save_path = os.path.join(os.environ['RESULTCACHING_HOME'], function_identifier + '.pkl')
        assert os.path.exists(save_path)


def get_args(as_stimulus_set=False):
    activations_extractor = pytorch_custom()
    layers = ['relu1', 'relu2']
    stimuli = ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png']
    if as_stimulus_set:
        stimuli = _build_stimulus_set(stimuli)
        stimuli.identifier = 'test-neural-stimuli'
    batch_size = 2
    return activations_extractor, layers, stimuli, batch_size

import numpy as np
import os
import pytest
import xarray as xr

from brainio.assemblies import NeuroidAssembly
from model_tools.activations.saving import LayerActivationsSaver


@pytest.mark.parametrize(["a_size", "b_size"], [
    (3, 12),
    (12, 3)
])
def test_merge_layer_activation_files(a_size, b_size):
    saver = LayerActivationsSaver(test_merge_layer_activation_files)

    a = NeuroidAssembly(data=np.random.rand(4, a_size),
                        dims=['stimulus_path', 'neuroid'],
                        coords={'stimulus_path': ['a', 'b', 'c', 'd'],
                                'neuroid_num': ('neuroid', list(range(a_size))),
                                'model': ('neuroid', ['model_a'] * a_size),
                                'layer': ('neuroid', ['layer_a'] * a_size),
                                })
    b = NeuroidAssembly(data=np.random.rand(4, b_size),
                        dims=['stimulus_path', 'neuroid'],
                        coords={'stimulus_path': ['a', 'b', 'c', 'd'],
                                'neuroid_num': ('neuroid', list(range(b_size))),
                                'model': ('neuroid', ['model_a'] * b_size),
                                'layer': ('neuroid', ['layer_b'] * b_size),
                                })
    a_identifier = 'test'
    b_identifier = 'test_tmp'
    stimuli_identifier = 'test_stimuli'

    assert not os.path.exists(saver.activations_save_path(a_identifier, stimuli_identifier))
    assert not os.path.exists(saver.activations_save_path(b_identifier, stimuli_identifier))
    saver.save_batch_activations(a, a_identifier, stimuli_identifier)
    saver.save_batch_activations(b, b_identifier, stimuli_identifier)
    assert os.path.exists(saver.activations_save_path(a_identifier, stimuli_identifier))
    assert os.path.exists(saver.activations_save_path(b_identifier, stimuli_identifier))

    c = saver.merge_layer_activations_files(a_identifier, b_identifier, stimuli_identifier)
    d = xr.concat([a, b], dim='neuroid')
    assert os.path.exists(saver.activations_save_path(a_identifier, stimuli_identifier))
    assert not os.path.exists(saver.activations_save_path(b_identifier, stimuli_identifier))
    assert c.shape == d.shape
    assert (c == d).all()

    saver.delete_activations_file(a_identifier, stimuli_identifier)
    assert not os.path.exists(saver.activations_save_path(a_identifier, stimuli_identifier))

    test_save_path = saver.activations_save_path(a_identifier, stimuli_identifier)
    os.rmdir(os.path.dirname(test_save_path))

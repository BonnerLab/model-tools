import os
from typing import List, Tuple

from brainio.assemblies import NeuroidAssembly
from result_caching import get_function_identifier

from model_tools.activations.core import ActivationsExtractorHelper


def function_identifier(identifier: str, stimuli_identifier: str) -> str:
    call_args = {'identifier': identifier, 'stimuli_identifier': stimuli_identifier}
    return get_function_identifier(function=ActivationsExtractorHelper.from_paths,
                                   call_args=call_args)


def activations_save_path(identifier: str, stimuli_identifier: str) -> str:
    storage_directory = os.path.expanduser(os.getenv('RESULTCACHING_HOME', '~/.result_caching'))
    filename = function_identifier(identifier, stimuli_identifier)
    save_path = os.path.join(storage_directory, filename + '.nc')
    return save_path


def save_batch_activations(batch_activations: NeuroidAssembly, identifier: str, stimuli_identifier: str) -> None:
    # Save a batch of activations by extending an existing file (creates if doesn't already exist)
    save_path = activations_save_path(identifier, stimuli_identifier)
    batch_activations.to_netcdf(path=save_path,
                                extending_dim='stimulus_path',
                                unlimited_dims=['stimulus_path', 'neuroid'])


def load_activations(identifier: str, stimuli_identifier: str) -> NeuroidAssembly:
    # Lazily load activations and convert them to a NeuroidAssembly
    save_path = activations_save_path(identifier, stimuli_identifier)
    assert os.path.exists(save_path)
    return NeuroidAssembly(from_file=save_path)


def delete_activations_file(identifier: str, stimuli_identifier: str) -> None:
    # Used to clear activations after computing them if user didn't want them cached,
    # and used for deleting temporary files when getting activations for additional layers
    save_path = activations_save_path(identifier, stimuli_identifier)
    assert os.path.exists(save_path)
    os.remove(save_path)


def merge_layer_activations_files(identifier: str, identifier_tmp: str, stimuli_identifier: str) -> NeuroidAssembly:
    save_path = activations_save_path(identifier, stimuli_identifier)
    save_path_tmp = activations_save_path(identifier_tmp, stimuli_identifier)
    assert os.path.exists(save_path)
    assert os.path.exists(save_path_tmp)

    # Get all variables we'll need from the assemblies and then close them
    assembly, assembly_tmp = load_activations(identifier, stimuli_identifier), \
                             load_activations(identifier_tmp, stimuli_identifier)
    num_stimuli = assembly.sizes['stimulus_path']
    num_neuroid, num_neuroid_tmp = assembly.sizes['neuroid'], \
                                   assembly_tmp.sizes['neuroid']
    del assembly, assembly_tmp      # Closing because they point to files that we will rename and write to below!

    # Merge onto the file with more activations (i.e. the smaller file should have the `tmp` identifier)
    if num_neuroid < num_neuroid_tmp:
        num_neuroid, num_neuroid_tmp = num_neuroid_tmp, num_neuroid
        os.rename(save_path_tmp, save_path_tmp + '-')
        os.rename(save_path, save_path_tmp)
        os.rename(save_path_tmp + '-', save_path)

    # Pick a batch size for traversing activations dimension
    max_mem = 51380224          # Number of activations in ResNet50 conv1 when passed a batch of 64 224x224 images
    batch_size = max([max_mem // num_stimuli, 1])

    # Reopen the assembly file we'll be merging
    assembly_tmp = load_activations(identifier_tmp, stimuli_identifier)

    # Walk through activations_tmp and incrementally append to the file for activations
    for i in range(0, num_neuroid_tmp, batch_size):
        batch_start, batch_end = i, i + batch_size
        assembly_tmp_batch = assembly_tmp.isel(neuroid=slice(batch_start, batch_end))
        assembly_tmp_batch.to_netcdf(path=save_path,
                                     extending_dim='neuroid',
                                     unlimited_dims=['stimulus_path', 'neuroid'])

    # Clean up the temporary activations file
    del assembly_tmp
    delete_activations_file(identifier_tmp, stimuli_identifier)

    # Lazily load the entire set of activations and return
    assembly = load_activations(identifier, stimuli_identifier)
    return assembly


def stored_layers_overlap(identifier: str, stimuli_identifier: str, layers: List[str]) -> \
        Tuple[bool, List[str], List[str]]:
    """
    Figure out which layers have already been computed for this model and stimuli, and which are missing
    :param identifier: Activations model identifier.
    :param stimuli_identifier: Stimuli identifier.
    :param layers: Which layers have been requested.
    :return: Tuple of bool and 2 lists.
             A) Whether or not any layers have been computed and stored to disk.
             B) The layers which have already been computed and stored to disk.
             C) The missing layers.
    """
    save_path = activations_save_path(identifier, stimuli_identifier)

    # No cached activations for this model and stimulus set. All layers are missing.
    if not os.path.exists(save_path):
        return False, [], layers

    assembly = load_activations(identifier, stimuli_identifier)
    layers_computed = set(assembly['layer'].values)
    layers_missing = set(layers) - layers_computed

    return True, list(layers_computed), list(layers_missing)
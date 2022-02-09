import os
import uuid

import functools
import logging
from collections import OrderedDict
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm import tqdm

from brainio.assemblies import NeuroidAssembly, walk_coords
from brainio.stimuli import StimulusSet
from model_tools.utils import fullname

from model_tools.activations.saving import LayerActivationsSaver


class Defaults:
    batch_size = 64


class ActivationsExtractorHelper:
    def __init__(self, get_activations, preprocessing, identifier=None, batch_size=Defaults.batch_size):
        '''
        :param identifier: an activations identifier for the stored results file. None to disable saving.
        '''
        self._logger = logging.getLogger(fullname(self))

        self._batch_size = batch_size
        self.identifier = identifier
        self.get_activations = get_activations
        self.preprocess = preprocessing or (lambda x: x)
        self._stimulus_set_hooks = {}
        self._batch_activations_hooks = {}
        self._saver = LayerActivationsSaver(self.from_paths)

    def __call__(self, stimuli, layers, stimuli_identifier=None):
        '''
        :param stimuli_identifier: a stimuli identifier for the stored results file. False to disable saving.
        '''
        if isinstance(stimuli, StimulusSet):
            return self.from_stimulus_set(stimulus_set=stimuli, layers=layers, stimuli_identifier=stimuli_identifier)
        else:
            return self.from_paths(stimuli_paths=stimuli, layers=layers, stimuli_identifier=stimuli_identifier)

    def from_stimulus_set(self, stimulus_set, layers, stimuli_identifier=None):
        '''
        :param stimuli_identifier: a stimuli identifier for the stored results file.
            False to disable saving. None to use `stimulus_set.identifier`
        '''
        if stimuli_identifier is None:
            stimuli_identifier = stimulus_set.identifier
        for hook in self._stimulus_set_hooks.copy().values():  # copy to avoid stale handles
            stimulus_set = hook(stimulus_set)
        stimuli_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
        activations = self.from_paths(stimuli_paths=stimuli_paths, layers=layers, stimuli_identifier=stimuli_identifier)
        activations = attach_stimulus_set_meta(activations, stimulus_set)
        return activations

    def from_paths(self, stimuli_paths, layers, stimuli_identifier=None):
        if layers is None:
            layers = ['logits']  # Output layer
        layers = list(set(layers))  # Remove any duplicates

        # In case stimuli paths are duplicates (e.g. multiple trials), we first reduce them to only the paths that need
        # to be run individually, compute activations for those, and then expand the activations to all paths again.
        # This is done here, before storing, so that we only store the reduced activations.
        reduced_paths = self._reduce_paths(stimuli_paths)

        if not self.identifier or not stimuli_identifier:  # Clear saved file after obtaining activations
            self._logger.debug(f'self.identifier `{self.identifier}` or stimuli_identifier {stimuli_identifier} '
                               f'are not set, will not store')
            clear_cache = True
            identifier = str(uuid.uuid4())  # Random identifier
            self._logger.debug(f'Running function: {self._saver.function_identifier(identifier, stimuli_identifier)} '
                               f'for all layers')
            activations = self._get_activations(identifier=identifier, stimuli_identifier=stimuli_identifier,
                                                layers=layers, stimuli_paths=reduced_paths)
        else:  # Return existing activations and only recompute when needed
            clear_cache = False
            identifier = self.identifier
            is_stored, layers_computed, layers_missing = \
                self._saver.stored_layers_overlap(identifier, stimuli_identifier, layers)

            if not is_stored:  # No existing activations stored. Need to compute them
                self._logger.debug(f'Running function: {self._saver.function_identifier(identifier, stimuli_identifier)} '
                                   f'for all layers')
                activations = self._get_activations(identifier=identifier, stimuli_identifier=stimuli_identifier,
                                                    layers=layers, stimuli_paths=reduced_paths)
            elif len(layers_missing) == 0:  # We have all the required layers stored
                self._logger.debug(f'Loading from storage: {self._saver.function_identifier(identifier, stimuli_identifier)}')
                activations = self._saver.load_activations(identifier, stimuli_identifier)
                if len(layers) < len(layers_computed):  # Only a subset of the stored layers have been requested
                    activations = activations.sel(neuroid=np.isin(activations.layer, layers))
            else:  # Compute the missing layers and add them to the stored file
                self._logger.debug(f'Running function: {self._saver.function_identifier(identifier, stimuli_identifier)} '
                                   f'for missing layers: {layers_missing}')
                identifier_tmp = identifier + '-temp'
                activations_tmp = self._get_activations(identifier=identifier_tmp,
                                                        stimuli_identifier=stimuli_identifier,
                                                        layers=layers_missing, stimuli_paths=reduced_paths)
                del activations_tmp     # Points to a file that we'll be reading from in the function below, so close it
                activations = self._saver.merge_layer_activations_files(identifier, identifier_tmp, stimuli_identifier)
                if len(layers) < len(layers_missing) + len(layers_computed):  # Only a subset of the stored layers have been requested
                    activations = activations.sel(neuroid=np.isin(activations.layer, layers))

        # Expand the activations to account for multiple trials
        activations = self._expand_paths(activations, original_paths=stimuli_paths)

        # Load activations into memory and clear up saved activations file, if caching not requested
        if clear_cache:
            # Note that this will run out of memory for many stimuli or activations
            try:
                activations = activations.load()
            finally:
                self._saver.delete_activations_file(identifier, stimuli_identifier)

        return activations

    def _get_activations(self, identifier, stimuli_identifier, layers, stimuli_paths):
        self._logger.info('Running stimuli')

        for batch_start in tqdm(range(0, len(stimuli_paths), self._batch_size),
                                unit_scale=self._batch_size, desc='activations'):
            # Obtain the batch activations at each layer
            batch_end = min(batch_start + self._batch_size, len(stimuli_paths))
            batch_inputs = stimuli_paths[batch_start:batch_end]
            batch_activations = self._get_batch_activations(batch_inputs,
                                                            layer_names=layers,
                                                            batch_size=self._batch_size)

            # Apply any hooks to the batch activations (e.g. max pooling)
            for hook in self._batch_activations_hooks.copy().values():  # copy to avoid handle re-enabling messing with the loop
                batch_activations = hook(batch_activations)

            # Package the batch activations as a NeuroidAssembly
            batch_activations = self._package(batch_activations, stimuli_paths)

            # Append the batch activations to an incrementally growing file on disk
            save_batch_activations(batch_activations, identifier, stimuli_identifier)

        # Lazily load all activations from disk
        activations = load_activations(identifier, stimuli_identifier)

        return activations

    def _reduce_paths(self, stimuli_paths):
        return list(set(stimuli_paths))

    def _expand_paths(self, activations, original_paths):
        activations_paths = activations['stimulus_path'].values
        argsort_indices = np.argsort(activations_paths)
        sorted_x = activations_paths[argsort_indices]
        sorted_index = np.searchsorted(sorted_x, original_paths)
        index = [argsort_indices[i] for i in sorted_index]
        return activations[{'stimulus_path': index}]

    def register_batch_activations_hook(self, hook):
        r'''
        The hook will be called every time a batch of activations is retrieved.
        The hook should have the following signature::

            hook(batch_activations) -> batch_activations

        The hook should return new batch_activations which will be used in place of the previous ones.
        '''

        handle = HookHandle(self._batch_activations_hooks)
        self._batch_activations_hooks[handle.id] = hook
        return handle

    def register_stimulus_set_hook(self, hook):
        r'''
        The hook will be called every time before a stimulus set is processed.
        The hook should have the following signature::

            hook(stimulus_set) -> stimulus_set

        The hook should return a new stimulus_set which will be used in place of the previous one.
        '''

        handle = HookHandle(self._stimulus_set_hooks)
        self._stimulus_set_hooks[handle.id] = hook
        return handle

    def _get_batch_activations(self, inputs, layer_names, batch_size):
        inputs, num_padding = self._pad(inputs, batch_size)
        preprocessed_inputs = self.preprocess(inputs)
        activations = self.get_activations(preprocessed_inputs, layer_names)
        assert isinstance(activations, OrderedDict)
        activations = self._unpad(activations, num_padding)
        return activations

    def _pad(self, batch_images, batch_size):
        num_images = len(batch_images)
        if num_images % batch_size == 0:
            return batch_images, 0
        num_padding = batch_size - (num_images % batch_size)
        padding = np.repeat(batch_images[-1:], repeats=num_padding, axis=0)
        return np.concatenate((batch_images, padding)), num_padding

    def _unpad(self, layer_activations, num_padding):
        return change_dict(layer_activations, lambda values: values[:-num_padding or None])

    def _package(self, layer_activations, stimuli_paths) -> NeuroidAssembly:
        shapes = [a.shape for a in layer_activations.values()]
        self._logger.debug('Activations shapes: {}'.format(shapes))
        self._logger.debug('Packaging individual layers')
        layer_assemblies = [self._package_layer(single_layer_activations, layer=layer, stimuli_paths=stimuli_paths) for
                            layer, single_layer_activations in tqdm(layer_activations.items(), desc='layer packaging')]
        # merge manually instead of using merge_data_arrays since `xarray.merge` is very slow with these large arrays
        # complication: (non)neuroid_coords are taken from the structure of layer_assemblies[0] i.e. the 1st assembly;
        # using these names/keys for all assemblies results in KeyError if the first layer contains flatten_coord_names
        # (see _package_layer) not present in later layers, e.g. first layer = conv, later layer = transformer layer
        self._logger.debug('Merging layer assemblies')
        model_assembly = np.concatenate([a.values for a in layer_assemblies],
                                        axis=layer_assemblies[0].dims.index('neuroid'))
        nonneuroid_coords = {coord: (dims, values) for coord, dims, values in walk_coords(layer_assemblies[0])
                             if set(dims) != {'neuroid'}}
        neuroid_coords = {coord: [dims, values] for coord, dims, values in walk_coords(layer_assemblies[0])
                          if set(dims) == {'neuroid'}}
        for layer_assembly in layer_assemblies[1:]:
            for coord in neuroid_coords:
                neuroid_coords[coord][1] = np.concatenate((neuroid_coords[coord][1], layer_assembly[coord].values))
            assert layer_assemblies[0].dims == layer_assembly.dims
            for dim in set(layer_assembly.dims) - {'neuroid'}:
                for coord in layer_assembly[dim].coords:
                    assert (layer_assembly[coord].values == nonneuroid_coords[coord][1]).all()
        neuroid_coords = {coord: (dims_values[0], dims_values[1])  # re-package as tuple instead of list for xarray
                          for coord, dims_values in neuroid_coords.items()}
        model_assembly = type(layer_assemblies[0])(model_assembly, coords={**nonneuroid_coords, **neuroid_coords},
                                                   dims=layer_assemblies[0].dims)
        return model_assembly

    def _package_layer(self, layer_activations, layer, stimuli_paths) -> NeuroidAssembly:
        assert layer_activations.shape[0] == len(stimuli_paths)
        activations, flatten_indices = flatten(layer_activations, return_index=True)  # collapse for single neuroid dim
        assert flatten_indices.shape[1] in [1, 2, 3]
        # see comment in _package for an explanation why we cannot simply have 'channel' for the FC layer
        if flatten_indices.shape[1] == 1:  # FC
            flatten_coord_names = ['channel', 'channel_x', 'channel_y']
        elif flatten_indices.shape[1] == 2:  # Transformer
            flatten_coord_names = ['channel', 'embedding']
        elif flatten_indices.shape[1] == 3:  # 2DConv
            flatten_coord_names = ['channel', 'channel_x', 'channel_y']
        flatten_coords = {
            flatten_coord_names[i]: [sample_index[i] if i < flatten_indices.shape[1] else np.nan for sample_index in
                                     flatten_indices]
            for i in range(len(flatten_coord_names))}
        layer_assembly = NeuroidAssembly(
            activations,
            coords={**{'stimulus_path': stimuli_paths,
                       'neuroid_num': ('neuroid', list(range(activations.shape[1]))),
                       'model': ('neuroid', [self.identifier] * activations.shape[1]),
                       'layer': ('neuroid', [layer] * activations.shape[1]),
                       },
                    **{coord: ('neuroid', values) for coord, values in flatten_coords.items()}},
            dims=['stimulus_path', 'neuroid']
        )
        neuroid_id = ['.'.join([f'{value}' for value in values]) for values in zip(*[
            layer_assembly[coord].values for coord in ['model', 'layer', 'neuroid_num']])]
        layer_assembly['neuroid_id'] = 'neuroid', neuroid_id
        return layer_assembly

    def insert_attrs(self, wrapper):
        wrapper.from_stimulus_set = self.from_stimulus_set
        wrapper.from_paths = self.from_paths
        wrapper.register_batch_activations_hook = self.register_batch_activations_hook
        wrapper.register_stimulus_set_hook = self.register_stimulus_set_hook


def change_dict(d, change_function, keep_name=False, multithread=False):
    if not multithread:
        map_fnc = map
    else:
        pool = ThreadPool()
        map_fnc = pool.map

    def apply_change(layer_values):
        layer, values = layer_values
        values = change_function(values) if not keep_name else change_function(layer, values)
        return layer, values

    results = map_fnc(apply_change, d.items())
    results = OrderedDict(results)
    if multithread:
        pool.close()
    return results


def lstrip_local(path):
    parts = path.split(os.sep)
    try:
        start_index = parts.index('.brainio')
    except ValueError:  # not in list -- perhaps custom directory
        return path
    path = os.sep.join(parts[start_index:])
    return path


def attach_stimulus_set_meta(assembly, stimulus_set):
    stimulus_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
    stimulus_paths = [lstrip_local(path) for path in stimulus_paths]
    assembly_paths = [lstrip_local(path) for path in assembly['stimulus_path'].values]
    assert (np.array(assembly_paths) == np.array(stimulus_paths)).all()
    assembly['stimulus_path'] = stimulus_set['image_id'].values
    assembly = assembly.rename({'stimulus_path': 'image_id'})
    for column in stimulus_set.columns:
        assembly[column] = 'image_id', stimulus_set[column].values
    assembly = assembly.stack(presentation=('image_id',))
    return assembly


class HookHandle:
    next_id = 0

    def __init__(self, hook_dict):
        self.hook_dict = hook_dict
        self.id = HookHandle.next_id
        HookHandle.next_id += 1
        self._saved_hook = None

    def remove(self):
        hook = self.hook_dict[self.id]
        del self.hook_dict[self.id]
        return hook

    def disable(self):
        self._saved_hook = self.remove()

    def enable(self):
        self.hook_dict[self.id] = self._saved_hook
        self._saved_hook = None


def flatten(layer_output, return_index=False):
    flattened = layer_output.reshape(layer_output.shape[0], -1)
    if not return_index:
        return flattened

    def cartesian_product_broadcasted(*arrays):
        '''
        http://stackoverflow.com/a/11146645/190597
        '''
        broadcastable = np.ix_(*arrays)
        broadcasted = np.broadcast_arrays(*broadcastable)
        dtype = np.result_type(*arrays)
        rows, cols = functools.reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
        out = np.empty(rows * cols, dtype=dtype)
        start, end = 0, rows
        for a in broadcasted:
            out[start:end] = a.reshape(-1)
            start, end = end, end + rows
        return out.reshape(cols, rows).T

    index = cartesian_product_broadcasted(*[np.arange(s, dtype='int') for s in layer_output.shape[1:]])
    return flattened, index

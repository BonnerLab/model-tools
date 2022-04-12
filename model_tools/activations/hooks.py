from abc import ABC, abstractmethod
import logging
import os
from typing import Optional, Union, Iterable, Dict

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from brainio.stimuli import StimulusSet
from model_tools.activations import ActivationsModel
from model_tools.activations.core import flatten, change_dict
from model_tools.utils import fullname, s3
from model_tools.utils.pca import IncrementalPCAPytorch, PCAPytorch
from result_caching import store_dict

Stimuli = Union[Iterable[str], StimulusSet, Iterable[os.PathLike]]
BasePCA = Union[IncrementalPCAPytorch, PCAPytorch]


class LayerHookBase(ABC):

    def __init__(self, activations_extractor: ActivationsModel, identifier: Optional[str] = None):
        self._extractor = activations_extractor
        self.identifier = identifier
        self.handle = None

    def __call__(self, batch_activations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.setup(batch_activations)
        return change_dict(batch_activations, self.layer_apply, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor: ActivationsModel, identifier: Optional[str] = None, **kwargs):
        hook = cls(activations_extractor=activations_extractor, identifier=identifier, **kwargs)
        assert not cls.is_hooked(activations_extractor), f"{cls.__name__} is already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor: ActivationsModel) -> bool:
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())

    def setup(self, batch_activations: Dict[str, np.ndarray]) -> None:
        pass

    @abstractmethod
    def layer_apply(self, layer: str, activations: np.ndarray) -> np.ndarray:
        pass


class LayerGlobalMaxPool2d(LayerHookBase):

    def __init__(self, *args, identifier: Optional[str] = None, **kwargs):
        if identifier is None:
            identifier = 'maxpool'

        super(LayerGlobalMaxPool2d, self).__init__(*args, **kwargs, identifier=identifier)
    
    def layer_apply(self, layer: str, activations: np.ndarray) -> np.ndarray:
        if activations.ndim != 4:
            return activations
        return np.max(activations, axis=(2, 3))


class LayerRandomProjection(LayerHookBase):

    def __init__(self, *args,
                 n_components: int = 1000,
                 force: bool = False,
                 identifier: Optional[str] = None,
                 **kwargs):
        if identifier is None:
            identifier = f'randproj_ncomponents={n_components}_force={force}'

        super(LayerRandomProjection, self).__init__(*args, **kwargs, identifier=identifier)
        self._n_components = n_components
        self._force = force
        self._layer_ws = {}

    def layer_apply(self, layer: str, activations: np.ndarray) -> np.ndarray:
        activations = flatten(activations)
        if activations.shape[1] <= self._n_components and not self._force:
            return activations
        if layer not in self._layer_ws:
            w = np.random.normal(size=(activations.shape[-1], self._n_components)) / np.sqrt(self._n_components)
            self._layer_ws[layer] = w
        else:
            w = self._layer_ws[layer]
        activations = activations @ w
        return activations


class LayerPCA(LayerHookBase):

    def __init__(self, *args,
                 n_components: int = 1000,
                 force: bool = False,
                 stimuli: Optional[Stimuli] = None,
                 stimuli_identifier: Optional[str] = None,
                 identifier: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 **kwargs):
        if stimuli is None:
            # Default to ImageNet validation with 1 image per class
            stimuli = _get_imagenet_val(n_components)
            stimuli_identifier = 'brainscore-imagenetval'
        if isinstance(stimuli, StimulusSet) and stimuli_identifier is None and hasattr(stimuli, 'identifier'):
            stimuli_identifier = stimuli.identifier
        if stimuli_identifier is None:
            raise ValueError('If passing a list of paths for stimuli '
                             'or a StimulusSet without an identifier attribute, '
                             'you must provide a stimuli_identifier')

        if identifier is None:
            identifier = f'pca_ncomponents={n_components}_force={force}_stimuli_identifier={stimuli_identifier}'

        super(LayerPCA, self).__init__(*args, **kwargs, identifier=identifier)
        self._n_components = n_components
        self._force = force
        self._stimuli_identifier = stimuli_identifier
        self._stimuli = stimuli
        self._batch_size = batch_size
        self._device = device
        self._logger = logging.getLogger(fullname(self))
        self._layer_pcas = {}

    def setup(self, batch_activations) -> None:
        layers = batch_activations.keys()
        missing_layers = [layer for layer in layers if layer not in self._layer_pcas]
        if len(missing_layers) == 0:
            return
        layer_pcas = self._pcas(identifier=self._extractor.identifier,
                                layers=missing_layers,
                                n_components=self._n_components,
                                force=self._force,
                                stimuli_identifier=self._stimuli_identifier)
        self._layer_pcas = {**self._layer_pcas, **layer_pcas}
    
    def layer_apply(self, layer: str, activations: np.ndarray) -> np.ndarray:
        pca = self._layer_pcas[layer]
        activations = flatten(activations)
        if pca is None:
            return activations
        return pca.transform(torch.from_numpy(activations).to(self._device))

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _pcas(self, identifier, layers, n_components, force, stimuli_identifier) -> Dict[str, BasePCA]:
        self._logger.debug(f'Retrieving {stimuli_identifier} activations')
        self.handle.disable()
        activations = self._extractor(self._stimuli, layers=layers, stimuli_identifier=False)
        activations = {layer: activations.sel(layer=layer).values
                       for layer in np.unique(activations['layer'])}
        assert len(set(layer_activations.shape[0] for layer_activations in activations.values())) == 1, "stimuli differ"
        self.handle.enable()

        self._logger.debug(f'Computing {stimuli_identifier} principal components')
        progress = tqdm(total=len(activations), desc="layer principal components", leave=False)

        def init_and_progress(layer, activations):
            activations = flatten(activations)
            if activations.shape[1] <= n_components and not force:
                self._logger.debug(f"Not computing principal components for {layer} "
                                   f"activations {activations.shape} as shape is small enough already")
                progress.update(1)
                return None
            n_components_ = n_components if activations.shape[1] > n_components else activations.shape[1]
            if self._batch_size is None:
                pca = PCAPytorch(n_components_, device=self._device)
                pca.fit(torch.from_numpy(activations).to(self._device))
            else:
                pca = IncrementalPCAPytorch(n_components_, device=self._device)
                for i in range(0, activations.shape[0], self._batch_size):
                    activations_batch = torch.from_numpy(activations[i:i + self._batch_size]).to(self._device)
                    pca.fit_partial(activations_batch)
            return pca

        layer_pcas = change_dict(activations, init_and_progress, keep_name=True,
                                 multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        progress.close()
        return layer_pcas


def _get_imagenet_val(num_images):
    _logger = logging.getLogger(fullname(_get_imagenet_val))
    num_classes = 1000
    num_images_per_class = (num_images - 1) // num_classes
    base_indices = np.arange(num_images_per_class).astype(int)
    indices = []
    for i in range(num_classes):
        indices.extend(50 * i + base_indices)
    for i in range((num_images - 1) % num_classes + 1):
        indices.extend(50 * i + np.array([num_images_per_class]).astype(int))

    framework_home = os.path.expanduser(os.getenv('MT_HOME', '~/.model-tools'))
    imagenet_filepath = os.getenv('MT_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
    imagenet_dir = f"{imagenet_filepath}-files"
    os.makedirs(imagenet_dir, exist_ok=True)

    if not os.path.isfile(imagenet_filepath):
        os.makedirs(os.path.dirname(imagenet_filepath), exist_ok=True)
        _logger.debug(f"Downloading ImageNet validation to {imagenet_filepath}")
        s3.download_file("imagenet2012-val.hdf5", imagenet_filepath)

    filepaths = []
    with h5py.File(imagenet_filepath, 'r') as f:
        for index in indices:
            imagepath = os.path.join(imagenet_dir, f"{index}.png")
            if not os.path.isfile(imagepath):
                image = np.array(f['val/images'][index])
                Image.fromarray(image).save(imagepath)
            filepaths.append(imagepath)

    return filepaths

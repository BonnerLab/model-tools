from abc import ABC, abstractmethod
import logging
from typing import Any, Union, Optional, List, Dict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from brainio.stimuli import StimulusSet
from brainio.assemblies import NeuroidAssembly
from model_tools.activations import PytorchWrapper, TensorflowWrapper, TensorflowSlimWrapper, KerasWrapper
from model_tools.activations.core import change_dict
from model_tools.utils import fullname
from result_caching import store_dict

ActivationsModel = Union[PytorchWrapper, TensorflowWrapper, TensorflowSlimWrapper, KerasWrapper]


class ModelActivationsAnalysis(ABC):

    def __init__(self, activations_model: ActivationsModel):
        self._activations_model = activations_model
        self._logger = logging.getLogger(fullname(self))
        self._layer_results = None
        self._metadata = {}

    @abstractmethod
    def analysis_func(self, assembly: NeuroidAssembly, **kwargs) -> Any:
        """
        This is the only function that needs to be implemented in subclasses.
        Any additional kwargs that this function takes must also be passed to __call__().
        :param assembly: An assembly on which to perform analysis
            (e.g. acitvations or neural recordings in response to stimuli).
        :param kwargs: Additional arguments required for this analysis (e.g. a batch size).
            Any additional kwargs that this function takes must also be passed to __call__().
        """
        pass

    @property
    def results(self) -> Dict[str, Any]:
        """
        This property can be overidden in subclasses if you wish to transform the format of the results
        (e.g. merge the dictionary into an xarray DataArray).
        """
        return self._layer_results

    @property
    def identifier(self) -> str:
        return self._activations_model.identifier

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def __call__(self,
                 stimuli: Union[StimulusSet, List[str], List[Path]],
                 layers: List[str],
                 stimuli_identifier: Optional[Union[bool, str]] = None,
                 multithread: bool = False,
                 **kwargs) -> Dict[str, Any]:
        """
        Perform an analysis on a model given a set of stimuli.
        :param stimuli: Stimuli to obtain activations from.
        :param layers: Layers to obtain activations from.
        :param stimuli_identifier: Identifier used for stimulus set (used for caching filename).
            Setting to False will prevent activations and analysis results from being cached.
            If caching is desired, either 'stimuli' must be a StimulusSet or 'stimuli_identifier' must be a string.
        :param multithread: Whether or not to multithread analysis across layers.
        :param kwargs: Additional arguments that are passed down to 'self.analysis_func()'.
        :return: A dictionary mapping layer names to arbitrarily-structured analysis results.
        """
        if isinstance(stimuli, StimulusSet) and stimuli_identifier is None:
            stimuli_identifier = stimuli.identifier

        if stimuli_identifier:
            self._layer_results = self._run_analysis_stored(identifier=self._activations_model.identifier,
                                                            stimuli=stimuli,
                                                            layers=layers,
                                                            stimuli_identifier=stimuli_identifier,
                                                            multithread=multithread,
                                                            analysis_kwargs=kwargs)
            self._metadata['stimuli_identifier'] = stimuli_identifier
        else:
            self._layer_results = self._run_analysis(stimuli=stimuli,
                                                     layers=layers,
                                                     stimuli_identifier=stimuli_identifier,
                                                     multithread=multithread,
                                                     **kwargs)

        return self.results

    @store_dict(dict_key='layers', identifier_ignore=['stimuli', 'layers', 'multithread'])
    def _run_analysis_stored(self,
                             identifier: str,
                             stimuli: Union[StimulusSet, List[str], List[Path]],
                             layers: List[str],
                             stimuli_identifier: str,
                             multithread: bool = False,
                             analysis_kwargs: Dict = {}) -> Dict[str, Any]:
        layer_results = self._run_analysis(stimuli=stimuli,
                                           layers=layers,
                                           stimuli_identifier=stimuli_identifier,
                                           multithread=multithread,
                                           **analysis_kwargs)
        self._metadata['stimuli_identifier'] = stimuli_identifier
        return layer_results

    def _run_analysis(self,
                      stimuli: Union[StimulusSet, List[str], List[Path]],
                      layers: List[str],
                      stimuli_identifier: Optional[Union[bool, str]] = None,
                      multithread: bool = False,
                      **kwargs) -> Dict[str, Any]:
        self._logger.debug('Obtaining activations')
        layer_activations = self._activations_model(stimuli=stimuli, layers=layers,
                                                    stimuli_identifier=stimuli_identifier)
        layer_activations = {layer: layer_activations.sel(layer=layer)
                             for layer in np.unique(layer_activations['layer'])}

        self._logger.debug('Performing analyses')
        progress = tqdm(total=len(layer_activations), desc="layer analyses")

        def do_layer_analysis(layer, activations):
            result = self.analysis_func(assembly=activations, **kwargs)
            progress.update(1)
            return result

        layer_results = change_dict(layer_activations, change_function=do_layer_analysis,
                                    keep_name=True, multithread=multithread)
        progress.close()
        return layer_results
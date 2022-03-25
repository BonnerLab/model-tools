import os
from model_tools.activations.core import change_dict


class GlobalMaxPool2d:
    def __init__(self, activations_extractor):
        self._extractor = activations_extractor

    def __call__(self, batch_activations):
        def apply(layer, activations):
            if activations.ndim != 4:
                return activations
            return activations.max(axis=(2, 3))

        return change_dict(batch_activations, apply, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor):
        hook = GlobalMaxPool2d(activations_extractor)
        assert not cls.is_hooked(activations_extractor), "GlobalMaxPool2d already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())

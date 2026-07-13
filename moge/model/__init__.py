import importlib
from typing import *

if TYPE_CHECKING:
    from .v2 import mogeModel as mogeModelV2


def import_model_class_by_version(version: str) -> Type[Union['mogeModelV1', 'mogeModelV2']]:
    assert version in ['v1', 'v2'], f'Unsupported model version: {version}'
    
    try:
        module = importlib.import_module(f'.{version}', __package__)
    except ModuleNotFoundError:
        raise ValueError(f'Model version "{version}" not found.')

    cls = getattr(module, 'mogeModel')
    return cls

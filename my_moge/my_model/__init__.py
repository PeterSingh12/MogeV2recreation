import importlib
from typing import *

if TYPE_CHECKING:
    from .v2 import my_mogeModel as my_mogeModelV2


def import_model_class_by_version(version: str) -> Type[Union['my_mogeModelV1', 'my_mogeModelV2']]:
    assert version in ['v1', 'v2'], f'Unsupported model version: {version}'
    
    try:
        module = importlib.import_module(f'.{version}', __package__)
    except ModuleNotFoundError:
        raise ValueError(f'Model version "{version}" not found.')

    cls = getattr(module, 'my_mogeModel')
    return cls

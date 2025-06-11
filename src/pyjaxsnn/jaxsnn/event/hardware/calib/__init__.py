# pylint: disable=line-too-long
from typing import NamedTuple


class WaferConfig(NamedTuple):
    file: str
    name: str
    weight_scaling: float

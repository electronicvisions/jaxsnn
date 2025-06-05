from __future__ import annotations
import dataclasses

from typing import (
    TYPE_CHECKING,
    Optional,
    Tuple,
    Union,
    Callable,
    Dict,
    TypeVar,
)

import jax
import numpy as np

if TYPE_CHECKING:
    T_cls = TypeVar("T_cls", bound=type)  # pylint: disable=invalid-name

    def _tm_struct(
        cls: T_cls  # pylint: disable=unused-argument
    ) -> T_cls:
        ...

else:
    from tree_math import struct as _tm_struct  # pylint: disable=unused-import


ArrayLike = Union[jax.Array, np.ndarray, float]
Array = Union[jax.Array, np.ndarray]


# Base state hierarchy
@dataclasses.dataclass
class BaseState:
    """Base class for all neuron states across paradigms"""


@dataclasses.dataclass
class BasePopulation:
    """Base class for all neuron populations across paradigms"""


# Generic type variables for paradigm-agnostic typing
Data = TypeVar("Data")  # Generic data type (can be jax.Array, Spike, etc.)
StateT = TypeVar('StateT', bound=BaseState)

# Core types that all paradigms use
Parameter = jax.Array
Parameters = Dict[str, Parameter]

# Generic I/O types - Data type will be specialized by paradigms
Inputs = Dict[str, Data]
Outputs = Dict[str, Data]
States = Dict[str, StateT]

# Module-level function types - basis for all paradigms
InitFn = Callable[[jax.Array], Optional[Parameter]]
StepFn = Callable[
    [Dict[str, Data], StateT, Parameter],
    Tuple[Optional[StateT], Data],
]
StateFn = Callable[[], Tuple[Optional[StateT], Optional[Data]]]

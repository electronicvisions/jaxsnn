from __future__ import annotations
from typing import (
    Tuple,
    Callable,
    Dict,
    Any,
    Optional,
    Generic,
    Protocol,
    TypeVar,
)
import dataclasses

import jax
from jaxsnn.base.types import (
    BaseState,
    StateT,
    Parameter,
    InitFn,
    Parameters,
    BasePopulation,
)


# Base protocol that all paradigms must implement
class GeneratorFunctions(Protocol):
    """Base protocol for all module function containers"""
    init: InitFn
    step: StepFn
    state: StateFn


GeneratorFns = TypeVar("GeneratorFns", bound=GeneratorFunctions)


# Discrete-specific BaseModule
@dataclasses.dataclass
class BaseModule(Generic[GeneratorFns]):
    """
    Base class for discrete-time network modules that follow the base protocol.

    This provides the common structure for discrete-time paradigms.
    Event-driven paradigms use EventBaseModule instead due to incompatible
    function signatures.
    """
    generator: Callable[..., GeneratorFns]
    parameters: Dict[str, Any]
    _fns: Optional[GeneratorFns] = dataclasses.field(
        init=False, repr=False, default=None
    )

    @property
    def fns(self) -> GeneratorFns:
        """Lazily-assigned functions. Must be set before access."""
        if self._fns is None:
            raise AttributeError(
                "'fns' has not been set. Please assign the generated "
                "functions before accessing them."
            )
        return self._fns

    @fns.setter
    def fns(self, value: GeneratorFns) -> None:
        """Setter for the functions."""
        self._fns = value


# Specification for discrete-time paradigm
DenseData = jax.Array
IOData = Dict[str, DenseData]

# Discrete state types
States = Dict[str, Optional[BaseState]]

# Discrete-specific model apply function
ModelApplyFn = Callable[[IOData, Parameters], Tuple[Optional[States], IOData]]

# Discrete-specific model init function
ModelInitFn = Callable[[jax.Array], Parameters]

# Discrete function signatures using the base StateT TypeVar
StepFn = Callable[
    [Dict[str, DenseData], StateT, Parameter],
    Tuple[Optional[StateT], DenseData],
]

StateFn = Callable[[], Tuple[Optional[StateT], Optional[DenseData]]]

ApplyFn = Callable[
    [Parameters, IOData],
    Optional[IOData]
]


# Define the function containers at module level to avoid circular dependencies
@dataclasses.dataclass
class ProjectionFunctions:
    """Function container for projection modules"""
    init: InitFn
    state: StateFn
    step: StepFn


@dataclasses.dataclass
class PopulationFunctions:
    """Function container for population modules"""
    init: InitFn
    state: StateFn
    step: StepFn


@dataclasses.dataclass
class Projection(BaseModule[ProjectionFunctions]):
    """A projection module for the discrete paradigm."""
    # Provide the nested Functions as an alias for API compatibility
    Functions = ProjectionFunctions
    generator: Callable[[int, int], ProjectionFunctions]


@dataclasses.dataclass
class Population(BasePopulation, BaseModule[PopulationFunctions]):
    """A population module for the discrete paradigm."""
    # Provide the nested Functions as an alias for API compatibility
    Functions = PopulationFunctions
    generator: Callable[[float], PopulationFunctions]
    size: int


@dataclasses.dataclass
class SourcePopulation(Population):
    """A source population module for the discrete paradigm."""

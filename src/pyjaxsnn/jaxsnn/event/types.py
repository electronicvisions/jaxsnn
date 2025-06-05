from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    NamedTuple,
    Tuple,
    TypeVar,
    Dict,
    Optional,
    Protocol,
    Union,
)

import numpy as np
import jax
import jax.numpy as jnp
import optax

from jaxsnn.base.types import (
    BaseState,
    # Core types
    StateT,
    Parameters,
    # Base functions and protocols
    InitFn,
    # I/O types - will be specialized
    States,
    BasePopulation,
)
try:
    from jaxsnn.event.hardware.modules.base_module import BaseModule
except ImportError:
    BaseModule = Any  # Fallback if the module is not available

if TYPE_CHECKING:
    from jaxsnn.event.hardware.experiment import Experiment

    T_cls = TypeVar("T_cls", bound=type)  # pylint: disable=invalid-name

    def _tm_struct(
        cls: T_cls,  # pylint: disable=unused-argument
    ) -> T_cls:
        ...

else:
    from tree_math import struct as _tm_struct


class EventT(Protocol):
    time: jax.Array
    idx: jax.Array


class LayeredEventT(Protocol):
    time: jax.Array
    idx: jax.Array
    internal: jax.Array
    layer_idx: jax.Array


@dataclass
@_tm_struct
class Event:
    time: jax.Array
    idx: jax.Array


@dataclass
@_tm_struct
class Spike:
    """
    Represents a spike event in the network.
    Empty events are described with idx=-1 and time=jnp.inf.

    :param time: Time of the spike.
    :param idx: Index of the neuron that spiked.
    :param current: Current associated with the spike (if any).
    :param layer_idx: Index of the layer where the spike occurred.
    :param internal: Boolean flag indicating if the spike is internal.
    """
    time: Union[jax.Array, np.ndarray] = field(
        default_factory=lambda: jnp.array(jnp.inf))
    idx: Union[jax.Array, np.ndarray] = field(
        default_factory=lambda: jnp.array(-1, dtype=int))
    current: Union[jax.Array, np.ndarray] = field(
        default_factory=lambda: jnp.array(0.0))
    layer_idx: Union[jax.Array, np.ndarray] = field(
        default_factory=lambda: jnp.array(-1, dtype=int))
    internal: Union[jax.Array, np.ndarray] = field(
        default_factory=lambda: jnp.array(False, dtype=bool))

    @property
    def shape_(self):
        return self.time.shape  # pylint: disable=no-member

    def __getitem__(self, key) -> Spike:
        return jax.tree_util.tree_map(lambda leaf: leaf[key], self)

    @classmethod
    def empty(cls, shape) -> Spike:
        default_spike = Spike()
        return jax.tree_util.tree_map(
            lambda x: jnp.full(shape, x, dtype=x.dtype), default_spike
        )

    def where(self, cond: jax.Array) -> Spike:
        """Conditional selection like jnp.where."""
        return jax.tree_util.tree_map(
            lambda spike_leaf, empty_leaf: jnp.where(
                cond, spike_leaf, empty_leaf
            ),
            self, self.empty((1,)),
        )

    def sort(self, axis: int = 0) -> Spike:
        """Sorts the spike events by time."""
        perm = jnp.argsort(self.time, axis=axis)
        return jax.tree_util.tree_map(
            lambda leaf: jnp.take_along_axis(leaf, perm, axis=axis), self
        )

    def set_item(self, key, new_value: Spike) -> Spike:
        """Returns a new Spike with values at key updated."""
        return jax.tree_util.tree_map(
            lambda leaf, new_leaf: leaf.at[key].set(new_leaf), self, new_value
        )

    def concatenate(self, other: Spike, axis: int = 0) -> Spike:
        """Concatenates two Spike objects along the specified axis."""
        return jax.tree_util.tree_map(
            lambda leaf,
            other_leaf: jnp.concatenate([leaf, other_leaf], axis=axis),
            self, other
        )

    def empty_like(self) -> Spike:
        """Returns a new Spike with the same shape but empty values."""
        return self.empty(self.shape_)

    def get_internal(self) -> Spike:
        """Returns a new Spike containing only internal spikes."""
        return self.where(self.internal)


@dataclass
@_tm_struct
class Carry:
    parameters: Parameters
    spikes: IOData
    external_spikes: Optional[IOData]
    states: States
    queue_heads: Dict[str, jax.Array]
    queue_indices: Dict[str, jax.Array]


@dataclass
@_tm_struct
class Step:
    parameters: Parameters
    spikes: IOData
    external_spikes: Optional[IOData]
    state: StepState
    step_idx: int
    layer_idx: int
    queue_head: Dict[str, jax.Array]
    queue_indices: Dict[str, jax.Array]


@dataclass
@_tm_struct
class StepState(Generic[StateT]):
    neuron_state: StateT
    time: jax.Array


@dataclass
@_tm_struct
class ProbeStepState(Generic[StateT]):
    neuron_state: StateT
    time: float
    input_queue: jax.Array
    probe_queue: jax.Array


@dataclass
@_tm_struct
class ProbeEvent(Generic[StateT]):
    time: jax.Array
    state: StateT
    probe: jax.Array


class OptState(NamedTuple):
    """
    Container for the optimizer/training state across steps.

    This immutable state groups the underlying optimizer's internal state,
    the current set of model params, and the JAX PRNG key used for
    stochastic operations (e.g., sampling).

    :param opt_state: Optimizer-specific internal state to carry across updates
        (e.g., from optax.init/optax.update).
    :param params: Ordered collection of learnable model parameters to be
        optimized.
    :param rng: JAX PRNG key used for randomized computations; should be split
        and updated between steps.
    """
    opt_state: optax.OptState
    params: Dict[str, Optional[jax.Array]]
    rng: jax.Array


# Event paradigm specializations
EventData = Spike  # Event paradigm uses EventPropSpike as Data
EventState = Optional[BaseState]   # Event paradigm state specialization

# Specialized I/O types for event paradigm
IOData = Dict[str, EventData]

ModelApplyFn = Callable[[IOData, Parameters], Tuple[Optional[States], IOData]]
ModelInitFn = Callable[[jax.Array], Parameters]

Dataset = Tuple[Spike, jax.Array]

# Event-specific types for topology
StepInput = Tuple[Parameters, IOData, StepState, Any, int, jax.Array, Any]
QueueHead = jax.Array
QueueIndex = jax.Array

# Event-specific function signatures
EventStepFn = Callable[
    [Tuple[Parameters, IOData, StepState, Any, int, QueueHead, Any]],
    Tuple[EventData, StepState, QueueHead, QueueIndex],
]
EventStateFn = Callable[[], Optional[StepState]]
EventFn = Callable[[int], Optional[EventData]]
AdjointStepFn = Callable[
    [Tuple[
        Parameters,
        EventData,
        QueueIndex,
        EventData,
        StepState,
        Parameters,
        EventData
    ]],
    Tuple[Parameters, StepState, QueueHead, QueueIndex],
]
NextInputFn = Callable[
    [
        Dict[str, Spike],
        jax.Array,
        float,
        float,
    ],
    Tuple[jax.Array, jax.Array, Spike]
]
MinDelayCheckFn = Callable[
    [
        Dict[str, Spike],
        jax.Array,
    ],
    Tuple[jax.Array, jax.Array]
]
DynamicsFn = Callable[[BaseState, jax.Array], BaseState]
AddGradFn = Callable[
    [Parameters, int, StepState, jax.Array, IOData, int],
    Tuple[Parameters, IOData],
]
SolverFn = Callable[[StateT, float], Spike]

if TYPE_CHECKING:
    GenModuleFn = Callable[
        [int, Experiment, Optional[BaseModule], Optional[BaseModule]],
        BaseModule
    ]


# Event paradigm function containers
@dataclass
class ProjectionFunctions:
    """Function container for event projection modules"""
    init: InitFn
    state: EventStateFn
    event: EventFn  # Event-specific
    hx_module: Optional[GenModuleFn] = None


@dataclass
class PopulationFunctions:
    """Function container for event population modules"""
    init: InitFn
    step: EventStepFn
    state: EventStateFn
    event: EventFn  # Event-specific
    adjoint_step: Optional[AdjointStepFn] = None  # Event-specific
    hx_module: Optional[GenModuleFn] = None


@dataclass
class SourcePopulationFunctions:
    """Function container for event source population modules"""
    init: InitFn
    state: EventStateFn
    event: EventFn
    hx_module: Optional[GenModuleFn] = None


# Event-specific base module (parallel to BaseModule)
EventGeneratorFns = TypeVar("EventGeneratorFns")


@dataclass
class EventBaseModule(Generic[EventGeneratorFns]):
    """
    Base class for event-driven network modules.
    Parallel to BaseModule but with event-specific function signatures.
    """
    generator: Callable[..., EventGeneratorFns]
    parameters: Dict[str, Any]
    _fns: Optional[EventGeneratorFns] = field(
        init=False, repr=False, default=None
    )

    @property
    def fns(self) -> EventGeneratorFns:
        """Lazily-assigned functions. Must be set before access."""
        if self._fns is None:
            raise AttributeError(
                "'fns' has not been set. Please assign the generated "
                "functions before accessing them."
            )
        return self._fns

    @fns.setter
    def fns(self, value: EventGeneratorFns) -> None:
        """Setter for the functions."""
        self._fns = value


# Module classes - use EventBaseModule
@dataclass
class Projection(EventBaseModule[ProjectionFunctions]):
    """Event-driven projection module"""
    Functions = ProjectionFunctions  # API compatibility alias
    generator: Callable[[int, int], ProjectionFunctions]
    min_delay: float
    n_steps: int = 0


@dataclass
class Population(
    BasePopulation,
    EventBaseModule[PopulationFunctions],
):
    """Event-driven population module"""
    Functions = PopulationFunctions  # API compatibility alias
    generator: Callable[
        [
            List[str],
            List[str],
            Dict[str, float],
            Dict[str, int],
            float,
            str,
            List[str],
            str,
        ],
        PopulationFunctions
    ]
    size: int
    n_steps: int


@dataclass
class SourcePopulation(
    BasePopulation,
    EventBaseModule[SourcePopulationFunctions],
):
    """Event-driven source population module"""
    Functions = SourcePopulationFunctions  # API compatibility alias
    generator: Callable
    size: int
    n_steps: int = 0


HXSpikes = Dict[str, Spike]

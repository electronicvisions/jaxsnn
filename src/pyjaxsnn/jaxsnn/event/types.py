# pylint: disable=disallowed-name,invalid-name,unsubscriptable-object,
from __future__ import annotations

import dataclasses
from typing import Callable, Generic, List, NamedTuple, Tuple, TypeVar, Union

import jax
import optax
import tree_math


@dataclasses.dataclass
@tree_math.struct
class LIFState:
    """State of a LIF neuron

    Parameters:
        V (jax.Array): membrane potential
        I (jax.Array): synaptic input current
    """

    V: jax.Array
    I: jax.Array


@dataclasses.dataclass
@tree_math.struct
class Spike:
    time: jax.Array
    idx: jax.Array

    @property
    def shape(self):
        return self.time.shape  # pylint: disable=no-member

    def __getitem__(self, key) -> Spike:
        return Spike(
            self.time[key],
            self.idx[key],
        )


@dataclasses.dataclass
@tree_math.struct
class EventPropSpike:
    time: jax.Array
    idx: jax.Array
    current: jax.Array

    @property
    def shape(self):
        return self.time.shape  # pylint: disable=no-member

    def __getitem__(self, key) -> EventPropSpike:
        return EventPropSpike(
            self.time[key],
            self.idx[key],
            self.current[key],
        )


@dataclasses.dataclass
@tree_math.struct
class InputQueue:
    spikes: EventPropSpike
    head: int = 0

    @property
    def is_empty(self) -> bool:
        return self.head >= self.spikes.time.size  # pylint: disable=no-member

    def peek(self) -> EventPropSpike:
        return self.spikes[self.head]

    def pop(self) -> EventPropSpike:
        spike = self.spikes[self.head]
        self.head += 1
        return spike

    def next_time_or_default(self, default):
        return jax.lax.cond(
            self.is_empty, lambda: default, lambda: self.peek().time
        )


State = TypeVar("State")


@dataclasses.dataclass
@tree_math.struct
class StepState(Generic[State]):
    neuron_state: State
    time: float
    input_queue: InputQueue


class WeightInput(NamedTuple):
    input: jax.Array


class WeightRecurrent(NamedTuple):
    input: jax.Array
    recurrent: jax.Array


Weight = Union[WeightInput, WeightRecurrent]
SingleInit = Callable[[jax.random.KeyArray, int], Tuple[int, Weight]]
SingleApply = Callable[[int, Weight, EventPropSpike], EventPropSpike]
SingleInitApply = Tuple[SingleInit, SingleApply]

Init = Callable[[jax.random.KeyArray, int], List[Weight]]
Apply = Callable[[List[Weight], EventPropSpike], List[EventPropSpike]]
InitApply = Tuple[Init, Apply]

# for hardware data
SingleApplyHW = Callable[[int, Weight, EventPropSpike, Spike], EventPropSpike]
SingleInitApplyHW = Tuple[SingleInit, SingleApplyHW]
ApplyHW = Callable[
    [List[EventPropSpike], List[Weight], EventPropSpike], List[EventPropSpike]
]
InitApplyHW = Tuple[Init, ApplyHW]


class OptState(NamedTuple):
    opt_state: optax.OptState
    weights: List[Weight]


# define the interface that a root solver has
# take the LIFState, current time and t_mxa and return a Spike
Solver = Callable[[LIFState, float, float], Spike]


class TestResult(NamedTuple):
    loss: float
    accuracy: float
    t_first_spike: jax.Array
    recording: jax.Array


# loss function return loss and some recording
LossAndRecording = Tuple[float, Tuple[jax.Array, List[EventPropSpike]]]
LossFn = Callable[
    [List[Weight], Tuple[EventPropSpike, jax.Array]], LossAndRecording
]


# when working with hw, we also have the known spikes as input
HWLossFn = Callable[
    [List[Weight], Tuple[EventPropSpike, jax.Array], List[EventPropSpike]],
    LossAndRecording,
]

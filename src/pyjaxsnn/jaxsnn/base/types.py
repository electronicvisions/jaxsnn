from __future__ import annotations

import dataclasses
from typing import Generic, Type, TypeVar, Union, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import tree_math

PyTreeState = TypeVar("PyTreeState")
ArrayLike = Union[jnp.ndarray, np.ndarray, float]
Array = Union[jnp.ndarray, np.ndarray]
JaxArray = Type[jnp.ndarray]


@dataclasses.dataclass
@tree_math.struct
class Spike:
    time: JaxArray
    idx: JaxArray

    @property
    def shape(self):
        return self.time.shape

    def __getitem__(self, key) -> Spike:
        return Spike(self.time[key], self.idx[key])


@dataclasses.dataclass
@tree_math.struct
class EventPropSpike:
    time: JaxArray
    idx: JaxArray
    current: JaxArray

    @property
    def shape(self):
        return self.time.shape

    def __getitem__(self, key) -> EventPropSpike:
        return EventPropSpike(self.time[key], self.idx[key], self.current[key])


@dataclasses.dataclass
@tree_math.struct
class InputQueue:
    spikes: EventPropSpike
    head: int = 0

    @property
    def is_empty(self) -> bool:
        return self.head >= self.spikes.time.size

    def peek(self) -> EventPropSpike:
        return self.spikes[self.head]

    def pop(self) -> EventPropSpike:
        spike = self.spikes[self.head]
        self.head += 1
        return spike

    def next_time_or_default(self, default):
        return jax.lax.cond(self.is_empty, lambda: default, lambda: self.peek().time)


State = TypeVar("State")


@dataclasses.dataclass
@tree_math.struct
class StepState(Generic[State]):
    neuron_state: State
    time: float
    input_queue: InputQueue


class WeightInput(NamedTuple):
    input: JaxArray


class WeightRecurrent(NamedTuple):
    input: JaxArray
    recurrent: JaxArray


Weight = Union[WeightInput, WeightRecurrent]

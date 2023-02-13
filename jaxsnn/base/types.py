# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle
from __future__ import annotations

import dataclasses
from typing import Generic, Tuple, Type, TypeVar, Union

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
    time: Array
    idx: Array

    @property
    def shape(self):
        return self.time.shape

    def __getitem__(self, key) -> Spike:
        return Spike(self.time[key], self.idx[key])


@dataclasses.dataclass
@tree_math.struct
class InputQueue:
    spikes: Spike
    head: int = 0

    @property
    def is_empty(self) -> bool:
        return self.head == len(self.spikes.time)

    def peek(self) -> Spike:
        return self.spikes[self.head]

    def pop(self) -> Spike:
        spike = self.spikes[self.head]
        self.head += 1
        return spike


State = TypeVar("State")


@dataclasses.dataclass
@tree_math.struct
class StepState(Generic[State]):
    neuron_state: State
    time: float
    input_queue: InputQueue


Weight = Union[Tuple[Array, Array], Array]

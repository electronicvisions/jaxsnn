# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

import jax.numpy as jnp
import numpy as np

from typing import TypeVar, Union, Tuple, Generic, Type
import tree_math
import dataclasses

PyTreeState = TypeVar("PyTreeState")
ArrayLike = Union[jnp.ndarray, np.ndarray, float]
Array = Union[jnp.ndarray, np.ndarray]
JaxArray = Type[jnp.ndarray]

# spikes should be ordered in time
@dataclasses.dataclass
@tree_math.struct
class Spike:
    time: Array
    idx: Array


State = TypeVar("State")


@dataclasses.dataclass
@tree_math.struct
class StepState(Generic[State]):
    neuron_state: State
    time: float
    input_spikes: Spike
    running_idx: int


Weight = Union[Tuple[Array, Array], Array]

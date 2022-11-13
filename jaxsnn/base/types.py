# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

import jax.numpy as jnp
import numpy as np

from typing import TypeVar, Union, Tuple, Generic
import tree_math

PyTreeState = TypeVar("PyTreeState")
ArrayLike = Union[jnp.ndarray, np.ndarray, float]

@tree_math.struct
class Spike:
    time: ArrayLike
    idx: ArrayLike

State = TypeVar('State')

@tree_math.struct
class StepState(Generic[State]):
    state: State
    time: float
    running_idx: int


Weight = Union[Tuple[ArrayLike, ArrayLike], ArrayLike]
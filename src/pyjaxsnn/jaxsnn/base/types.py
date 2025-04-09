from __future__ import annotations

import dataclasses
from typing import Union

import jax
import numpy as np
import tree_math

ArrayLike = Union[jax.Array, np.ndarray, float]
Array = Union[jax.Array, np.ndarray]


# pylint: disable=invalid-name, disallowed-name
@dataclasses.dataclass
@tree_math.struct
class LIState:
    """State of a leaky-integrate neuron.

    Parameters:
        V (jax.Array): membrane voltage
        I (jax.Array): input current
    """

    V: jax.Array
    I: jax.Array


LIFState = LIState

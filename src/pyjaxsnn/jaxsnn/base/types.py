from __future__ import annotations

from typing import Union

import jax
import numpy as np

ArrayLike = Union[jax.Array, np.ndarray, float]
Array = Union[jax.Array, np.ndarray]

# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

import jax.numpy as jnp
import numpy as np

from typing import TypeVar, Union

PyTreeState = TypeVar("PyTreeState")
ArrayLike = Union[jnp.ndarray, np.ndarray, float]

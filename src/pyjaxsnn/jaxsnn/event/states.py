from dataclasses import dataclass
import jax

from jaxsnn.base.types import (
    BaseState,
    _tm_struct,
)


@dataclass
@_tm_struct
# pylint: disable=invalid-name, disallowed-name
class LIFState(BaseState):
    V: jax.Array  # Membrane potential
    I: jax.Array  # Input current

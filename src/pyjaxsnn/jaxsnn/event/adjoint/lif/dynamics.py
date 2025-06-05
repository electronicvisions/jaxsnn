from typing import Callable, Union
import jax
import jax.numpy as jnp

from jaxsnn.event.functional.lif.dynamics import exponential_flow
from jaxsnn.event.states import LIFState


def adjoint_lif_exponential_flow(
    tau_syn: Union[jax.Array, float],
    tau_mem: Union[jax.Array, float],
) -> Callable[[LIFState, jax.Array], LIFState]:
    A = jnp.array(
        [[- 1. / tau_mem, 0.0],
         [1. / tau_syn, -1. / tau_syn]])
    return exponential_flow(A)

# pylint: disable=invalid-name
from typing import (
    Callable,
    Tuple,
    Dict,
)

import jax
import jax.numpy as jnp
from jaxsnn.discrete.types import (
    DenseData,
    Population,
    Parameter,
)
from jaxsnn.discrete.functional.threshold import superspike
from jaxsnn.discrete.functional.lif import (
    LIFState,
    lif_step,
    LIFParameters,
)


def LIF(
    size,
    params: LIFParameters = LIFParameters(),
    method: Callable = superspike,
) -> Population:
    """
    Layer constructor function for a leaky-integrate and fire layer.

    :param size: Number of neurons in the layer.
    :param params: Parameters for the LIF neuron model.
    :param method: Surrogate gradient method for the threshold function.

    :return: A Population object containing the layer definition.
    """

    def generator(dt: float) -> Population.Functions:
        """
        Generates the runtime functions for the LIF layer.

        :param dt: Simulation time step.

        :return: A Population.Functions object holding the collection of
            functions (init, state, step).
        """

        # Dummy init for when neuron parameters are also initialized
        def init_fn(
            rng: jax.Array,  # pylint: disable=unused-argument
        ) -> None:
            return None

        def state_fn() -> Tuple[LIFState, DenseData]:
            return LIFState(jnp.zeros(size), jnp.zeros(size)), jnp.zeros(size)

        def step_fn(
            inputs: Dict[str, DenseData],
            state: LIFState,
            parameters: Parameter
        ) -> Tuple[LIFState, DenseData]:
            return lif_step(
                inputs=inputs,
                state=state,  # Generic function preserves the exact type
                parameters=parameters,
                method=method,
                v_leak=params.v_leak,
                tau_mem=params.tau_mem,
                tau_syn=params.tau_syn,
                v_th=params.v_th,
                v_reset=params.v_reset,
                dt=dt,
            )

        return Population.Functions(init_fn, state_fn, step_fn)

    parameters = {
        "size": size,
        "lif_params": params,
    }

    return Population(generator, parameters, size)

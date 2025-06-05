from typing import Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jaxsnn.discrete.types import (
    DenseData,
    Population,
    Parameter,
)
from jaxsnn.discrete.functional.li import (
    LIParameters,
    li_step,
    LIState,
)


def LI(  # pylint: disable=invalid-name
    size: int,
    params: LIParameters = LIParameters(),
) -> Population:
    """
    Layer constructor function for a leaky-integrator layer.

    :param size: Number of neurons in the layer.
    :param params: Parameters for the LI neuron model.

    :return: A Population object containing the layer definition.
    """

    def generator(
        dt: float,  # pylint: disable=invalid-name
    ) -> Population.Functions:
        """
        Generates the runtime functions for the LI layer.

        :param dt: Simulation time step.

        :return: A Population.Functions object holding the collection of
            functions (init, state, step).
        """

        def init_fn(
            rng: jax.Array,  # pylint: disable=unused-argument
        ) -> Optional[Parameter]:
            return None

        def state_fn() -> Tuple[LIState, DenseData]:
            return LIState(jnp.zeros(size), jnp.zeros(size)), jnp.zeros(size)

        def step_fn(
            inputs: Dict[str, DenseData],
            state: LIState,
            parameters: Parameter
        ) -> Tuple[LIState, DenseData]:
            return li_step(
                inputs,
                state,
                parameters,
                v_leak=params.v_leak,
                tau_mem=params.tau_mem,
                tau_syn=params.tau_syn,
                dt=dt,
            )

        return Population.Functions(init_fn, state_fn, step_fn)

    parameters = {
        "size": size,
        "li_params": params,
    }

    return Population(generator, parameters, size)

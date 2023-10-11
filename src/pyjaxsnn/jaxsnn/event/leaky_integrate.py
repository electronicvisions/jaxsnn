from functools import partial

import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.discrete.threshold import heaviside
from jaxsnn.event.types import LIFState, Spike


def kernel_fn(kernel, time, spike_time):
    return heaviside(time - spike_time) * jax.scipy.linalg.expm(
        kernel * (time - spike_time)
    )


def superposition(kernel, spike_time, initial_state, time):
    return np.einsum(
        "ijk, ik -> j",
        jax.vmap(partial(kernel_fn, kernel, time))(spike_time),
        initial_state,
    )


def li_cell(
    kernel: jax.Array, time_steps: jax.Array, weights: jax.Array, spikes: Spike
) -> LIFState:
    # don't integrate over inf spike times
    first_inf = np.searchsorted(spikes.time, 1_000_000, side="right")
    spikes = Spike(spikes.time[:first_inf], spikes.idx[:first_inf])

    current = weights[spikes.idx] * np.where(spikes.idx == -1, 0.0, 1.0)
    voltage = np.zeros(len(spikes.idx))
    initial_state = np.stack((voltage, current), axis=1)
    final_state = jax.vmap(
        partial(superposition, kernel, spikes.time, initial_state)
    )(time_steps)
    return LIFState(V=final_state[:, 0], I=final_state[:, 1])


# multiple neurons
leaky_integrator = jax.vmap(li_cell, in_axes=(None, None, 1, None), out_axes=1)


def LeakyIntegrator(  # pylint: disable=invalid-name
    n_hidden: int,
    t_max: float,
    params: LIFParameters,
    mean: float = 0.5,
    std: float = 2.0,
    time_steps: int = 20,
):  # pylint: disable=too-many-arguments
    def init_fn(rng: jax.random.KeyArray, input_shape: int):
        return (
            n_hidden,
            jax.random.normal(rng, (input_shape, n_hidden)) * std + mean,
        )

    kernel = np.array(
        [[-params.tau_mem_inv, params.tau_mem_inv], [0, -params.tau_syn_inv]]
    )
    return init_fn, partial(
        leaky_integrator, kernel, np.linspace(0, t_max, time_steps)
    )

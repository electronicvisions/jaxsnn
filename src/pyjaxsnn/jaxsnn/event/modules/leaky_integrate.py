from functools import partial

import jax
import jax.numpy as jnp
from jaxsnn.base.params import LIFParameters
from jaxsnn.discrete.functional.threshold import heaviside
from jaxsnn.event.types import LIFState, Spike


def kernel_fn(kernel, time, spike_time):
    return heaviside(time - spike_time) * jax.scipy.linalg.expm(
        kernel * (time - spike_time)
    )


def superposition(kernel, spike_time, initial_state, time):
    return jnp.einsum(
        "ijk, ik -> j",
        jax.vmap(partial(kernel_fn, kernel, time))(spike_time),
        initial_state,
    )


def li_cell(
    kernel: jax.Array, time_steps: jax.Array, weights: jax.Array, spikes: Spike
) -> LIFState:
    # don't integrate over inf spike times
    first_inf = jnp.searchsorted(spikes.time, 1_000_000, side="right")
    spikes = Spike(spikes.time[:first_inf], spikes.idx[:first_inf])

    current = weights[spikes.idx] * jnp.where(spikes.idx == -1, 0.0, 1.0)
    voltage = jnp.zeros(len(spikes.idx))
    initial_state = jnp.stack((voltage, current), axis=1)
    final_state = jax.vmap(
        partial(superposition, kernel, spikes.time, initial_state)
    )(time_steps)
    return LIFState(V=final_state[:, 0], I=final_state[:, 1])


# multiple neurons
leaky_integrator = jax.vmap(li_cell, in_axes=(None, None, 1, None), out_axes=1)


def LeakyIntegrator(  # pylint: disable=invalid-name
    size: int,
    t_max: float,
    params: LIFParameters,
    mean: float = 0.5,
    std: float = 2.0,
    time_steps: int = 20,
):  # pylint: disable=too-many-arguments
    def init_fn(rng: jax.Array, input_shape: int):
        rng, layer_rng = jax.random.split(rng)
        return (
            rng,
            size,
            jax.random.normal(layer_rng, (input_shape, size)) * std + mean
        )

    kernel = jnp.array(
        [[-1. / params.tau_mem, 1. / params.tau_mem],
         [0, -1. / params.tau_syn]]
    )
    return init_fn, partial(
        leaky_integrator, kernel, jnp.linspace(0, t_max, time_steps)
    )

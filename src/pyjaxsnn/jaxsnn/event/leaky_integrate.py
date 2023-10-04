from functools import partial

import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.discrete.threshold import heaviside
from jaxsnn.event.types import LIFState, Spike


def kernel(A, t, t0):
    return heaviside(t - t0) * jax.scipy.linalg.expm(A * (t - t0))  # type: ignore


def f(A, t0, x0, t):
    return np.einsum("ijk, ik -> j", jax.vmap(partial(kernel, A, t))(t0), x0)


def li_cell(A: jax.Array, ts: jax.Array, weights: jax.Array, spikes: Spike) -> LIFState:
    # don't integrate over inf spike times
    first_inf = np.searchsorted(spikes.time, 1_000_000, side="right")
    spikes = Spike(spikes.time[:first_inf], spikes.idx[:first_inf])

    current = weights[spikes.idx] * np.where(spikes.idx == -1, 0.0, 1.0)
    voltage = np.zeros(len(spikes.idx))
    xk = np.stack((voltage, current), axis=1)
    ys = jax.vmap(partial(f, A, spikes.time, xk))(ts)
    return LIFState(V=ys[:, 0], I=ys[:, 1])


# multiple neurons
leaky_integrator = jax.vmap(li_cell, in_axes=(None, None, 1, None), out_axes=1)


def LeakyIntegrator(
    n_hidden: int,
    t_max: float,
    p: LIFParameters,
    mean: float = 0.5,
    std: float = 2.0,
    time_steps: int = 20,
):
    def init_fn(rng: jax.random.KeyArray, input_shape: int):
        return n_hidden, jax.random.normal(rng, (input_shape, n_hidden)) * std + mean

    ts = np.linspace(0, t_max, time_steps)
    A = np.array([[-p.tau_mem_inv, p.tau_mem_inv], [0, -p.tau_syn_inv]])
    return init_fn, partial(leaky_integrator, A, ts)

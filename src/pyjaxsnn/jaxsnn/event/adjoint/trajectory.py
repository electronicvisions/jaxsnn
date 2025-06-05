from typing import (
    Callable,
    Tuple,
)

import jax
import jax.numpy as jnp


def adjoint_trajectory(
    multi_layer_adjoint_step_fn: Callable,
    n_steps: int,
    res: Tuple,
    g: Tuple,
) -> Tuple:
    """
    Run the adjoint trajectory backward through time using JAX's scan.

    :param multi_layer_adjoint_step_fn: Function performing one adjoint step
        for multiple layers.
    :param n_steps: Number of time steps to run backward.
    :param res: Tuple containing forward pass results (spikes, states
        weights, queue indices).
    :param g: Tuple containing adjoint initial states (adjoint spikes,
        adjoint state, gradients).

    :returns: Tuple of (gradients, adjoint spikes, adjoint states, None).
    """
    carry = (
        res[2],  # weights
        res[0],  # spikes
        res[3],  # queue indices
        g[0],    # adjoint spikes
        g[1],    # adjoint state
        g[2],    # grads
    )

    # TODO: Check if adjoint_spikes.time == 0 becuase add_grad uses "add"
    carry, _ = jax.lax.scan(
        multi_layer_adjoint_step_fn,
        carry,
        jnp.arange(n_steps - 1, -1, -1)
    )

    adjoint_spikes = carry[3]
    adjoint_states = carry[4]
    grads = carry[5]

    # TODO: Check if reverting and flipping is needed as in
    # https://gerrit.bioai.eu:9443/c/jax-snn/+/25195/33/src/pyjaxsnn/jaxsnn/event/adjoint_lif.py#b348
    # Probably not need to flip because we run the scan backwards already but
    # maybe we need to roll

    return grads, adjoint_spikes, None, adjoint_states, None

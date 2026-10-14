from typing import (
    Callable,
    Tuple,
)

import jax
import jax.numpy as jnp
from jaxsnn.event.types import AdjointTrajectoryCarry


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
    carry = AdjointTrajectoryCarry(
        parameters=res[2],
        observables=res[0],
        queue_indices=res[3],
        adjoint_observables=g[0],
        adjoint_states=g[1],
        parameters_grads=g[2],
    )

    # TODO: Check if adjoint_spikes.time == 0 becuase add_grad uses "add"
    carry, _ = jax.lax.scan(
        multi_layer_adjoint_step_fn,
        carry,
        jnp.arange(n_steps - 1, -1, -1)
    )

    # TODO: Check if reverting and flipping is needed as in
    # https://gerrit.bioai.eu:9443/c/jax-snn/+/25195/33/src/pyjaxsnn/jaxsnn/event/adjoint_lif.py#b348
    # Probably not need to flip because we run the scan backwards already but
    # maybe we need to roll

    return (
        carry.parameters_grads,
        carry.adjoint_observables,
        None,
        carry.adjoint_states,
        None,
    )

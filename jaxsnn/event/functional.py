from typing import Callable, Tuple

import jax
import jax.numpy as np

from jaxsnn.types import Array


def f(A, x0, t):
    return np.dot(jax.scipy.linalg.expm(A * t), x0)  # type: ignore


def jump_condition(dynamics, v_th, x0, t):
    return dynamics(x0, t)[0] - v_th  # this implements the P y(t) - b above


def step(
    dynamics: Callable,
    solver: Callable,
    weights: Tuple[Array, Array],
    input_spikes: Array,
    y: Array,
    dt: float,
) -> Tuple[Array, float, int]:
    """Determine the next spike (input or internal), and integrate the neurons to that point.

    Args:
        dynamics (Callable): Function describing neuron dynamics
        solver (Callable): Parallel root solver
        weights (Tuple[Array, Array]): Input and recurrent weights
        input_spikes (Array): Input spike times per neuron, inf if no spike for a neuron
        y (Array): Current state (voltage, current) of the neurons with shape (n_neurons, 2)
        dt (float): Maximum time step if no spike

    Returns:
       Tuple: (State after transition, time of transition, index of spike or -1 if external / no spike)
    """
    input_weights, recurrent_weights = weights
    t_spike = solver(y, 1e-3)

    t_spike = np.where(np.isnan(t_spike), np.inf, t_spike)
    spike_idx = np.argmin(t_spike)
    spike_time = t_spike[spike_idx]

    # only regard future input spikes
    input_spikes = np.where(input_spikes > 0.0, input_spikes, np.inf)
    input_spike_idx = np.argmin(input_spikes)
    input_spike_time = input_spikes[input_spike_idx]

    # determine if we have a recurrent spike
    recurrent_spike = spike_time < input_spike_time

    # integrate
    t_dyn = np.minimum(np.minimum(spike_time, input_spike_time), dt)
    y_minus = dynamics(y, t_dyn)

    # reset
    y_minus = jax.lax.cond(
        recurrent_spike, lambda: y_minus.at[spike_idx, 0].set(0.0), lambda: y_minus
    )

    # transistion
    tr_row = jax.lax.cond(
        recurrent_spike,
        lambda: recurrent_weights[spike_idx],
        lambda: input_weights[input_spike_idx],
    )
    y_plus = jax.lax.cond(
        t_dyn == dt,
        lambda: y_minus,
        lambda: y_minus.at[:, 1].set(y_minus[:, 1] + tr_row),
    )

    stored_idx = jax.lax.cond(recurrent_spike, lambda: spike_idx, lambda: -1)
    return y_plus, t_dyn, stored_idx


def forward_integration(
    step_fn: Callable,
    n_spikes: int,
    weights: Tuple[Array, Array],
    input_spikes: Array,
    t_max,
):
    """Move neurons forward for n spikes

    Args:
        step_fn (Callable): Single step function
        n_spikes (int): Number of spikes
         weights (Tuple[Array, Array]): Input and recurrent weights
        input_spikes (Array): Input spike times per neuron, inf if no spike for a neuron
        t_max (float): Maximum time until which the net is moved forward
    """

    def body(state, it):
        t, y = state  # t is current lower bound

        dt = t_max - t
        y_plus, dt_dyn, spike_idx = step_fn(weights, input_spikes - t, y, dt)

        t = t + dt_dyn
        state = (t, y_plus)
        return state, (t, spike_idx)

    t = 0
    initial_state = np.zeros((weights[1].shape[0], 2))
    return jax.lax.scan(body, (t, initial_state), np.arange(n_spikes))

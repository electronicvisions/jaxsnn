# pylint: disable=invalid-name
from functools import partial
from typing import Tuple

import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.discrete.threshold import superspike


@partial(jax.jit, static_argnames=["k"])
def one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def lif_current_encoder(
    voltage,
    input_current: jax.Array,
    params: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[jax.Array, jax.Array]:
    r"""Computes a single euler-integration step of a leaky integrator. More
    specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    Parameters:
        input (jax.Array): the input current at the current time step
        voltage (jax.Array): current state of the LIF neuron
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    dv = dt * params.tau_mem_inv * ((params.v_leak - voltage) + input_current)
    voltage = voltage + dv
    z = superspike(voltage - params.v_th)

    voltage = voltage - z * (voltage - params.v_reset)
    return voltage, z


def constant_current_lif_encode(
    input_current: jax.Array,
    seq_length: int,
):
    """
    Encodes input currents as fixed (constant) voltage currents, and simulates
    the spikes that occur during a number of timesteps/iterations (seq_length).

    Example:
        >>> data = np.array([2, 4, 8, 16])
        >>> seq_length = 2 # Simulate two iterations
        >>> constant_current_lif_encode(data, seq_length)
         # State in terms of membrane voltage
        (DeviceArray([[0.2000, 0.4000, 0.8000, 0.0000],
                 [0.3800, 0.7600, 0.0000, 0.0000]]),
         # Spikes for each iteration
         DeviceArray([[0., 0., 0., 1.],
                 [0., 0., 1., 1.]]))

    Parameters:
        input_current (jax.Array): The input array, representing LIF current
        seq_length (int): The number of iterations to simulate

    Returns:
        An array with an extra dimension of size `seq_length` containing
        spikes (1) or no spikes (0).
    """
    init = np.zeros(*input_current.shape)
    input_current = np.tile(input_current, (seq_length, 1))
    return jax.lax.scan(lif_current_encoder, init, input_current)


@partial(jax.jit, static_argnames=["seq_length"])
def spatio_temporal_encode_inner(
    input_values: jax.Array,
    seq_length: int,
    t_late: float,
    dt: float,
):
    """
    Encodes n-dimensional input coordinates with range [0, 1], and simulates
    the spikes that occur during a number of timesteps/iterations (seq_length).

    Example:
        >>> data = np.array([2, 4, 8, 16])
        >>> seq_length = 2 # Simulate two iterations
        >>> spatio_temporal_encode(data, seq_length)
         # Spikes for each iteration
         DeviceArray([[0., 0., 0., 1.],
                 [0., 0., 1., 1.]]))

    Parameters:
        input_values (torch.Tensor): The input tensor, representing 2d points
        seq_length (int): The number of iterations to simulate
        t_early (float): Earliest time at which coordinates may be encoded
        t_late (float): Latest time at which coordinates may be encoded
        dt (float): Time delta between simulation steps

    Returns:
        A tensor with an extra dimension of size `seq_length` containing
        spikes (1) or no spikes (0).
    """
    if len(input_values.shape) > 2:
        raise ValueError(
            "Tensor with input values must be one or two dimensional"
        )

    idx = (input_values * t_late / dt).round().astype(int)
    idx = np.clip(idx, 0, seq_length)
    encoded = np.eye(seq_length)[:, idx]
    return encoded


def spatio_temporal_encode(T, t_late, DT):
    def init_fn(rng, input_shape):
        return (input_shape, None, rng)

    def apply_fn(params, inputs, **kwargs):  # pylint: disable=unused-argument
        return spatio_temporal_encode_inner(inputs, T, t_late, DT), None

    return init_fn, apply_fn

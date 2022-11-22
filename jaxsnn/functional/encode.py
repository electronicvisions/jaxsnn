from functools import partial
from typing import Tuple

import jax.numpy as np
from jax import jit
from jax.lax import scan

from jaxsnn.functional.leaky_integrate_and_fire import LIFParameters
from jaxsnn.functional.threshold import superspike
from jaxsnn.base.types import Array


@partial(jit, static_argnames=["k"])
def one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def lif_current_encoder(
    voltage,
    input_current: Array,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[Array, Array]:
    r"""Computes a single euler-integration step of a leaky integrator. More
    specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    Parameters:
        input (Array): the input current at the current time step
        voltage (Array): current state of the LIF neuron
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    dv = dt * p.tau_mem_inv * ((p.v_leak - voltage) + input_current)
    voltage = voltage + dv
    z = superspike(voltage - p.v_th)

    voltage = voltage - z * (voltage - p.v_reset)
    return voltage, z


def constant_current_lif_encode(
    input_current,
    seq_length: int,
):
    """
    Encodes input currents as fixed (constant) voltage currents, and simulates the spikes that
    occur during a number of timesteps/iterations (seq_length).

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
        input_current (Array): The input array, representing LIF current
        seq_length (int): The number of iterations to simulate

    Returns:
        An array with an extra dimension of size `seq_length` containing spikes (1) or no spikes (0).
    """
    init = np.zeros(*input_current.shape)
    input_current = np.tile(input_current, (seq_length, 1))
    return scan(lif_current_encoder, init, input_current)


@partial(jit, static_argnames=["seq_length"])
def spatio_temporal_encode(
    input_values,
    seq_length: int,
    t_late: float,
    dt: float,
):
    """
    Encodes n-dimensional input coordinates with range [0, 1], and simulates the spikes that
    occur during a number of timesteps/iterations (seq_length).

    Example:
        >>> data = np.array([2, 4, 8, 16])
        >>> seq_length = 2 # Simulate two iterations
        >>> spatio_temporal_encode(data, seq_length)
         # Spikes for each iteration
         DeviceArray([[0., 0., 0., 1.],
                 [0., 0., 1., 1.]]))

    Parameters:
        input_values (torch.Tensor): The input tensor, representing points in 2d space
        seq_length (int): The number of iterations to simulate
        t_early (float): Earliest time at which coordinates may be encoded
        t_late (float): Latest time at which coordinates may be encoded
        dt (float): Time delta between simulation steps

    Returns:
        A tensor with an extra dimension of size `seq_length` containing spikes (1) or no spikes (0).
    """
    if len(input_values.shape) > 2:
        raise ValueError("Tensor with input values must be one or two dimensional")

    idx = (input_values * t_late / dt).round().astype(int)
    idx = np.clip(idx, 0, seq_length)
    encoded = np.eye(seq_length)[:, idx]
    return encoded


def SpatioTemporalEncode(T, t_late, DT):
    def init_fn(rng, input_shape):
        return (input_shape, None, rng)

    def apply_fn(params, inputs, **kwargs):
        return spatio_temporal_encode(inputs, T, t_late, DT), None

    return init_fn, apply_fn


if __name__ == "__main__":
    # data = np.array([[0.1, 0.1, 0.9, 0.9], [0.5, 0.8, 0.5, 0.2]])
    # value = spatio_temporal_encode(data, 10)

    data = np.array([0.5, 0.8, 0.5, 0.2])

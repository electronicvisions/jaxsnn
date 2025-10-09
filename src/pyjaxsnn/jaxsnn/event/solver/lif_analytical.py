"""Analytically find the time of the next spike for a LIF neuron for the
special cases of $\tau_mem = \tau_syn$ and $\tau_mem = 2 * \tau_syn$.

When using `jax.vmap` to do this root solving in parallel, `jax.lax.cond`
is mapped to `jax.lax.switch`, meaning that both branches are executed.
Therefore, special care is taken to ensure that no NaNs occur, which would
affect gradient calculation."""
from functools import partial
from typing import Callable

from jaxsnn.event.solver.lif_analytical_single_time import ttfs_single_time
from jaxsnn.event.solver.lif_analytical_double_time import ttfs_double_time


def ttfs_solver(
    tau_mem: float,
    tau_syn: float,
    v_th: float,
    leak: float,
) -> Callable:
    """Find the next spike time for the special case tau_mem = tau_syn.

    :param tau_mem: Membrane time constant
    :param tau_syn: Synaptic time constant
    :param v_th: Threshold voltage
    :param leak: Leak term

    :return: Time of next threshold crossing or t_max if no crossing
    """
    if tau_mem == tau_syn:
        return partial(ttfs_single_time, tau_mem, tau_syn, v_th, leak)
    if tau_mem == 2 * tau_syn:
        return partial(ttfs_double_time, tau_mem, tau_syn, v_th, leak)
    raise ValueError("This solver only supports the special cases of "
                     "$\\tau_mem = \\tau_syn$ and $\\tau_mem = 2 * \\tau_syn$")

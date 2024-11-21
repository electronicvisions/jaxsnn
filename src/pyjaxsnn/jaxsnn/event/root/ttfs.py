"""Analytically find the time of the next spike for a LIF neuron for the
special cases of $\tau_mem = \tau_syn$ and $\tau_mem = 2 * \tau_syn$.

When using `jax.vmap` to do this root solving in parallel, `jax.lax.cond`
is mapped to `jax.lax.switch`, meaning that both branches are executed.
Therefore, special care is taken to ensure that no NaNs occur, which would
affect gradient calculation."""

from functools import partial

import jax
import jax.numpy as jnp
from jaxsnn.event.types import LIFState


# src of lambertw function: https://github.com/jax-ml/jax/issues/13680

def _real_lambertw_recursion(w: jax.Array, x: jax.Array) -> jax.Array:  # pylint: disable=invalid-name
    return w / (1 + w) * (1 + jnp.log(x / w))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _lambertwk0(x: float, max_steps: int = 5):  # pylint: disable=invalid-name
    # See https://en.wikipedia.org/wiki/Lambert_W_function#Numerical_evaluation
    w_0 = jax.lax.select(
        x > jnp.e,
        jnp.log(x) - jnp.log(jnp.log(x)),
        x / jnp.e
    )
    w_0 = jax.lax.select(
        x > 0,
        w_0,
        jnp.e * x / (1 + jnp.e * x + jnp.sqrt(1 + jnp.e * x)) * jnp.log(
            1 + jnp.sqrt(1 + jnp.e * x))
    )

    w, _ = jax.lax.scan(  # pylint: disable=invalid-name
        lambda carry, _: (_real_lambertw_recursion(carry, x),) * 2,
        w_0,
        xs=None, length=max_steps
    )

    w = jax.lax.select(  # pylint: disable=invalid-name
        jnp.isclose(x, 0.0),
        0.0,
        w
    )

    return w


@_lambertwk0.defjvp
def _lambertw_jvp(max_steps, primals, tangents):
    # Note: All branches for lambert W satisfy this JVP.
    x, = primals  # pylint: disable=invalid-name
    t, = tangents  # pylint: disable=invalid-name

    y = _lambertwk0(x, max_steps)  # pylint: disable=invalid-name
    dydx = 1 / (x + jnp.exp(y))

    jvp = jax.lax.select(
        jnp.isclose(x, -1 / jnp.e),
        jnp.nan,
        dydx * t
    )

    return y, jvp


@jnp.vectorize
def lambertw(x, k=0, max_steps=5):  # pylint: disable=invalid-name
    if k != 0:
        raise NotImplementedError()

    return _lambertwk0(x, max_steps=max_steps)


def ttfs_ratio1_inner_most(
    a_1: jax.Array,
    b: jax.Array,  # pylint: disable=invalid-name
    w_arg: jax.Array,
    tau_mem: float,
    t_max: float,
):
    has_spike = b / a_1 > lambertw(w_arg)
    return jax.lax.cond(
        has_spike,
        lambda: tau_mem * (b / a_1 - lambertw(w_arg)),
        lambda: t_max,
    )


def ttfs_ratio2_inner_most(
    a_1: jax.Array, denominator: jax.Array, tau_mem: float, t_max: float
) -> jax.Array:
    inner_log = 2 * a_1 / denominator
    return jax.lax.cond(
        inner_log > 1,
        lambda: tau_mem * jnp.log(jnp.maximum(inner_log, 1)),
        lambda: t_max,
    )


def ttfs_ratio1_inner(
    a_1: jax.Array,
    b: jax.Array,  # pylint: disable=invalid-name
    w_arg: jax.Array,
    tau_mem: float,
    t_max: float,
):
    has_spike = w_arg >= -1 / jnp.e
    return jax.lax.cond(
        has_spike,
        ttfs_ratio1_inner_most,
        lambda *args: t_max,
        a_1, b, w_arg, tau_mem, t_max
    )


def ttfs_ratio2_inner(
    a_1: jax.Array,
    a_2: jax.Array,
    second_term: jax.Array,
    tau_mem: float,
    t_max: float,
):
    epsilon = 1e-6
    denominator = a_2 + jnp.sqrt(jnp.maximum(second_term, epsilon))
    save_denominator = jnp.where(
        jnp.abs(denominator) > epsilon, denominator, epsilon
    )
    return jax.lax.cond(
        jnp.abs(denominator) > epsilon,
        ttfs_ratio2_inner_most,
        lambda *args: t_max,
        a_1,
        save_denominator,
        tau_mem,
        t_max,
    )


def ttfs_solver(tau_mem: float, tau_syn: float, v_th: float,
                state: LIFState, t_max: float):
    """Find the next spike time for special cases $\tau_mem = \tau_syn$ and
    $\tau_mem = 2 * \tau_syn$

    Args:
        tau_mem (float): Membrane time constant
        tau_syn (float): Synaptic time constant
        v_th (float): Threshold Voltage
        state (LIFState): State of the neuron (voltage, current)
        t_max (float): Maximum time which is to be searched

    Returns:
        float: Time of next threshold crossing or t_max if no crossing
    """
    v_0, i_0 = state.V, state.I
    a_1 = i_0
    a_2 = v_0 + i_0
    b = -v_0  # pylint: disable=invalid-name

    ratio = jnp.round(tau_mem / tau_syn).astype(int)

    def case_tau_mem_eq_tau_syn(a_1, b, tau_mem, t_max, v_th):  # pylint: disable=invalid-name
        w_arg = -v_th / a_1 * jnp.exp(b / a_1)
        has_spike = a_1 > 0
        return jax.lax.cond(
            has_spike,
            ttfs_ratio1_inner,
            lambda *args: t_max,
            a_1, b, w_arg, tau_mem, t_max
        )

    def case_tau_mem_eq_2tau_syn(a_1, a_2, tau_mem, t_max, v_th):
        second_term = a_2**2 - 4 * a_1 * v_th
        has_spike = second_term > 0
        return jax.lax.cond(
            has_spike,
            ttfs_ratio2_inner,
            lambda *args: t_max,
            a_1,
            a_2,
            second_term,
            tau_mem,
            t_max,
        )

    return jax.lax.switch(
        ratio - 1,
        [
            lambda: case_tau_mem_eq_tau_syn(a_1, b, tau_mem, t_max, v_th),
            lambda: case_tau_mem_eq_2tau_syn(a_1, a_2, tau_mem, t_max, v_th),
        ]
    )

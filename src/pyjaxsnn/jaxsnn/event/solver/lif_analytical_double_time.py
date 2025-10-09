from typing import Tuple

import jax
import jax.numpy as jnp
from jaxsnn.event.states import LIFState


def smallest_normal_like(x: jax.Array, mul: float = 1.0) -> jax.Array:  # pylint: disable=invalid-name
    return jnp.finfo(x.dtype).smallest_normal * mul


def safe_sqrt(x: jax.Array, eps: float | None = None) -> jax.Array:  # pylint: disable=invalid-name
    e = smallest_normal_like(x) if eps is None else eps  # pylint: disable=invalid-name
    return jnp.sqrt(jnp.maximum(x, e))


def safe_log(x: jax.Array, eps: float | None = None) -> jax.Array:  # pylint: disable=invalid-name
    e = smallest_normal_like(x) if eps is None else eps  # pylint: disable=invalid-name
    return jnp.log(jnp.clip(x, e, jnp.inf))


def ttfs_over_threshold(  # pylint: disable=unused-argument, too-many-arguments
    tau_mem: float,
    v_th: float,
    v_leak: float,
    state: LIFState,
    t_max: float,
) -> jax.Array:
    return jnp.array(0.0)


def ttfs_over_threshold_fwd(
    tau_mem: float,
    v_th: float,
    v_leak: float,
    state: LIFState,
    t_max: float,
) -> Tuple[jax.Array, Tuple]:
    """Forward pass for the custom VJP function."""
    time = ttfs_over_threshold(tau_mem, v_th, v_leak, state, t_max)
    return time, (state, tau_mem, v_th, v_leak)


def ttfs_over_threshold_bwd(res, g):  # pylint: disable=invalid-name
    """Backward pass for the custom VJP function."""
    state, tau_mem, v_th, v_leak = res
    # Vanishes iff dV/dt == 0 at the threshold, where dt/dV_0 truly diverges
    denominator = v_th - v_leak - state.I
    eps = smallest_normal_like(denominator)
    state.V = g * tau_mem / jnp.where(
        jnp.abs(denominator) > eps,
        denominator,
        jnp.sign(denominator + (denominator == 0)) * eps
    )
    state.I = jnp.zeros_like(state.I)
    return (None, None, None, state, None)


ttfs_over_threshold = jax.custom_vjp(ttfs_over_threshold)
ttfs_over_threshold.defvjp(ttfs_over_threshold_fwd, ttfs_over_threshold_bwd)


def ttfs_double_time_nonzero_current(  # pylint: disable=invalid-name, too-many-locals
    state: LIFState,
    tau_mem: float,
    t_max: float,
    v_th: float,
    v_leak: float,
) -> jax.Array:
    v_0, i_0 = state.V - v_leak, state.I
    a_1 = -i_0
    a_2 = v_0 + i_0

    v_diff = v_leak - v_th
    determinant = a_2**2 - 4 * a_1 * v_diff

    sqrt_det = safe_sqrt(determinant)
    eps = smallest_normal_like(sqrt_det)

    # a2 >= 0
    ratio_sum = - (2.0 * a_1) / jnp.where(
        jnp.abs(a_2 + sqrt_det) > eps,
        a_2 + sqrt_det,
        jnp.sign(a_2 + sqrt_det + (a_2 + sqrt_det == 0)) * eps,
    )
    # a2 < 0
    ratio_diff = (sqrt_det - a_2) / jnp.where(
        jnp.abs(2.0 * v_diff) > eps,
        2.0 * v_diff,
        jnp.sign(2.0 * v_diff + (v_diff == 0)) * eps,
    )

    ratio = jnp.where(a_2 >= 0, ratio_sum, ratio_diff)
    safe_time = tau_mem * safe_log(jnp.maximum(ratio, 1))

    has_spike = (
        (determinant >= smallest_normal_like(determinant))
        & (ratio > 1)
    )
    return jnp.where(has_spike, safe_time, t_max)


def ttfs_double_time_zero_current(
    state: LIFState,
    tau_mem: float,
    t_max: float,
    v_th: float,
    v_leak: float
) -> jax.Array:
    v_0 = v_leak - state.V
    diff = jnp.asarray(v_leak - v_th)
    safe_time = tau_mem * (safe_log(v_0) - safe_log(diff))
    has_spike = diff >= smallest_normal_like(diff)
    return jnp.where(has_spike, safe_time, t_max)


def ttfs_subthreshold(  # pylint: disable=too-many-arguments
    tau_mem: float,
    v_th: float,
    v_leak: float,
    state: LIFState,
    t_max: float,
    eps: float | None = None,
) -> jax.Array:
    current_eps = smallest_normal_like(state.I) if eps is None else eps
    return jax.lax.cond(
        jnp.abs(state.I) < current_eps,
        ttfs_double_time_zero_current,
        ttfs_double_time_nonzero_current,
        state,
        tau_mem,
        t_max,
        v_th,
        v_leak,
    )


def ttfs_double_time(  # pylint: disable=too-many-arguments
    tau_mem: float,
    tau_syn: float,  # pylint: disable=unused-argument
    v_th: float,
    v_leak: float,
    state: LIFState,
    t_max: float,
) -> jax.Array:

    return jax.lax.cond(
        # TODO: Maybe use small epsilon below the threshold
        state.V >= v_th,
        ttfs_over_threshold,
        ttfs_subthreshold,
        tau_mem,
        v_th,
        v_leak,
        state,
        t_max
    )

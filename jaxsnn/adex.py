from jaxsnn.tree_solver import ArrayLike

import jax.numpy as jnp
import tree_math

@tree_math.struct
class AdexParameters:
    g_l : ArrayLike
    g_exp : ArrayLike
    Delta_exp : ArrayLike
    tau_w_inv : ArrayLike
    a : ArrayLike
    V_l : ArrayLike
    V_exp : ArrayLike
    C_m_inv : ArrayLike
    tau_syn_inv : ArrayLike

@tree_math.struct
class AdexState:
    v : ArrayLike
    w : ArrayLike
    I : ArrayLike

def adex_dynamics(s : AdexState, p: AdexParameters):
    return AdexState(
        v = (p.g_l * p.C_m_inv) * (p.V_l - s.v) + (p.g_exp * p.C_m_inv) * p.Delta_exp * jnp.exp(1/p.Delta_exp * (s.v - p.V_exp)) - p.C_m_inv * s.w,
        w = -p.a * p.tau_w_inv * (p.V_l - s.v) - p.tau_w_inv * s.w,
    )

# Copyright 2022 Christian Pehle
#
# This is a port of the "Traub.mod" found in https://github.com/ModelDBRepository/254217/_mod/Traub.mod by Albert Gidon & Leora Menhaim (2004).
#
# Some original comments are reproduced below with the prefix ":". 
# Please refer to the original file for additional attributions.
#
# References:
#
# Gidon A, Zolnik TA, Fidzinski P, Bolduan F, Papoutsi A, Poirazi P, Holtkamp M, Vida I, Larkum ME (2020).
#   Dendritic action potentials and computation in human layer 2/3 cortical neurons 
#   Science 367:83-87
#
# Traub, R. D., Wong, R. K., Miles, R., and Michelson, H. (1991).
# 	A model of a CA3 hippocampal pyramidal neuron incorporating
# 	voltage-clamp data on intrinsic conductances.
# 	J Neurophysiol 66, 635-650.
#
# Kang, S., Kitano, K., and Fukai, T. (2004).
# 	Self-organized two-state membrane potential
# 	transitions in a network of realistically modeled
# 	cortical neurons. Neural Netw 17, 307-312.

import tree_math
import jax.numpy as jnp

from jaxsnn.channels.common import (
    channel_equilibrium,
    channel_time_constant,
    q_conversion,
)
from jaxsnn.base.types import ArrayLike


@tree_math.struct
class TraubParameters:
    g_Na: ArrayLike = 0.03  # (S/cm2)	:Traub et. al. 1991
    g_K: ArrayLike = 0.015  # (S/cm2) :Traub et. al. 1991
    g_L: ArrayLike = 0.00014  # (S/cm2) :Siu Kang - by email.
    e_L: ArrayLike = -62.0  # (mV)    :Siu Kang - by email.
    e_K: ArrayLike = -80  # (mV)    :Siu Kang - by email.
    e_Na: ArrayLike = 90  # (mV)    :Leora

# e_L = -74.0 		
# e_K = -80			
# e_Na = 90 	

# magic constants
_v_shift = 49.2  # shift to apply to all curves
_q_10 = 3
_target_temperature = 35
_original_temperature = 32

# temperature conversion
_q = q_conversion(_target_temperature, _original_temperature, _q_10)


@tree_math.struct
class TraubState:
    m: ArrayLike
    h: ArrayLike
    n: ArrayLike
    a: ArrayLike
    b: ArrayLike


def alpha_h(v: ArrayLike):
    return 0.128 * jnp.exp((17 - v) / 18)


def beta_h(v: ArrayLike):
    return 4 / (1 + jnp.exp((40 - v) / 5))


tau_h = channel_time_constant(alpha=alpha_h, beta=beta_h, v_shift=_v_shift, q=_q)
h_inf = channel_equilibrium(alpha=alpha_h, beta=beta_h)


def alpha_m(v):
    return jnp.where(
        v == 13.1, 0.32 * 4, 0.32 * (13.1 - v) / (jnp.exp((13.1 - v) / 4) - 1)
    )

def beta_m(v):
    return jnp.where(
        v == 40.1, 0.28 * 5, 0.28 * (v - 40.1) / (jnp.exp((v - 40.1) / 5) - 1)
    )


tau_m = channel_time_constant(alpha=alpha_m, beta=beta_m, v_shift=_v_shift, q=_q)
m_inf = channel_equilibrium(alpha=alpha_m, beta=beta_m, v_shift=_v_shift)


def alpha_n(v):
    return jnp.where(
        v == 35.1, 0.016 * 5, 0.016 * (35.1 - v) / (jnp.exp((35.1 - v) / 5) - 1)
    )


def beta_n(v):
    return 0.25 * jnp.exp((20 - v) / 40)


tau_n = channel_time_constant(alpha=alpha_n, beta=beta_n, v_shift=_v_shift, q=_q)
n_inf = channel_equilibrium(alpha=alpha_n, beta=beta_n, v_shift=_v_shift)


def g_Na(s: TraubState, p: TraubParameters):
    return p.g_Na * s.h * s.m * s.m


def I_Na(v: ArrayLike, s: TraubState, p: TraubParameters):
    return g_Na(s, p) * (v - p.e_Na)


def g_K(s: TraubState, p: TraubParameters):
    return p.g_K * s.n


def I_K(v: ArrayLike, s: TraubState, p: TraubParameters):
    return g_K(s, p) * (v - p.e_K)


def I_L(v: ArrayLike, s: TraubState, p: TraubParameters):
    return p.g_L * (v - p.e_L)


def dynamics(v: ArrayLike, s: TraubState, p: TraubParameters):
    return TraubState(
        m=(m_inf(v) - s.m) / tau_m(v),
        h=(h_inf(v) - s.h) / tau_h(v),
        n=(n_inf(v) - s.n) / tau_n(v),  # :phi=12 from Kang et. al. 2004
    )

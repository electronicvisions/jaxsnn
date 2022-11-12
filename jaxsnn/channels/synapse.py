# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle


# This is in part a port of "ISyn.inc" found in https://github.com/ModelDBRepository/254217/_mod/ISyn.inc
# and also incorporates the parameters found in https://github.com/ModelDBRepository/254217/Fig3andS9/syns.hoc
# Some of the original comments are preserved with prefix ":"

import jax.numpy as jnp
import tree_math

from jaxsnn.base.types import ArrayLike


@tree_math.struct
class SynapseState:
    a: ArrayLike
    b: ArrayLike


@tree_math.struct
class SynapseParameters:
    tau_r: ArrayLike
    tau_d: ArrayLike
    g_max: ArrayLike
    gamma: ArrayLike = 0.062
    e: ArrayLike = 0.0
    mg: ArrayLike = 0.0
    mg_dep: ArrayLike = 3.57


def weight_factor(tau_fast, tau_slow):
    t = (tau_fast * tau_slow) / (tau_slow - tau_fast) * jnp.log(tau_slow / tau_fast)
    e = -jnp.exp(-t / tau_fast) + jnp.exp(-t / tau_slow)
    return 1 / e


def dynamics(s: SynapseState, p: SynapseParameters):
    return SynapseState(a=-1 / p.tau_r * s.a, b=-1 / p.tau_d * s.b)


def transition(w: ArrayLike, s: SynapseState, p: SynapseParameters):
    return SynapseState(
        a=s.a + w * weight_factor(p.tau_r, p.tau_d),
        b=s.b + w * weight_factor(p.tau_r, p.tau_d),
    )


def g_syn(s: SynapseState, p: SynapseParameters):
    return p.g_max * (s.b - s.a)


def mg_dependence(v: ArrayLike, p: SynapseParameters):
    return 1 / (1 + jnp.exp(-p.gamma * v) * (p.mg / p.mg_dep))


def I_syn(v: ArrayLike, s: SynapseState, p: SynapseParameters):
    return g_syn(s) * mg_dependence(v, p) * (v - p.e)


# :  Synaptic parameters as in Eyal et al., (2018)
# : AMPA
ampa_parameters = SynapseParameters(g_max=0.7e-3, tau_r=0.3, tau_d=1.8, mg=0, e=0)  # uS

# : NMDA
nmda_parameters = SynapseParameters(
    g_max=1.3e-3,  # uS
    tau_r=8,
    tau_d=35,
    mg=1,
    e=0,
    gamma=0.077 / 0.082,  # rhodes et al., 2006
)

# : GABA
# : mid value between fast GABAa with 5 ms decay as in Salin and Prince, 1996 for
# : fast basket cells inhibition and slow martinotti GABAa with 20 ms decay (Gidon and Segev 2012)
basket_gaba_parameters = SynapseParameters(
    g_max=0.5e-3, tau_r=0.5, tau_d=5.0, mg=0, e=-75  # uS
)

martinotti_gaba_parameters = SynapseParameters(
    g_max=0.5e-3, tau_r=2.0, tau_d=23.0, mg=0.0, e=-75  # uS
)

# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

import jax
import jax.numpy as np


def heaviside(x):
    return 0.5 + 0.5 * np.sign(x)


@jax.custom_vjp
def superspike(x, alpha=80):
    r"""Surrogate gradient used in the 'Superspike' paper.

    References:

    F. Zenke, S. Ganguli, **"SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"**,
    Neural Computation 30, 1514-1541 (2018),
    `doi:10.1162/neco_a_01086 <https://www.mitpressjournals.org/doi/full/10.1162/neco_a_01086>`_
    """
    return heaviside(x)


def superspike_fwd(x, alpha):
    return heaviside(x), (x, alpha)


def superspike_bwd(res, g):
    (x, alpha) = res
    grad = g / (alpha * np.abs(x) + 1.0) ** 2
    return (grad, None)


superspike.defvjp(superspike_fwd, superspike_bwd)


@jax.custom_vjp
def triangular(x, alpha=0.3):
    r"""Triangular/piecewise linear surrogate / pseudo-derivative.

    References:

    S.K. Esser et al., **"Convolutional networks for fast, energy-efficient neuromorphic computing"**,
    Proceedings of the National Academy of Sciences 113(41), 11441-11446, (2016),
    `doi:10.1073/pnas.1604850113 <https://www.pnas.org/content/113/41/11441.short>`_

    G. Bellec et al., **"A solution to the learning dilemma for recurrent networks of spiking neurons"**,
    Nature Communications 11(1), 3625, (2020),
    `doi:10.1038/s41467-020-17236-y <https://www.nature.com/articles/s41467-020-17236-y>`_
    """
    return heaviside(x)


def triangular_fwd(x, alpha):
    return heaviside(x), (x, alpha)


def triangular_bwd(res, g):
    (x, alpha) = res
    grad = g * alpha * np.threshold(1.0 - np.abs(x), 0, 0)
    return (grad, None)


triangular.defvjp(triangular_fwd, triangular_bwd)

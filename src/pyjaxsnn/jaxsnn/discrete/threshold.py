# pylint: disable=invalid-name
import jax
import jax.numpy as np


def heaviside(x):
    return 0.5 + 0.5 * np.sign(x)


@jax.custom_vjp
def superspike(x, alpha=80):  # pylint: disable=unused-argument
    r"""Surrogate gradient used in the 'Superspike' paper.

    References:
    https://www.mitpressjournals.org/doi/full/10.1162/neco_a_01086
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
def triangular(x, alpha=0.3):  # pylint: disable=unused-argument
    r"""Triangular/piecewise linear surrogate / pseudo-derivative.

    References:

    https://www.pnas.org/content/113/41/11441.short

    https://www.nature.com/articles/s41467-020-17236-y
    """
    return heaviside(x)


def triangular_fwd(x, alpha):
    return heaviside(x), (x, alpha)


def triangular_bwd(res, g):
    (x, alpha) = res
    grad = g * alpha * np.threshold(1.0 - np.abs(x), 0, 0)
    return (grad, None)


triangular.defvjp(triangular_fwd, triangular_bwd)

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

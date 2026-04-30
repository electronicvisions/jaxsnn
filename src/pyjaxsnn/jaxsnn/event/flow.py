import jax
import jax.numpy as jnp
from jax.scipy import linalg
from jaxsnn.base.params import LIFParameters


def lif_wrap(func):
    def inner(*args):
        res = func(jnp.stack([args[0].V, args[0].I]), *args[1:])
        args[0].V = res[0]
        args[0].I = res[1]
        return args[0]

    return inner


def exponential_flow(kernel: jax.Array):
    def flow(initial_state: jax.Array, time: float):
        return jnp.dot(linalg.expm(kernel * time), initial_state)

    return lif_wrap(flow)


def lif_exponential_flow(params: LIFParameters):
    kernel = jnp.array(
        [[-1. / params.tau_mem, 1. / params.tau_mem],
         [0, -1. / params.tau_syn]])
    return exponential_flow(kernel)


def lif_exponential_flow_vec(params: LIFParameters):
    """Vec'd dynamics flow for per-neuron tau_mem and tau_syn.

    Each scalar field of `params` may be a scalar (broadcast) or an array of
    shape (N,). Returns a function `flow(state, time)` where `state` is a
    `LIFState` with `V` and `I` of shape (N,) and `time` is scalar. Internally
    builds a per-neuron [N, 2, 2] kernel and applies expm per neuron via vmap;
    no outer vmap is required by callers.

    This complements `lif_exponential_flow` (scalar params) without replacing
    it. Used by `MultiPopulationRecurrentLIF` and any future factory that
    needs per-population time constants in a single connected graph.
    """
    inv_tau_mem = jnp.atleast_1d(jnp.asarray(1. / params.tau_mem))
    inv_tau_syn = jnp.atleast_1d(jnp.asarray(1. / params.tau_syn))

    def _per_neuron_kernel(itm, its):
        return jnp.array([[-itm, itm], [0, -its]])

    kernels = jax.vmap(_per_neuron_kernel)(inv_tau_mem, inv_tau_syn)

    def _per_neuron_step(kernel, state_2, time):
        return jnp.dot(linalg.expm(kernel * time), state_2)

    def flow(state, time):
        # state.V, state.I have shape (N,)
        stacked = jnp.stack([state.V, state.I], axis=-1)
        evolved = jax.vmap(_per_neuron_step, in_axes=(0, 0, None))(
            kernels, stacked, time)
        state.V = evolved[..., 0]
        state.I = evolved[..., 1]
        return state

    return flow

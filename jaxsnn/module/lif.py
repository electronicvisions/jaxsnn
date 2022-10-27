from functools import partial

import diffrax
import jax.numpy as np
from jax import jit, random
from jax.lax import scan
from jaxsnn.functional.lif import LIFParameters, LIFState, lif_step, liv_derivative


def LIF(out_dim, scale_in=0.7, scale_rec=0.2):
    """Layer constructor function for a lif (leaky-integrated-fire) layer."""

    def init_fn(rng, input_shape):
        rng, i_key, r_key = random.split(rng, 3)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        recurrent_weights = scale_rec * random.normal(r_key, (out_dim, out_dim))
        return out_dim, (input_weights, recurrent_weights)

    def apply_fn(params, inputs, **kwargs):
        batch = inputs.shape[1]
        shape = (batch, out_dim)
        state = LIFState(np.zeros(shape), np.zeros(shape), np.zeros(shape))
        (state, _), spikes = scan(lif_step, (state, params), inputs)

        return spikes

    return init_fn, apply_fn


def LIFStep(out_dim, method, scale_in=0.7, scale_rec=0.2, **kwargs):
    """Layer constructor function for a lif (leaky-integrated-fire) layer."""

    def init_fn(rng, input_shape):
        rng, i_key, r_key = random.split(rng, 3)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        recurrent_weights = scale_rec * random.normal(r_key, (out_dim, out_dim))
        return out_dim, (input_weights, recurrent_weights), rng

    def state_fn(batch_size, **kwargs):
        shape = (batch_size, out_dim)
        state = LIFState(np.zeros(shape), np.zeros(shape), np.zeros(shape))
        return state

    lif_step_fn = jit(partial(lif_step, method=method))

    def apply_fn(state, params, inputs, **kwargs):
        return lif_step_fn((state, params), inputs)

    # def derivative_fn(state, params, **kwargs):
    #     return liv_derivative(None, state)

    # @jit
    # def apply_fn(state, params, inputs, **kwargs):

    #     term = diffrax.ODETerm(liv_derivative)

    #     z, v, i = state
    #     solver = diffrax.Tsit5()
    #     sol = diffrax.diffeqsolve(term, solver, t0=0, t1=1e-3, dt0=5e-4, y0=(v, i))
    #     v_decayed, i_decayed = sol.ys[0][0], sol.ys[1][0]

    #     input_weights, recurrent_weights = params
    #     tau_syn_inv, tau_mem_inv, v_leak, v_th, v_reset = LIFParameters()
    #     z_new = method(v_decayed - v_th)
    #     # compute reset
    #     v_new = (1 - z_new) * v_decayed + z_new * v_reset
    #     # compute current jumps
    #     i_new = i_decayed + np.matmul(z, recurrent_weights)
    #     i_new = i_new + np.matmul(inputs, input_weights)
    #     return (LIFState(z_new, v_new, i_new), params), z_new

    return init_fn, apply_fn, state_fn

import jax
import jax.api as api
import jax.numpy as np
import tqdm

from functools import partial

from jax import jit, grad, random, vmap
from jax.experimental import optimizers
from jax.lax import scan

@api.custom_vjp
def heaviside(x):
  return 0.5 + 0.5 * np.sign(x)

def heaviside_fwd(x):
  return heaviside(x), (x,)

def heaviside_bwd(res, g):
  x, = res
  grad = g / (100.0 * np.abs(x) + 1.0) ** 2
  return (grad,)

heaviside.defvjp(heaviside_fwd, heaviside_bwd)

def lif_step(
    state,
    spikes,
    tau_syn_inv=1.0 / 5e-3,
    tau_mem_inv=1.0 / 1e-2,
    v_leak=0.0,
    v_th=1.0,
    v_reset=0.0,
    dt=0.001,
):
    z, v, i, input_weights, recurrent_weights = state
    
    # compute voltage updates
    dv = dt * tau_mem_inv * ((v_leak - v) + i)
    v_decayed = v + dv

    # compute current updates
    di = -dt * tau_syn_inv * i
    i_decayed = i + di

    # compute new spikes
    z_new = heaviside(v_decayed - v_th)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * v_reset
    # compute current jumps
    i_new = (
        i_decayed
        + np.matmul(recurrent_weights, z)
    )
    i_new = i_new + np.einsum("sn,ns->n", spikes, input_weights)

    return (z_new, v_new, i_new, input_weights, recurrent_weights), z_new

def lif_integrate(init, spikes):
    return scan(lif_step, init, spikes)

def lif_init(input_weights, recurrent_weights):
    n = recurrent_weights.shape[0]
    return (np.zeros(n), np.zeros(n), np.zeros(n), input_weights, recurrent_weights)

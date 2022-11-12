from typing import Tuple

import jax.numpy as np
from jax import custom_vjp

from jaxsnn.functional.lif import LIFParameters, LIFState, lif_step


@custom_vjp
def lif_adjoint_step(init, spikes, p, dt):
    return lif_step(init, spikes, p, dt)


def lif_adjoint_step_fwd(
    init,
    spikes,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple:
    state, weights = init
    return (lif_adjoint_step(init, spikes, p, dt),)

    z_new, s_new = lif_step(init, spikes, params=p, dt=dt)
    s_old, weights = init

    # dv before spiking
    dv_m = p.tau_mem_inv * ((p.v_leak - s_old.v) + s_old.i)
    # dv after spiking
    dv_p = p.tau_mem_inv * ((p.v_leak - s_new.v) + s_old.i)

    ctx = input_tensor, z_new, dv_m, dv_p, input_weights, recurrent_weights
    return z_new, s_new.v, s_new.i


def lif_adjoint_step_bwd(ctx, doutput, lambda_v, lambda_i):
    (
        input_tensor,
        z,
        dv_m,
        dv_p,
        input_weights,
        recurrent_weights,
    ) = ctx.saved_tensors
    tau_syn_inv = ctx.tau_syn_inv
    tau_mem_inv = ctx.tau_mem_inv
    dt = ctx.dt

    dw_input = lambda_i.t().mm(input_tensor)
    dw_rec = lambda_i.t().mm(z)

    # lambda_i decay
    dlambda_i = tau_syn_inv * (lambda_v - lambda_i)
    lambda_i = lambda_i + dt * dlambda_i

    # lambda_v decay
    lambda_v = lambda_v - tau_mem_inv * dt * lambda_v

    output_term = z * (1 / dv_m) * (doutput)
    output_term[output_term != output_term] = 0.0

    jump_term = z * (dv_p / dv_m)
    jump_term[jump_term != jump_term] = 0.0

    lambda_v = (1 - z) * lambda_v + jump_term * lambda_v + output_term

    dinput = lambda_i.mm(input_weights)
    drecurrent = lambda_i.mm(recurrent_weights)

    return (dinput, drecurrent, lambda_v, lambda_i, dw_input, dw_rec, None, None)


lif_adjoint_step.defvjp(lif_adjoint_step_fwd, lif_adjoint_step_bwd)

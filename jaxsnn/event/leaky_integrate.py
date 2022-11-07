from functools import partial

import jax
import jax.numpy as np


def heaviside(x):
    return 0.5 + 0.5 * np.sign(x)


def kernel(A, t, t0):
    return heaviside(t - t0) * jax.scipy.linalg.expm(A * (t - t0))  # type: ignore


def f(A, t0, x0, t):
    return np.einsum("ijk, ik -> j", jax.vmap(partial(kernel, A, t))(t0), x0)


def li_cell(A, weights, spikes, ts):
    spike_times, spike_idx = spikes

    current = weights[spike_idx] * np.where(spike_idx == -1, 0.0, 1.0)
    voltage = np.zeros(len(spike_idx))
    xk = np.stack((voltage, current), axis=1)

    ys = jax.jit(jax.vmap(partial(f, A, spike_times, xk)))(ts)
    return ys


leaky_integrator = jax.vmap(li_cell, in_axes=(None, 1, None, None), out_axes=1)


def max_over_time(output):
    return np.max(output[::, 0], axis=0)


@jax.jit
def nll_loss(x, targets):
    x = np.maximum(x, 0)
    preds = jax.nn.log_softmax(x)
    loss = -np.sum(targets * preds)
    return loss


if __name__ == "__main__":
    # define dynamics
    tau_mem = 1e-3
    tau_syn = 5e-4
    v_th = 0.3
    tau_mem_inv = 1 / tau_mem
    tau_syn_inv = 1 / tau_syn
    A = np.array([[-tau_mem_inv, tau_mem_inv], [0, -tau_syn_inv]])

    # input spikes and weights
    weights = np.array(
        [
            [0.5, 0.8],
            [0.6, 0.9],
            [0.7, 0.1],
        ]
    )
    spike_idx = np.array([0, 1, 2, 1, -1])
    spike_times = np.array([1e-4, 1.2e-4, 2e-4, 2.2e-4, 1e-2])

    # time grid to evaluate on
    ts = np.arange(0, 1e-2, 1e-4)

    inputs = (spike_times, spike_idx)
    targets = np.array([1.0, 0.0])
    batch = (inputs, targets)

    def loss_fn(weights, batch):
        inputs, targets = batch
        output = leaky_integrator(A, weights, inputs, ts)
        print(output)
        max_voltage = max_over_time(output)

        return nll_loss(max_voltage, targets)

    print(loss_fn(weights, batch))

    # try this combi
    weights = np.array(
        [
            [1.6818058, 0.21550512],
            [0.8131372, 1.9255047],
            [0.9046461, 0.63647914],
            [0.8039566, 0.8206086],
        ]
    )

    inputs = (
        np.array([2.6588023e-05, 3.2267941e-04, 1.0000000e-01]),
        np.array([0, 1, -1]),
    )
    ts = np.arange(0, 1e-2, 1e-4)

    output = leaky_integrator(A, weights, inputs, ts)

    nll_loss(np.array([0.0, 0.0]), np.array([1.0, 0.0]))

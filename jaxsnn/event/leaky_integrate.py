from functools import partial

import jax
import jax.numpy as np


def heaviside(x):
    return 0.5 + 0.5 * np.sign(x)


def kernel(A, t, t0):
    return heaviside(t - t0) * jax.scipy.linalg.expm(A * (t - t0))  # type: ignore


def f(A, t0, x0, t):
    return np.einsum("ijk, ik -> j", jax.vmap(partial(kernel, A, t))(t0), x0)


def li_cell(A, weights, inputs, ts):
    spike_times, spike_idx = inputs

    # place a mask where spike_idx == -1
    current = weights[spike_idx] * np.where(spike_idx == -1, 0.0, 1.0)
    voltage = np.zeros(len(spike_idx))
    xk = np.stack((voltage, current), axis=1)  # type: ignore
    ys = jax.vmap(partial(f, A, spike_times, xk))(ts)
    return ys


leaky_integrator = jax.vmap(li_cell, in_axes=(None, 1, None, None), out_axes=1)


def max_over_time(output):
    return np.max(output[::, 0], axis=0)


def nll_loss(max_voltage, targets):
    preds = jax.nn.softmax(max_voltage)
    loss = -np.sum(targets * preds)

    return loss  # + regularization


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
    spike_times = np.array([1e-4, 1.2e-4, 2e-4, 2.2e-4, 5e-4])

    # time grid to evaluate on
    ts = np.arange(0, 1e-2, 1e-4)

    inputs = (spike_times, spike_idx)
    targets = np.array([1.0, 0.0])
    batch = (inputs, targets)

    def loss_fn(weights, batch):
        inputs, targets = batch
        output = leaky_integrator(A, weights, inputs, ts)
        max_voltage = max_over_time(output)

        return nll_loss(max_voltage, targets)

    print(loss_fn(weights, batch))

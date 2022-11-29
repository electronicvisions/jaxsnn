import json
from functools import partial
from typing import Callable, List, Tuple
import datetime as dt
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random
import optax

from jaxsnn.base.types import Array, Spike, Weight
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset import linear_dataset
from jaxsnn.event.leaky_integrate_and_fire import LIF, LIFParameters, RecursiveLIF
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.loss import spike_time_loss
from jaxsnn.event.plot import (
    plt_accuracy,
    plt_dataset,
    plt_loss,
    plt_no_spike_prob,
    plt_prediction,
    plt_2dloss,
    plt_t_spike_neuron,
)
from jaxsnn.event.root import ttfs_solver


def loss_and_acc(
    loss_fn: Callable,
    params: List[Weight],
    dataset: Tuple[Spike, Array],
):
    batched_loss = jax.vmap(loss_fn, in_axes=(None, 0))
    loss, (t_first_spike, _) = batched_loss(params, dataset)
    accuracy = np.argmin(dataset[1], axis=-1) == np.argmin(t_first_spike, axis=-1)
    return (
        np.mean(loss),
        np.mean(accuracy),
        t_first_spike,
    )


def train():
    # neuron params
    tau_mem = 1e-2
    tau_syn = 5e-3
    t_max = 6 * tau_syn
    v_th = 0.6
    p = LIFParameters(tau_mem_inv=1 / tau_mem, tau_syn_inv=1 / tau_syn, v_th=v_th)

    # training params
    step_size = 1e-3
    n_batches = 100
    batch_size = 32
    epochs = 50

    # net
    hidden_size = 4
    output_size = 2
    n_spikes_hidden = 20
    n_spikes_output = 30
    seed = 42
    optimizer_fn = optax.adam

    rng = random.PRNGKey(seed)
    param_rng, train_rng, test_rng = random.split(rng, 3)
    trainset = linear_dataset(train_rng, tau_syn, [n_batches, batch_size])
    testset = linear_dataset(test_rng, tau_syn, [n_batches, batch_size])
    input_size = trainset[0].idx.shape[-1]
    solver = partial(ttfs_solver, tau_mem, p.v_th)

    # declare net
    init_fn, apply_fn = serial(
        t_max,
        RecursiveLIF(
            hidden_size, n_spikes=n_spikes_hidden, t_max=t_max, p=p, solver=solver
        ),
        LIF(output_size, n_spikes=n_spikes_output, t_max=t_max, p=p, solver=solver),
    )

    # init params and optimizer
    params = init_fn(param_rng, input_size)

    optimizer = optimizer_fn(step_size)
    opt_state = optimizer.init(params)

    # declare update function
    loss_fn = batch_wrapper(partial(spike_time_loss, apply_fn, tau_mem))

    # define update function
    def update(
        input: Tuple[optax.OptState, List[Weight]],
        batch: Tuple[Spike, Array],
    ):
        opt_state, params = input
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), value

    def epoch(state, _):
        state, _ = jax.lax.scan(update, state, trainset)
        params = state[1]
        test_result = loss_and_acc(loss_fn, params, testset)
        return state, (test_result, params)

    # train the net
    (opt_state, params), (res, params_over_time) = jax.lax.scan(
        epoch, (opt_state, params), np.arange(epochs)
    )
    loss, acc, t_spike = res

    class_0 = np.argmin(testset[1], axis=-1) == 0
    t_spike_correct = np.where(class_0, t_spike[..., 0], t_spike[..., 1])
    t_spike_false = np.where(class_0, t_spike[..., 1], t_spike[..., 0])

    # plotting
    dt_string = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = f"jaxsnn/plots/event/linear/{dt_string}"
    observe = ((0, 0, "^"), (0, 1, "s"), (0, 2, "D"))

    # loss and accuracy
    fig, ax1 = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    plt_loss(ax1[0], loss)
    plt_accuracy(ax1[1], acc)
    plt_no_spike_prob(ax1[2], t_spike_correct, t_spike_false)
    fig.tight_layout()
    plt.xlabel("Epoch")
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(f"{folder}_loss.png", dpi=300)

    # 2d spike times
    fig, ax1 = plt.subplots(1, 2, figsize=(10, 4))
    plt_t_spike_neuron(fig, ax1, testset, t_spike, LIFParameters.tau_syn_inv)
    fig.savefig(f"{folder}_spike_times.png", dpi=150)

    # trajectory
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    plt_2dloss(axs, t_spike, testset, observe, tau_mem, tau_syn)
    fig.tight_layout()
    fig.savefig(f"{folder}_trajectory.png", dpi=150)

    # classification
    fig, ax1 = plt.subplots(1, 2, figsize=(7, 4))
    plt_dataset(ax1[0], testset, observe, LIFParameters.tau_syn_inv)
    plt_prediction(ax1[1], testset, t_spike, LIFParameters.tau_syn_inv)
    fig.tight_layout()
    fig.savefig(f"{folder}_classification.png", dpi=150)

    # spikes hidden layer

    # save experiment data
    experiment = {
        "epochs": epochs,
        "tau_mem": tau_mem,
        "tau_syn": tau_syn,
        "v_th": v_th,
        "step_size": step_size,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "net": [input_size, hidden_size, output_size],
        "n_spikes": [input_size, n_spikes_hidden, n_spikes_output],
        "optimizer": optimizer_fn.__name__,
        "loss": round(loss[-1].item(), 5),
        "accuracy": round(acc[-1].item(), 5),
        "target": [np.min(testset[1]).item(), np.max(testset[1]).item()],
    }
    with open(f"{folder}_params.json", "w") as outfile:
        json.dump(experiment, outfile, indent=4)


if __name__ == "__main__":
    train()

    # 2D Loss?
    # Plot Decision boundary

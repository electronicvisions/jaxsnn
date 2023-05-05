import datetime as dt
import json
from functools import partial
from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as np
import optax
from jax import random

from jaxsnn.base.types import Array, Spike, Weight
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset import linear_dataset
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.leaky_integrate_and_fire import LIF, LIFParameters
from jaxsnn.event.loss import loss_and_acc, loss_wrapper, target_time_loss
from jaxsnn.event.plot import plt_and_save
from jaxsnn.event.root import ttfs_solver


def train(folder):
    p = LIFParameters()
    t_late = 2.0 * p.tau_syn
    t_max = 4.0 * p.tau_syn

    # training params
    step_size = 1e-3
    n_batches = 100
    batch_size = 32
    epochs = 50

    # net
    hidden_size = 4
    output_size = 2
    n_neurons = hidden_size + output_size
    n_spikes_hidden = 20
    n_spikes_output = 30
    seed = 42
    optimizer_fn = optax.adam

    rng = random.PRNGKey(seed)
    param_rng, train_rng, test_rng = random.split(rng, 3)
    trainset = linear_dataset(train_rng, t_late, [n_batches, batch_size])
    testset = linear_dataset(test_rng, t_late, [n_batches, batch_size])
    input_size = trainset[0].idx.shape[-1]
    solver = partial(ttfs_solver, p.tau_mem, p.v_th)

    # declare net
    init_fn, apply_fn = serial(
        LIF(hidden_size, n_spikes=n_spikes_hidden, t_max=t_max, p=p, solver=solver),
        LIF(output_size, n_spikes=n_spikes_output, t_max=t_max, p=p, solver=solver),
    )

    # init params and optimizer
    params = init_fn(param_rng, input_size)

    optimizer = optimizer_fn(step_size)
    opt_state = optimizer.init(params)

    # declare update function
    loss_fn = partial(
        loss_wrapper, apply_fn, target_time_loss, p.tau_mem, n_neurons, output_size
    )
    loss_fn = batch_wrapper(loss_fn)

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
        state, _ = jax.lax.scan(update, state, trainset[:2])
        params = state[1]
        test_result = loss_and_acc(loss_fn, params, testset[:2])
        return state, (test_result, params)

    # train the net
    (opt_state, params), (res, params_over_time) = jax.lax.scan(
        epoch, (opt_state, params), np.arange(epochs)
    )
    loss, acc, t_spike, recording = res

    # generate plots
    plt_and_save(
        folder,
        testset,
        recording,
        t_spike,
        params_over_time,
        loss,
        acc,
        p.tau_syn,
        hidden_size,
        epochs,
    )

    # save experiment data
    max_acc = round(np.max(acc).item(), 3)
    print(f"Max acc: {max_acc} after {np.argmax(acc)} epochs")

    experiment = {
        "epochs": epochs,
        "tau_mem": p.tau_mem,
        "tau_syn": p.tau_syn,
        "v_th": p.v_th,
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
    with open(f"{folder}/params_{max_acc}.json", "w") as outfile:
        json.dump(experiment, outfile, indent=4)


if __name__ == "__main__":
    dt_string = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = f"jaxsnn/plots/event/linear/{dt_string}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"Running experiment, results in folder: {folder}")
    train(folder)

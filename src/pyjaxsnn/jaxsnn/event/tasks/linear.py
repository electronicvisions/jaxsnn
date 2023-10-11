import datetime as dt
import json
import logging
from functools import partial
from pathlib import Path

import jax
import jax.numpy as np
import optax
from jax import random
from jaxsnn.event import custom_lax
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset import linear_dataset
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.leaky_integrate_and_fire import LIF, LIFParameters
from jaxsnn.event.loss import loss_and_acc, loss_wrapper, target_time_loss
from jaxsnn.event.training import update
from jaxsnn.event.types import OptState
from jaxsnn.event.utils import time_it

log = logging.getLogger("root")


def train(seed: int, folder: str):
    params = LIFParameters()
    t_late = 2.0 * params.tau_syn
    t_max = 4.0 * params.tau_syn

    # training weights
    step_size = 1e-3
    n_batches = 100
    batch_size = 32
    epochs = 50

    # net
    hidden_size = 4
    output_size = 2

    # number of spikes in each layer need to be defined
    n_spikes_hidden = 20
    n_spikes_output = 30

    # define trainset and testset
    rng = random.PRNGKey(seed)
    param_rng, train_rng, test_rng = random.split(rng, 3)
    trainset = linear_dataset(train_rng, t_late, [n_batches, batch_size])
    testset = linear_dataset(test_rng, t_late, [n_batches, batch_size])

    # define net
    init_fn, apply_fn = serial(
        LIF(hidden_size, n_spikes=n_spikes_hidden, t_max=t_max, params=params),
        LIF(output_size, n_spikes=n_spikes_output, t_max=t_max, params=params),
    )

    # init weights and optimizer
    input_size = trainset[0].idx.shape[-1]
    weights = init_fn(param_rng, input_size)
    n_neurons = hidden_size + output_size

    # define and init optimizer
    optimizer_fn = optax.adam
    optimizer = optimizer_fn(step_size)
    opt_state = optimizer.init(weights)

    # defined loss and update function
    loss_fn = batch_wrapper(
        partial(
            loss_wrapper,
            apply_fn,
            target_time_loss,
            params.tau_mem,
            n_neurons,
            output_size,
        )
    )
    update_fn = partial(update, optimizer, loss_fn, params)

    def epoch(opt_state: OptState, i: int):
        res, duration = time_it(
            jax.lax.scan, update_fn, opt_state, trainset[:2]
        )
        opt_state, (recording, grad) = res
        test_result = loss_and_acc(loss_fn, opt_state.weights, testset[:2])
        log.info(
            f"Epoch {i}, "
            f"loss: {test_result[0]:.4f}, "
            f"acc: {test_result[1]:.3f}, "
            f"spikes: {np.sum(recording[1][1][0].idx >= 0, axis=-1).mean():.1f}, "
            f"grad: {grad[0].input.mean():.5f}, "
            f"in {duration:.2f} s"
        )
        return opt_state, (test_result, opt_state.weights)

    # iterate over epochs
    (opt_state, weights), (test_result, weights_over_time) = custom_lax.scan(
        epoch, OptState(opt_state, weights), np.arange(epochs)
    )

    # save experiment data
    max_acc = round(np.max(test_result.accuracy).item(), 3)
    log.info(
        f"Max acc: {max_acc} after {np.argmax(test_result.accuracy)} epochs"
    )

    experiment = {
        **params.as_dict(),
        "epochs": epochs,
        "step_size": step_size,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "net": [input_size, hidden_size, output_size],
        "n_spikes": [input_size, n_spikes_hidden, n_spikes_output],
        "optimizer": optimizer_fn.__name__,
        "loss": round(test_result.loss[-1].item(), 5),
        "accuracy": round(test_result.accuracy[-1].item(), 5),
        "target": [np.min(testset[1]).item(), np.max(testset[1]).item()],
    }
    with open(f"{folder}/params_{max_acc}.json", "w") as outfile:
        json.dump(experiment, outfile, indent=4)


if __name__ == "__main__":
    dt_string = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = f"data/event/linear/{dt_string}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    log.info(f"Running experiment, results in folder: {folder}")
    train(1, folder)

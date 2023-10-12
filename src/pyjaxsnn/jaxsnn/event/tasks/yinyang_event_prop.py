# pylint: disable=logging-not-lazy,logging-fstring-interpolation
import datetime as dt
import json
import logging
from functools import partial
from pathlib import Path

import jax.numpy as np
import optax
from jax import random
from jaxsnn.event import custom_lax
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset import yinyang_dataset as dataset
from jaxsnn.event.dataset.yinyang import good_params
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.leaky_integrate_and_fire import (
    LIFParameters,
    RecurrentEventPropLIF,
)
from jaxsnn.event.loss import loss_wrapper, mse_loss
from jaxsnn.event.plot import plt_and_save
from jaxsnn.event.training import epoch, update
from jaxsnn.event.types import OptState
from jaxsnn.event.utils import save_weights as save_weights_fn

log = logging.getLogger("root")


def train(  # pylint: disable=too-many-locals
    seed: int,
    folder: str,
    plot: bool = True,
    save_weights: bool = False,
):
    # neuron params, low v_reset only allows one spike per neuron
    params = LIFParameters(v_reset=-1_000.0, v_th=1.0)

    # training params
    step_size = 5e-3
    lr_decay = 0.99
    train_samples = 5_000
    test_samples = 3_000
    batch_size = 64
    epochs = 50
    n_train_batches = int(train_samples / batch_size)
    n_test_batches = int(test_samples / batch_size)
    t_max = 4.0 * params.tau_syn

    # net
    input_size = 5
    hidden_size = 100
    output_size = 3

    # number of spikes in each layer need to be defined
    n_spikes_hidden = 50
    n_spikes_output = n_spikes_hidden + 3

    # define trainset and testset
    rng = random.PRNGKey(seed)
    yinyang_params = good_params(params)
    param_rng, train_rng, test_rng = random.split(rng, 3)
    trainset = dataset(
        train_rng, [n_train_batches, batch_size], **yinyang_params
    )
    testset = dataset(test_rng, [n_test_batches, batch_size], **yinyang_params)

    # define net as one recursive layer
    weight_mean = [3.0, 0.5]
    weight_std = [1.6, 0.8]
    init_fn, apply_fn = serial(
        RecurrentEventPropLIF(
            layers=[hidden_size, output_size],
            n_spikes=n_spikes_output,
            t_max=t_max,
            params=params,
            mean=weight_mean,
            std=weight_std,
            wrap_only_step=False,
        )
    )

    # init weights
    input_size = trainset[0].idx.shape[-1]  # pylint: disable=no-member
    weights = init_fn(param_rng, input_size)
    n_neurons = weights[0].input.shape[0] + hidden_size + output_size

    # define and init optimizer
    optimizer_fn = optax.adam
    scheduler = optax.exponential_decay(step_size, n_train_batches, lr_decay)
    optimizer = optimizer_fn(scheduler)
    opt_state = optimizer.init(weights)

    # defined loss and update function
    loss_fn = batch_wrapper(
        partial(
            loss_wrapper,
            apply_fn,
            mse_loss,
            params.tau_mem,
            n_neurons,
            output_size,
        )
    )
    update_fn = partial(update, optimizer, loss_fn, params)
    epoch_fn = partial(epoch, update_fn, loss_fn, trainset, testset)

    # iterate over epochs
    res = custom_lax.scan(
        epoch_fn, OptState(opt_state, weights), np.arange(epochs)
    )
    (opt_state, weights), (test_result, weights_over_time, duration) = res

    time_string = dt.datetime.now().strftime("%H:%M:%S")
    if save_weights:
        save_weights_fn(weights, folder)

    # generate plots
    if plot:
        plt_and_save(
            folder,
            testset,
            test_result,
            weights_over_time,
            params.tau_syn,
            hidden_size,
            epochs,
            mock_hw=True,
        )

    # find best epoch
    best_epoch = np.argmax(test_result.accuracy)
    max_acc = round(test_result.accuracy[best_epoch].item(), 3)
    log.info(f"Max acc: {max_acc} after {best_epoch} epochs")

    # save experiment data
    experiment = {
        **params.as_dict(),
        **yinyang_params,
        "max_accuracy": max_acc,
        "seed": seed,
        "epochs": epochs,
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "step_size": step_size,
        "lr_decay": lr_decay,
        "batch_size": batch_size,
        "n_samples": train_samples,
        "net": [input_size, hidden_size, output_size],
        "n_spikes": [input_size, n_spikes_hidden, n_spikes_output],
        "optimizer": optimizer_fn.__name__,
        "loss": [round(float(loss), 5) for loss in test_result.loss],
        "accuracy": [round(float(acc), 5) for acc in test_result.accuracy],
        "time per epoch": [round(float(dur), 3) for dur in duration],
    }

    filename = f"{folder}/params_{max_acc}_{time_string}.json"
    with open(filename, "w", encoding="utf-8") as outfile:
        json.dump(experiment, outfile, indent=4)


if __name__ == "__main__":
    dt_string = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_folder = f"data/event/yinyang_event_prop/{dt_string}"
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    log.info(f"Running experiment, results in folder: {data_folder}")

    train(0, data_folder, plot=False, save_weights=False)

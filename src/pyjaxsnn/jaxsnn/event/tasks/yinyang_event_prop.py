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
    EventPropLIF,
    LIFParameters,
    RecurrentEventPropLIF,
)
from jaxsnn.event.loss import loss_wrapper, mse_loss
from jaxsnn.event.plot import plt_and_save
from jaxsnn.event.training import epoch, update
from jaxsnn.event.types import OptState
from jaxsnn.event.utils import save_params as save_params_fn

log = logging.getLogger("root")


def train(
    seed: int,
    folder: str,
    plot: bool = True,
    save_params: bool = False,
):
    # neuron params, low v_reset only allows one spike per neuron
    p = LIFParameters(v_reset=-1_000.0, v_th=1.0)

    # training params
    step_size = 5e-3
    lr_decay = 0.99
    train_samples = 5_000
    test_samples = 3_000
    batch_size = 64
    epochs = 50
    n_train_batches = int(train_samples / batch_size)
    n_test_batches = int(test_samples / batch_size)
    t_max = 4.0 * p.tau_syn

    # net
    input_size = 5
    hidden_size = 100
    output_size = 3

    # number of spikes in each layer need to be defined
    n_spikes_hidden = 50
    n_spikes_output = n_spikes_hidden + 3

    # define trainset and testset
    rng = random.PRNGKey(seed)
    yinyang_params = good_params(p)
    param_rng, train_rng, test_rng = random.split(rng, 3)
    trainset = dataset(train_rng, [n_train_batches, batch_size], **yinyang_params)
    testset = dataset(test_rng, [n_test_batches, batch_size], **yinyang_params)

    # define net as one recursive layer
    weight_mean = [3.0, 0.5]
    weight_std = [1.6, 0.8]
    init_fn, apply_fn = serial(
        # RecurrentEventPropLIF(
        #     layers=[hidden_size, output_size],
        #     n_spikes=n_spikes_output,
        #     t_max=t_max,
        #     p=p,
        #     mean=weight_mean,
        #     std=weight_std,
        #     wrap_only_step=False,
        # )
        EventPropLIF(
            n_hidden=hidden_size,
            n_spikes=n_spikes_output,
            t_max=t_max,
            p=p,
            mean=weight_mean[0],
            std=weight_std[0],
        ),
        EventPropLIF(
            n_hidden=output_size,
            n_spikes=n_spikes_output,
            t_max=t_max,
            p=p,
            mean=weight_mean[1],
            std=weight_std[1],
        ),
    )

    # init params
    input_size = trainset[0].idx.shape[-1]
    params = init_fn(param_rng, input_size)
    n_neurons = params[0].input.shape[0] + hidden_size + output_size

    # define and init optimizer
    optimizer_fn = optax.adam
    scheduler = optax.exponential_decay(step_size, n_train_batches, lr_decay)
    optimizer = optimizer_fn(scheduler)
    opt_state = optimizer.init(params)

    # defined loss and update function
    loss_fn = batch_wrapper(
        partial(loss_wrapper, apply_fn, mse_loss, p.tau_mem, n_neurons, output_size)
    )
    update_fn = partial(update, optimizer, loss_fn, p)
    epoch_fn = partial(epoch, update_fn, loss_fn, trainset, testset)

    # iterate over epochs
    res = custom_lax.scan(epoch_fn, OptState(opt_state, params), np.arange(epochs))
    (opt_state, params), (test_result, params_over_time, duration) = res

    time_string = dt.datetime.now().strftime("%H:%M:%S")
    if save_params:
        save_params_fn(params, folder)

    # generate plots
    if plot:
        plt_and_save(
            folder,
            testset,
            test_result,
            params_over_time,
            p.tau_syn,
            hidden_size,
            epochs,
            mock_hw=True,
        )

    # save experiment data
    max_acc = round(np.max(test_result.accuracy).item(), 3)
    log.info(f"Max acc: {max_acc} after {np.argmax(test_result.accuracy)} epochs")
    experiment = {
        **p.as_dict(),
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
        "loss": [round(float(l), 5) for l in test_result.loss],
        "accuracy": [round(float(a), 5) for a in test_result.accuracy],
        "time per epoch": [round(float(d), 3) for d in duration],
    }

    with open(f"{folder}/params_{max_acc}_{time_string}.json", "w") as outfile:
        json.dump(experiment, outfile, indent=4)


if __name__ == "__main__":
    dt_string = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = f"data/event/yinyang_event_prop/{dt_string}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    log.info(f"Running experiment, results in folder: {folder}")

    train(0, folder, plot=False, save_params=False)

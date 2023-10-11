import datetime as dt
import json
import logging
import time
from functools import partial
from pathlib import Path
from typing import List, Tuple

import hxtorch
import jax
import jax.numpy as np
import numpy as onp
import optax
from jax import random
from jaxsnn.event import custom_lax
from jaxsnn.event.compose import serial, serial_spikes_known
from jaxsnn.event.dataset import linear_dataset
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.hardware.calib import W_69_F0_LONG_REFRAC
from jaxsnn.event.hardware.experiment import Experiment
from jaxsnn.event.hardware.input_neuron import InputNeuron
from jaxsnn.event.hardware.neuron import Neuron
from jaxsnn.event.hardware.utils import simulate_hw_weights
from jaxsnn.event.leaky_integrate_and_fire import (
    EventPropLIF,
    HardwareLIF,
    LIFParameters,
)
from jaxsnn.event.loss import (
    loss_and_acc_scan,
    loss_wrapper_known_spikes,
    mse_loss,
)
from jaxsnn.event.plot import plt_and_save
from jaxsnn.event.types import Spike, Weight
from jaxsnn.event.utils import save_weights as save_weights_fn

log = logging.getLogger("root")


wafer_config = W_69_F0_LONG_REFRAC
MOCK_HW = False


def train(
    seed: int,
    folder: str,
    plot: bool = True,
    save_weights: bool = False,
):
    # neuron params, low v_reset only allows one spike per neuron
    params = LIFParameters(
        v_reset=-100.0, v_th=1.0, tau_syn_inv=1 / 6e-6, tau_mem_inv=1 / 12e-6
    )

    # training params
    step_size = 5e-4
    lr_decay = 0.99
    train_samples = 5_000
    test_samples = 3_000
    batch_size = 64
    epochs = 30
    n_train_batches = int(train_samples / batch_size)
    n_test_batches = int(test_samples / batch_size)
    t_late = 2.0 * params.tau_syn
    t_max = 4.0 * params.tau_syn
    duplication = 5
    duplicate_neurons = True

    # at least 50 us because otherwise we get jitter
    t_max_us = max(t_max / 1e-6, 50)
    log.info(f"Runtime in us: {t_max_us}")
    weight_mean = 3.0 / duplication
    weight_std = 1.6 / duplication
    bias_spike = 0.9 * params.tau_syn

    log.warning(
        f"Mock HW set to {MOCK_HW}, lr: {step_size}, bias spike: {bias_spike}"
    )

    correct_target_time = 0.5 * params.tau_syn
    wrong_target_time = 1.5 * params.tau_syn

    # net
    input_size = 4 + int(bias_spike is not None)
    if duplicate_neurons:
        input_size *= duplication

    hidden_size = 0
    output_size = 2

    # n_spikes
    n_spikes_hidden = input_size * duplication + hidden_size
    n_spikes_output = 2
    n_spikes = n_spikes_hidden + n_spikes_output

    optimizer_fn = optax.adam
    n_neurons = input_size + hidden_size + output_size
    log.info(f"Neurons: {n_neurons}")

    rng = random.PRNGKey(seed)
    param_rng, train_rng, test_rng = random.split(rng, 3)

    dataset_kwargs = {
        "mirror": True,
        "bias_spike": bias_spike,
        "correct_target_time": correct_target_time,
        "wrong_target_time": wrong_target_time,
        "duplication": duplication,
        "duplicate_neurons": duplicate_neurons,
    }

    trainset = linear_dataset(
        train_rng, t_late, [n_train_batches, batch_size], **dataset_kwargs
    )
    testset = linear_dataset(
        test_rng,
        t_late,
        [n_test_batches, batch_size],
        **dataset_kwargs,
    )

    # declare mock net
    _, hw_mock = serial(
        EventPropLIF(
            n_hidden=output_size,
            n_spikes=n_spikes,
            t_max=t_max,
            params=params,
            mean=weight_mean,
            std=weight_std,
            duplication=duplication if duplicate_neurons else None,
        )
    )

    init_fn, apply_fn = serial_spikes_known(
        HardwareLIF(
            n_hidden=output_size,
            n_spikes=n_spikes,
            t_max=t_max,
            params=params,
            mean=weight_mean,
            std=weight_std,
        )
    )

    # init weights and optimizer
    weights = init_fn(param_rng, input_size)
    scheduler = optax.exponential_decay(step_size, n_train_batches, lr_decay)

    optimizer = optimizer_fn(scheduler)
    opt_state = optimizer.init(weights)

    loss_fn = jax.jit(
        batch_wrapper(
            partial(
                loss_wrapper_known_spikes,
                apply_fn,
                mse_loss,
                params.tau_mem,
                n_neurons,
                output_size,
            ),
            in_axes=(None, 0, 0),
        )
    )

    # HW
    experiment = Experiment(calib_path=wafer_config.file)
    InputNeuron(input_size, params, experiment)
    Neuron(output_size, params, experiment)

    # define test function
    def test_loss_fn(weights, batch):
        input_spikes, _ = batch

        if MOCK_HW:
            hw_spikes = jax.vmap(hw_mock, in_axes=(None, 0))(
                weights, input_spikes
            )
        else:
            hw_spikes, _ = experiment.get_hw_results(
                input_spikes, weights, t_max_us, [n_spikes_output], {}
            )

        return weights, loss_fn(weights, batch, hw_spikes)

    # define update function
    def update(input, batch):
        opt_state, weights, time_data = input
        input_spikes, _ = batch

        if MOCK_HW:
            hw_spikes = jax.vmap(hw_mock, in_axes=(None, 0))(
                weights, input_spikes
            )
        else:
            hw_spikes, time_data = experiment.get_hw_results(
                input_spikes, weights, t_max_us, [n_spikes_output], time_data
            )

        return update_software(
            (opt_state, weights, time_data), batch, hw_spikes
        )

    @jax.jit
    def update_software(
        input: Tuple[optax.OptState, List[Weight], dict],
        batch: Tuple[Spike, jax.Array],
        hw_spikes,
    ):
        opt_state, weights, hw_time = input

        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(
            simulate_hw_weights(weights), batch, hw_spikes
        )

        grad = jax.tree_util.tree_map(
            lambda par, g: np.where(par == 0.0, 0.0, g / params.tau_syn),
            weights,
            grad,
        )

        updates, opt_state = optimizer.update(grad, opt_state)
        weights = optax.apply_updates(weights, updates)

        return (opt_state, weights, hw_time), (value, grad)

    def epoch(state, i):
        weights = state[1]
        test_result = loss_and_acc_scan(test_loss_fn, weights, testset[:2])

        start = time.time()
        state, (recording, grad) = custom_lax.scan(
            update, (state[0], state[1], {}), trainset[:2]
        )
        duration = time.time() - start

        masked = onp.ma.masked_where(
            recording[1][0] == np.inf, recording[1][0]
        )
        log.info(
            f"Epoch {i}, loss: {test_result[0]:.6f}, "
            f"acc: {test_result[1]:.3f}, "
            f"spikes: {np.sum(recording[1][1][0].idx >= 0, axis=-1).mean():.1f}, "
            f"grad: {grad[0].input.mean():.9f}, ",
            f"weights: {weights[0].input.mean():.5f}, ",
            f"time first output: {(masked.mean() / params.tau_syn):.2f} tau_s, "
            f"in {duration:.2f} s, ",
            f"hw time: {state[2].get('get_hw_results', 0.0):.2f} s, ",
            f"grenade run time: {state[2].get('grenade_run', 0.0):.2f} s",
        )
        return state[:2], (test_result, weights, duration)

    # train the net
    (opt_state, weights), (
        res,
        weights_over_time,
        durations,
    ) = custom_lax.scan(epoch, (opt_state, weights), np.arange(epochs))
    loss, acc, t_spike, recording = res  # type: ignore

    time_string = dt.datetime.now().strftime("%H:%M:%S")
    if save_weights:
        filenames = [f"{folder}/weights_1.npy", f"{folder}/weights_2.npy"]
        save_weights_fn(weights, filenames)

    # generate plots
    if plot:
        plt_and_save(
            folder,
            testset,
            recording,  # type: ignore
            t_spike,  # type: ignore
            weights_over_time,
            loss,  # type: ignore
            acc,  # type: ignore
            params.tau_syn,  # type: ignore
            hidden_size,
            epochs,
            duplication,
            duplicate_neurons,
        )

    # save experiment data
    max_acc = round(np.max(acc).item(), 3)  # type: ignore
    log.info(f"Max acc: {max_acc} after {np.argmax(acc)} epochs")  # type: ignore
    experiment = {
        "mock_hw": MOCK_HW,
        "max_accuracy": max_acc,
        "seed": seed,
        "epochs": epochs,
        "tau_mem": params.tau_mem,
        "tau_syn": params.tau_syn,
        "v_th": params.v_th,
        "v_reset": params.v_reset,
        "t_late": t_late,
        "bias_spike (tau_syn)": round(bias_spike / params.tau_syn, 4)
        if bias_spike is not None
        else None,
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "step_size": step_size,
        "lr_decay": lr_decay,
        "batch_size": batch_size,
        "n_samples": train_samples,
        "net": [input_size, hidden_size, output_size],
        "n_spikes": [input_size, n_spikes_hidden, n_spikes_output],
        "optimizer": optimizer_fn.__name__,
        "target (tau_syn)": [
            round(float(correct_target_time) / params.tau_syn, 4),
            round(float(wrong_target_time) / params.tau_syn, 4),
        ],
        "loss": [round(float(l), 5) for l in loss],  # type: ignore
        "accuracy": [round(float(a), 5) for a in acc],  # type: ignore
        "time per epoch": [round(float(d), 3) for d in durations],
    }

    with open(f"{folder}/params_{max_acc}_{time_string}.json", "w") as outfile:
        json.dump(experiment, outfile, indent=4)


if __name__ == "__main__":
    dt_string = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = f"jaxsnn/plots/hardware/linear/{dt_string}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    log.info(f"Running experiment, results in folder: {folder}")

    hxtorch.init_hardware()
    train(0, folder, plot=True, save_weights=True)
    hxtorch.release_hardware()

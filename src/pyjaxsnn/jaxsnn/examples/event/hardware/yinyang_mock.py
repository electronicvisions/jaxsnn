"""When training with the BSS-2 system, the forward pass is executed on BSS-2.
Spike times are returned. Because they are missing information about the actual
This training task allows setting the `MOCK_HW` variable to switch between the
actual BSS-2 system and a mock version, in which a first forward pass is also"""

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
from jaxsnn.event.dataset import yinyang_dataset
from jaxsnn.event.dataset.yinyang import good_params_for_hw
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.hardware.calib import W_69_F0_LONG_REFRAC
from jaxsnn.event.hardware.experiment import Experiment
from jaxsnn.event.hardware.input_neuron import InputNeuron
from jaxsnn.event.hardware.neuron import Neuron
from jaxsnn.event.hardware.utils import (
    add_linear_noise,
    add_noise_batch,
    cut_spikes_batch,
    filter_spikes_batch,
    simulate_hw_weights,
    sort_batch,
)
from jaxsnn.event.leaky_integrate_and_fire import (
    HardwareRecurrentLIF,
    LIFParameters,
    RecurrentEventPropLIF,
)
from jaxsnn.event.loss import (
    loss_and_acc_scan,
    loss_wrapper_known_spikes,
    mse_loss,
)
from jaxsnn.event.types import Spike, Weight
from jaxsnn.event.utils import load_weights_recurrent, save_weights_recurrent
from jaxsnn.examples.plot import plt_and_save

log = logging.getLogger("root")

# select one of multiple saved wafer configs
wafer_config = W_69_F0_LONG_REFRAC

# replace bss-2 by software simulation
MOCK_HW = True

# simulate some hw behaviour in mock-mode
SIM_HW_WEIGHTS_RANGE = True
SIM_HW_WEIGHTS_INT = False
NOISE = None
NOISE_BIAS = 4e-7

# how to adjust hw spike data
HW_CYCLE_CORRECTION = -50

# load weights from this folder
LOAD_FOLDER = None

# Limit the range of gradients. Because the current is added to the spikes
# from the hw in a second software run, it can happen that the software
# simulation and the hardware are a bit out of syncs, which can lead
# to some divergence on gradients
MAX_GRAD = [0.01, 0.01]


def train(
    seed: int,
    folder: str,
    plot: bool = True,
    save_weights: bool = False,
):
    # neuron params, low v_reset only allows one spike per neuron
    params = LIFParameters(
        v_reset=-1000.0, v_th=1.0, tau_syn_inv=1 / 6e-6, tau_mem_inv=1 / 12e-6
    )

    # training params
    step_size = 5e-3
    lr_decay = 0.99
    train_samples = 4_800
    test_samples = 3_000
    batch_size = 64
    epochs = 300
    n_train_batches = int(train_samples / batch_size)
    n_test_batches = int(test_samples / batch_size)
    t_max = 4.0 * params.tau_syn

    # at least 50 us because otherwise we get jitter
    t_max_us = int(max(t_max / 1e-6, 50))

    yinyang_params = good_params_for_hw(params)

    # input is duplicated because of dynamic range of hw
    duplication = yinyang_params["duplication"]
    weight_mean = [3.0 / duplication, 0.5]
    weight_std = [1.6 / duplication, 0.8]

    # how many input neurons do we have?
    input_size = 5
    if yinyang_params["duplicate_neurons"]:
        input_size *= duplication

    log.info(
        f"Mock HW set to {MOCK_HW}, "
        f"lr: {step_size}, "
        f"bias spike: {yinyang_params['bias_spike']}, "
        f"duplication: {duplication}, "
        f"sim hw weights: {SIM_HW_WEIGHTS_RANGE}, "
        f"sim integer: {SIM_HW_WEIGHTS_INT} "
        f"config: {wafer_config.name}, "
        f"noise: {NOISE}, "
        f"correction: {HW_CYCLE_CORRECTION}, "
        f"max_grad: {MAX_GRAD}"
    )

    # network size
    hidden_size = 100
    output_size = 3
    n_neurons = input_size + hidden_size + output_size

    # n_spikes
    n_spikes_input = 5 * duplication
    n_spikes_hidden = hidden_size
    n_spikes_output = output_size
    n_spikes = n_spikes_input + n_spikes_hidden + n_spikes_output

    # define trainset and testset
    rng = random.PRNGKey(seed)
    param_rng, train_rng, test_rng = random.split(rng, 3)
    trainset = yinyang_dataset(
        train_rng,
        [n_train_batches, batch_size],
        **yinyang_params,
    )
    testset = yinyang_dataset(
        test_rng,
        [n_test_batches, batch_size],
        **yinyang_params,
    )

    # recurrent net which mocks BSS-2
    _, hw_mock = serial(
        RecurrentEventPropLIF(
            layers=[hidden_size, output_size],
            n_spikes=n_spikes,
            t_max=t_max,
            params=params,
            mean=weight_mean,
            std=weight_std,
            wrap_only_step=False,
        )
    )
    hw_mock_batched = jax.jit(jax.vmap(hw_mock, in_axes=(None, 0)))

    # software net which adds the current in a second pass
    # and calculates the gradients with EventProp
    init_fn, apply_fn = serial_spikes_known(
        HardwareRecurrentLIF(
            layers=[hidden_size, output_size],
            n_spikes=n_spikes,
            t_max=t_max,
            params=params,
            mean=weight_mean,
            std=weight_std,
            duplication=duplication
            if yinyang_params["duplicate_neurons"]
            else None,
        )
    )

    # init weights
    if LOAD_FOLDER is None:
        weights = init_fn(param_rng, input_size)
    else:
        log.info(f"Loading weights from folder: {LOAD_FOLDER}")
        weights = [load_weights_recurrent(LOAD_FOLDER)]

    # define and init optimizer
    optimizer_fn = optax.adam
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
            pmap=False,
        )
    )

    # set up neurons on BSS-2
    if not MOCK_HW:
        experiment = Experiment(wafer_config)
        InputNeuron(input_size, params, experiment)
        Neuron(hidden_size, params, experiment)
        Neuron(output_size, params, experiment)

    def update(input, batch):
        (opt_state, weights, time_data), rng = input
        input_spikes, _ = batch

        if MOCK_HW:
            if SIM_HW_WEIGHTS_INT:
                hw_spikes = hw_mock_batched(
                    simulate_hw_weights(
                        weights, wafer_config.weight_scaling, as_int=True
                    ),
                    input_spikes,
                )
            else:
                hw_spikes = hw_mock_batched(weights, input_spikes)
        else:
            hw_spikes, time_data = experiment.get_hw_results(
                input_spikes,
                weights,
                t_max_us,
                n_spikes=[n_spikes_hidden, n_spikes_output],
                time_data=time_data,
                hw_cycle_correction=HW_CYCLE_CORRECTION,
            )

            # merge to one layer
            hw_spikes = [
                sort_batch(
                    Spike(
                        idx=np.concatenate(
                            (hw_spikes[0].idx, hw_spikes[1].idx), axis=-1
                        ),
                        time=np.concatenate(
                            (hw_spikes[0].time, hw_spikes[1].time), axis=-1
                        ),
                    )
                )
            ]

            # add time noise, this is necessary to not have two spikes at the
            # same time
            hw_spikes = [add_linear_noise(hw_spikes[0])]

        # add noise to training to better emulate BSS-2
        if NOISE is not None:
            rng, noise_rng = jax.random.split(rng, 2)
            hw_spikes = [
                add_noise_batch(
                    hw_spikes[0], noise_rng, std=NOISE, bias=NOISE_BIAS
                )
            ]
        res = update_software(
            (opt_state, weights, time_data), batch, hw_spikes
        )
        return (res[0], rng), res[1]

    # define test function
    def test_loss_fn(weights, batch):
        weights, rng = weights
        input_spikes, _ = batch

        if MOCK_HW:
            if SIM_HW_WEIGHTS_INT:
                hw_spikes = hw_mock_batched(
                    simulate_hw_weights(
                        weights, wafer_config.weight_scaling, as_int=True
                    ),
                    input_spikes,
                )
            else:
                hw_spikes = hw_mock_batched(weights, input_spikes)
            hw_spikes = [
                cut_spikes_batch(
                    filter_spikes_batch(hw_spikes[0], input_size),
                    n_spikes_hidden + n_spikes_output,
                )
            ]
        else:
            hw_spikes, _ = experiment.get_hw_results(
                input_spikes,
                weights,
                t_max_us,
                n_spikes=[n_spikes_hidden, n_spikes_output],
                time_data={},
                hw_cycle_correction=HW_CYCLE_CORRECTION,
            )

            # merge to one layer
            hw_spikes = [
                sort_batch(
                    Spike(
                        idx=np.concatenate(
                            (hw_spikes[0].idx, hw_spikes[1].idx), axis=-1
                        ),
                        time=np.concatenate(
                            (hw_spikes[0].time, hw_spikes[1].time), axis=-1
                        ),
                    )
                )
            ]

        # add time noise to not have multiple spikes at the same time
        hw_spikes = [add_linear_noise(hw_spikes[0])]

        if NOISE is not None:
            rng, noise_rng = jax.random.split(rng, 2)
            hw_spikes = [
                add_noise_batch(
                    hw_spikes[0], noise_rng, std=NOISE, bias=NOISE_BIAS
                )
            ]

        if SIM_HW_WEIGHTS_INT:
            loss_result = loss_fn(
                simulate_hw_weights(
                    weights, wafer_config.weight_scaling, as_int=True
                ),
                batch,
                hw_spikes,
            )
        else:
            loss_result = loss_fn(weights, batch, hw_spikes)
        return (weights, rng), loss_result

    @jax.jit
    def update_software(
        input: Tuple[optax.OptState, List[Weight], dict],
        batch: Tuple[Spike, jax.Array],
        hw_spikes,
    ):
        opt_state, weights, hw_time = input
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(
            weights, batch, hw_spikes
        )

        grad = jax.tree_util.tree_map(
            lambda par, g: np.where(par == 0.0, 0.0, g), weights, grad
        )

        # grad clipping
        if MAX_GRAD is not None:
            grad = jax.tree_util.tree_map(
                lambda par, g: np.where(np.abs(g) > MAX_GRAD[1], 0.0, g),
                weights,
                grad,
            )

        updates, opt_state = optimizer.update(grad, opt_state)
        weights = optax.apply_updates(weights, updates)
        if SIM_HW_WEIGHTS_RANGE:
            weights = simulate_hw_weights(weights, wafer_config.weight_scaling)

        return (opt_state, weights, hw_time), (value, grad)

    def epoch(state, i):
        # do testing before training for plot
        weights = state[1]

        rng = jax.random.PRNGKey(i)
        test_rng, train_rng = jax.random.split(rng, 2)
        test_result = loss_and_acc_scan(
            test_loss_fn, (weights, test_rng), testset[:2]
        )
        loss, acc, t_first_spike, recording = test_result

        start = time.time()
        (state, rng), (_, grad) = custom_lax.scan(
            update, (state, train_rng), trainset[:2]
        )

        duration = time.time() - start
        masked = onp.ma.masked_where(t_first_spike == np.inf, t_first_spike)
        number_of_hidden_spikes = np.sum(
            input_size <= recording[0].idx, axis=-1
        ).mean()
        input_param = weights[0].input[:, :hidden_size]
        recurrent_param = weights[0].recurrent[:hidden_size, hidden_size:]
        log.info(
            f"Epoch {i}, loss: {loss:.6f}, "
            f"acc: {acc:.3f}, "
            f"spikes: {number_of_hidden_spikes:.1f}, "
            f"output inf: {np.mean((t_first_spike == np.inf), axis=(0, 1))}, "
            f"grad: {np.mean(np.abs(grad[0].input[:, :,:hidden_size])):.4f}, {np.mean(np.abs(grad[0].recurrent[:, :hidden_size,hidden_size:])):.4f}, "
            f"max grad: {np.max(np.abs(grad[0].input[:, :,:hidden_size])):.4f}, {np.max(np.abs(grad[0].recurrent[:, :hidden_size,hidden_size:])):.4f}, "
            f"weights mean: {input_param.mean():.5f}, {recurrent_param.mean():.5f}, "
            f"weights std: {input_param.std():.5f}, {recurrent_param.std():.5f}, "
            f"param sat: {np.abs(input_param * wafer_config.weight_scaling >= 63).mean():.3f}, {np.abs(recurrent_param * wafer_config.weight_scaling >= 63).mean():.3f}, "
            f"time output: {(masked.mean() / params.tau_syn):.2f} tau_s, "
            f"in {duration:.2f} s, "
            f"hw time: {state[2].get('get_hw_results', 0.0):.2f} s, "
            f"grenade run time: {state[2].get('grenade_run', 0.0):.2f} s, "
            f"get observables: {state[2].get('get_observables', 0.0):.2f} s, "
            f"generate_network_time: {state[2].get('generate_network_time', 0.0):.2f} s, "
        )

        # reset timing stats
        for k, v in state[2].items():
            state[2][k] = 0.0
        return state, (test_result, weights, duration)

    # train the net
    res = custom_lax.scan(epoch, (opt_state, weights, {}), np.arange(epochs))
    (opt_state, weights, timing), (
        test_result,
        weights_over_time,
        durations,
    ) = res

    time_string = dt.datetime.now().strftime("%H:%M:%S")
    Path(folder).mkdir(parents=True, exist_ok=True)
    if save_weights:
        save_weights_recurrent(weights[0], folder)

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
            duplication,
            yinyang_params["duplicate_neurons"],
            MOCK_HW,
        )

    # find best epoch
    best_epoch = np.argmax(test_result.accuracy)
    max_acc = round(test_result.accuracy[best_epoch].item(), 3)
    log.info(f"Max acc: {max_acc} after {best_epoch} epochs")

    # save experiment data
    experiment = {
        **params.as_dict(),
        **wafer_config._asdict(),
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
        "time per epoch": [round(float(d), 3) for d in durations],
        "mock_hw": MOCK_HW,
        "noise": NOISE,
        "sim_hw_weights_range": SIM_HW_WEIGHTS_RANGE,
        "sim_hw_weights_int": SIM_HW_WEIGHTS_INT,
    }

    with open(f"{folder}/params_{max_acc}_{time_string}.json", "w") as outfile:
        json.dump(experiment, outfile, indent=4)


if __name__ == "__main__":
    dt_string = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = f"data/event/hardware/yinyang_{'mock' if MOCK_HW else 'no_mock'}/{dt_string}"
    log.info(f"Running experiment, results in folder: {folder}")

    if not MOCK_HW:
        hxtorch.init_hardware()
    train(0, folder, plot=True, save_weights=False)
    if not MOCK_HW:
        hxtorch.release_hardware()

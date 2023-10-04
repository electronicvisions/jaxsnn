import datetime as dt
import json
import time
from jaxsnn.event.hardware.experiment import Experiment
from jaxsnn.event.hardware.neuron import Neuron
from jaxsnn.event.hardware.input_neuron import InputNeuron
from functools import partial
from pathlib import Path
from typing import List, Tuple
import jax
import numpy as onp
import jax.numpy as np
import optax
from jax import random
import hxtorch
from jaxsnn.event.hardware.utils import (
    simulate_hw_weights,
)

from jaxsnn.base.types import Array, Spike, Weight
from jaxsnn.event.compose import serial, serial_spikes_known
from jaxsnn.event.dataset import linear_dataset
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.leaky_integrate_and_fire import (
    LIFParameters,
    HardwareLIF,
    EventPropLIF,
)
from jaxsnn.event.loss import (
    loss_wrapper_known_spikes,
    loss_and_acc_scan,
    mse_loss,
)
from jaxsnn.event.plot import plt_and_save
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.utils import save_params as save_params_fn
from jaxsnn.event import custom_lax
from jax.config import config
from jaxsnn.event.hardware.calib import W_69_F0_LONG_REFRAC

log = hxtorch.logger.get("hxtorch.snn.experiment")

config.update("jax_debug_nans", True)


wafer_config = W_69_F0_LONG_REFRAC
MOCK_HW = False


def train(
    seed: int,
    folder: str,
    plot: bool = True,
    print_epoch: bool = True,
    save_params: bool = False,
):
    # neuron params, low v_reset only allows one spike per neuron
    p = LIFParameters(
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
    t_late = 2.0 * p.tau_syn
    t_max = 4.0 * p.tau_syn
    duplication = 5
    duplicate_neurons = True

    # at least 50 us because otherwise we get jitter
    t_max_us = max(t_max / 1e-6, 50)
    log.INFO(f"Runtime in us: {t_max_us}")
    weight_mean = 3.0 / duplication
    weight_std = 1.6 / duplication
    bias_spike = 0.9 * p.tau_syn

    log.WARN(f"Mock HW set to {MOCK_HW}, lr: {step_size}, bias spike: {bias_spike}")

    correct_target_time = 0.5 * p.tau_syn
    wrong_target_time = 1.5 * p.tau_syn

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
    log.INFO(f"Neurons: {n_neurons}")

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

    solver = partial(ttfs_solver, p.tau_mem, p.v_th)

    # declare mock net
    _, hw_mock = serial(
        EventPropLIF(
            n_hidden=output_size,
            n_spikes=n_spikes,
            t_max=t_max,
            p=p,
            solver=solver,
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
            p=p,
            mean=weight_mean,
            std=weight_std,
        )
    )

    # init params and optimizer
    params = init_fn(param_rng, input_size)
    scheduler = optax.exponential_decay(step_size, n_train_batches, lr_decay)

    optimizer = optimizer_fn(scheduler)
    opt_state = optimizer.init(params)

    loss_fn = jax.jit(
        batch_wrapper(
            partial(
                loss_wrapper_known_spikes,
                apply_fn,
                mse_loss,
                p.tau_mem,
                n_neurons,
                output_size,
            ),
            in_axes=(None, 0, 0),
        )
    )

    # HW
    experiment = Experiment(calib_path=wafer_config.file)
    InputNeuron(input_size, p, experiment)
    Neuron(output_size, p, experiment)

    # define test function
    def test_loss_fn(params, batch):
        input_spikes, _ = batch

        if MOCK_HW:
            hw_spikes = jax.vmap(hw_mock, in_axes=(None, 0))(params, input_spikes)
        else:
            hw_spikes, _ = experiment.get_hw_results(
                input_spikes, params, t_max_us, [n_spikes_output], {}
            )

        return params, loss_fn(params, batch, hw_spikes)

    # define update function
    def update(input, batch):
        opt_state, params, time_data = input
        input_spikes, _ = batch

        if MOCK_HW:
            hw_spikes = jax.vmap(hw_mock, in_axes=(None, 0))(params, input_spikes)
        else:
            hw_spikes, time_data = experiment.get_hw_results(
                input_spikes, params, t_max_us, [n_spikes_output], time_data
            )

        return update_software((opt_state, params, time_data), batch, hw_spikes)

    @jax.jit
    def update_software(
        input: Tuple[optax.OptState, List[Weight], dict],
        batch: Tuple[Spike, Array],
        hw_spikes,
    ):
        opt_state, params, hw_time = input

        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(
            simulate_hw_weights(params), batch, hw_spikes
        )

        grad = jax.tree_util.tree_map(
            lambda par, g: np.where(par == 0.0, 0.0, g / p.tau_syn), params, grad
        )

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        return (opt_state, params, hw_time), (value, grad)

    def epoch(state, i):
        params = state[1]
        test_result = loss_and_acc_scan(test_loss_fn, params, testset[:2])

        start = time.time()
        state, (recording, grad) = custom_lax.simple_scan(
            update, (state[0], state[1], {}), trainset[:2]
        )
        duration = time.time() - start

        if print_epoch:
            masked = onp.ma.masked_where(recording[1][0] == np.inf, recording[1][0])
            log.INFO(
                f"Epoch {i}, loss: {test_result[0]:.6f}, "
                f"acc: {test_result[1]:.3f}, "
                f"spikes: {np.sum(recording[1][1][0].idx >= 0, axis=-1).mean():.1f}, "
                f"grad: {grad[0].input.mean():.9f}, ",
                f"params: {params[0].input.mean():.5f}, ",
                f"time first output: {(masked.mean() / p.tau_syn):.2f} tau_s, "
                f"in {duration:.2f} s, ",
                f"hw time: {state[2].get('get_hw_results', 0.0):.2f} s, ",
                f"grenade run time: {state[2].get('grenade_run', 0.0):.2f} s",
            )
        return state[:2], (test_result, params, duration)

    # train the net
    (opt_state, params), (res, params_over_time, durations) = custom_lax.simple_scan(
        epoch, (opt_state, params), np.arange(epochs)
    )
    loss, acc, t_spike, recording = res  # type: ignore

    time_string = dt.datetime.now().strftime("%H:%M:%S")
    if save_params:
        filenames = [f"{folder}/weights_1.npy", f"{folder}/weights_2.npy"]
        save_params_fn(params, filenames)

    # generate plots
    if plot:
        plt_and_save(
            folder,
            testset,
            recording,  # type: ignore
            t_spike,  # type: ignore
            params_over_time,
            loss,  # type: ignore
            acc,  # type: ignore
            p.tau_syn,  # type: ignore
            hidden_size,
            epochs,
            duplication,
            duplicate_neurons,
        )

    # save experiment data
    max_acc = round(np.max(acc).item(), 3)  # type: ignore
    log.INFO(f"Max acc: {max_acc} after {np.argmax(acc)} epochs")  # type: ignore
    experiment = {
        "mock_hw": MOCK_HW,
        "max_accuracy": max_acc,
        "seed": seed,
        "epochs": epochs,
        "tau_mem": p.tau_mem,
        "tau_syn": p.tau_syn,
        "v_th": p.v_th,
        "v_reset": p.v_reset,
        "t_late": t_late,
        "bias_spike (tau_syn)": round(bias_spike / p.tau_syn, 4)
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
            round(float(correct_target_time) / p.tau_syn, 4),
            round(float(wrong_target_time) / p.tau_syn, 4),
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
    log.INFO(f"Running experiment, results in folder: {folder}")

    hxtorch.init_hardware()
    train(0, folder, plot=True, print_epoch=True, save_params=True)
    hxtorch.release_hardware()

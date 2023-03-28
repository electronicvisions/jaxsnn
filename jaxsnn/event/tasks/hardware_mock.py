import datetime as dt
import json
import time
from functools import partial
from pathlib import Path
from typing import List, Tuple
import jax
import numpy as onp
import jax.numpy as np
import optax
from jax import random

from jaxsnn.base.types import Array, Spike, Weight
from jaxsnn.event.compose import serial, serial_spikes_known
from jaxsnn.event.dataset import yinyang_dataset
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.hardware.utils import (
    simulate_hw_weights,
    add_linear_noise,
    cut_spikes_batch,
    spike_similarity_batch
)
from jaxsnn.event.leaky_integrate_and_fire import (
    LIFParameters,
    RecurrentEventPropLIF,
    HardwareRecurrentLIF,
)
from jaxsnn.event.loss import (
    loss_and_acc_scan,
    loss_wrapper_known_spikes,
    mse_loss,
)
from jaxsnn.event.hardware.experiment import Experiment
from jaxsnn.event.hardware.neuron import Neuron
from jaxsnn.event.hardware.input_neuron import InputNeuron
from jaxsnn.event.plot import plt_and_save
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.utils import save_params_recurrent, load_params_recurrent
from jaxsnn.event.hardware.utils import sort_batch, filter_spikes_batch
from jaxsnn.event import custom_lax
from jax.config import config
import hxtorch
from jaxsnn.event.hardware.calib import W_69_F0_LONG_REFRAC_130_THRESHOLD

log = hxtorch.logger.get("hxtorch.snn.experiment")

config.update("jax_debug_nans", True)

wafer_config = W_69_F0_LONG_REFRAC_130_THRESHOLD

MOCK_HW = False
HW_CYCLE_CORRECTION = -50
SIM_HW_WEIGTHS = True
LOAD_FOLDER = None

def train(
    seed: int,
    folder: str,
    plot: bool = True,
    print_epoch: bool = True,
    save_params: bool = False,
):
    # neuron params, low v_reset only allows one spike per neuron
    p = LIFParameters(
        v_reset=-1000.0, v_th=1.0, tau_syn_inv=1 / 6e-6, tau_mem_inv=1 / 12e-6
    )

    # training params
    step_size = 5e-3
    lr_decay = 0.99
    train_samples = 5_000
    test_samples = 3_000
    batch_size = 64
    test_batch_size = 3_000
    epochs = 50
    n_train_batches = int(train_samples / batch_size)
    n_test_batches = int(test_samples / test_batch_size)
    t_late = 2.0 * p.tau_syn
    t_max = 4.0 * p.tau_syn

    # at least 50 us because otherwise we get jitter
    t_max_us = max(t_max / 1e-6, 50)

    # if this parameter is set, the input spikes are not duplicated over one neuron, but over multiple
    duplicate_neurons = True
    duplication = 5
    weight_mean = [3.0 / duplication, 0.5]
    weight_std = [1.6 / duplication, 0.5]

    # in units of t_late
    bias_spike = 0.5

    correct_target_time = 1.0 * p.tau_syn
    wrong_target_time = 1.5 * p.tau_syn

    log.WARN(f"Mock HW set to {MOCK_HW}, lr: {step_size}, bias spike: {bias_spike}, duplication: {duplication}, sim hw weights: {SIM_HW_WEIGTHS}")

    # net
    input_size = 4 + int(bias_spike is not None)
    if duplicate_neurons:
        input_size *= duplication

    hidden_size = 100
    output_size = 3

    # n_spikes
    n_spikes_input = 5 * duplication
    n_spikes_hidden = hidden_size
    n_spikes_output = output_size
    n_spikes = n_spikes_input + n_spikes_hidden + n_spikes_output

    optimizer_fn = optax.adam
    n_neurons = input_size + hidden_size + output_size

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

    trainset = yinyang_dataset(
        train_rng,
        t_late,
        [n_train_batches, batch_size],
        **dataset_kwargs,
    )
    testset = yinyang_dataset(
        test_rng,
        t_late,
        [n_test_batches, batch_size],
        **dataset_kwargs,
    )
    solver = partial(ttfs_solver, p.tau_mem, p.v_th)

    # recurrent Net
    _, hw_mock = serial(
        RecurrentEventPropLIF(
            layers=[hidden_size, output_size],
            n_spikes=n_spikes,
            t_max=t_max,
            p=p,
            solver=solver,
            mean=weight_mean,
            std=weight_std,
            wrap_only_step=False,
        )
    )
    hw_mock_batched = jax.jit(jax.vmap(hw_mock, in_axes=(None, 0)))

    init_fn, apply_fn = serial_spikes_known(
        HardwareRecurrentLIF(
            layers=[hidden_size, output_size],
            n_spikes=n_spikes,
            t_max=t_max,
            p=p,
            mean=weight_mean,
            std=weight_std,
            duplication=duplication if duplicate_neurons else None
        )
    )

    # init params and optimizer
    if LOAD_FOLDER is None:
        params = init_fn(param_rng, input_size)
    else:
        log.WARN(f"Loading params from folder: {LOAD_FOLDER}")
        params = [load_params_recurrent(LOAD_FOLDER)]


    scheduler = optax.exponential_decay(step_size, n_train_batches, lr_decay)

    optimizer = optimizer_fn(scheduler)
    opt_state = optimizer.init(params)

    loss_fn = jax.jit(batch_wrapper(
            partial(
                loss_wrapper_known_spikes,
                apply_fn,
                mse_loss,
                p.tau_mem,
                n_neurons,
                output_size,
            ),
            in_axes=(None, 0, 0),
        ))

    # HW
    experiment = Experiment(wafer_config)
    InputNeuron(input_size, p, experiment)
    Neuron(hidden_size, p, experiment)
    Neuron(output_size, p, experiment)


    def update(input, batch):
        opt_state, params, time_data = input
        input_spikes, _ = batch

        if MOCK_HW:
            if SIM_HW_WEIGTHS:
                hw_spikes = hw_mock_batched(simulate_hw_weights(params, wafer_config.weight_scaling), input_spikes)
            else:
                hw_spikes = hw_mock_batched(params, input_spikes)
        else:
            hw_spikes, time_data = experiment.get_hw_results(
                input_spikes,
                params,
                t_max_us,
                n_spikes=[n_spikes_hidden, n_spikes_output],
                time_data=time_data,
                hw_cycle_correction=HW_CYCLE_CORRECTION
            )

            # merge to one layer
            hw_spikes = [
                sort_batch(Spike(
                    idx=np.concatenate((hw_spikes[0].idx, hw_spikes[1].idx), axis=-1),
                    time=np.concatenate((hw_spikes[0].time, hw_spikes[1].time), axis=-1),
                ))
            ]

        # add time noise
        hw_spikes = [add_linear_noise(hw_spikes[0])]    
        return update_software((opt_state, params, time_data), batch, hw_spikes)
    
    # define test function
    def test_loss_fn(params, batch):
        input_spikes, _ = batch

        # sw_spikes = hw_mock_batched(simulate_hw_weights(params, wafer_config.weight_scaling), input_spikes)
        # sw_spikes = [cut_spikes_batch(filter_spikes_batch(sw_spikes[0], input_size), n_spikes_hidden + n_spikes_output)]
        if MOCK_HW:
            if SIM_HW_WEIGTHS:
                hw_spikes = hw_mock_batched(simulate_hw_weights(params, wafer_config.weight_scaling), input_spikes)
            else:
                hw_spikes = hw_mock_batched(params, input_spikes)

            hw_spikes = [cut_spikes_batch(filter_spikes_batch(hw_spikes[0], input_size), n_spikes_hidden + n_spikes_output)]
        else:
            hw_spikes, _ = experiment.get_hw_results(
                input_spikes,
                params,
                t_max_us,
                n_spikes=[n_spikes_hidden, n_spikes_output],
                time_data={},
                hw_cycle_correction=HW_CYCLE_CORRECTION,
            )

            # merge to one layer
            hw_spikes = [
                sort_batch(Spike(
                    idx=np.concatenate((hw_spikes[0].idx, hw_spikes[1].idx), axis=-1),
                    time=np.concatenate((hw_spikes[0].time, hw_spikes[1].time), axis=-1),
                ))
            ]

        # spike_similarity_batch(sw_spikes[0], hw_spikes[0])
    
        # add time noise to not have multiple spikes at the same time
        hw_spikes = [add_linear_noise(hw_spikes[0])]
        loss_result = loss_fn(params, batch, hw_spikes)
        return params, loss_result

    @jax.jit
    def update_software(
        input: Tuple[optax.OptState, List[Weight], dict],
        batch: Tuple[Spike, Array],
        hw_spikes
    ):
        opt_state, params, hw_time = input
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, hw_spikes
        )

        grad = jax.tree_util.tree_map(
            lambda par, g: np.where(par == 0.0, 0.0, g / p.tau_syn), params, grad
        )

        # set second layer grad to zero
        # grad = [WeightRecurrent(grad[0].input, grad[0].recurrent * 10)]
        # grad = [grad[0].input, np.zeros_like(grad[0].recurrent)]

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return (opt_state, params, hw_time), (value, grad)

    def epoch(state, i):
        # do testing before training for plot
        params = state[1]
        test_result = loss_and_acc_scan(test_loss_fn, params, testset[:2])
        loss, acc, t_first_spike, recording = test_result

        start = time.time()
        state, (_, grad) = custom_lax.simple_scan(update, state, trainset[:2])
        duration = time.time() - start
        if print_epoch:
            masked = onp.ma.masked_where(t_first_spike == np.inf, t_first_spike)
            number_of_hidden_spikes = np.sum(input_size <= recording[0].idx, axis=-1).mean()
            input_param = params[0].input[:,:hidden_size]
            recurrent_param = params[0].recurrent[:hidden_size,hidden_size:]
            log.INFO(
                f"Epoch {i}, loss: {loss:.6f}, "
                f"acc: {acc:.3f}, "
                f"spikes: {number_of_hidden_spikes:.1f}, "
                f"output inf: {np.mean((t_first_spike == np.inf), axis=(0, 1))}, "
                f"grad: {grad[0].input[:, :,:hidden_size].mean():.4f}, {grad[0].recurrent[:, :hidden_size,hidden_size:].mean():.8f}, ",
                f"params mean: {input_param.mean():.5f}, {recurrent_param.mean():.5f}, ",
                f"params std: {input_param.std():.5f}, {recurrent_param.std():.5f}, ",
                f"param sat: {np.abs(input_param * wafer_config.weight_scaling > 63).mean():.3f}, {np.abs(recurrent_param * wafer_config.weight_scaling > 63).mean():.3f}, "
                f"time output: {(masked.mean() / p.tau_syn):.2f} tau_s, "
            #     f"in {duration:.2f} s, ",
            #      f"hw time: {state[2].get('get_hw_results', 0.0):.2f} s, ",
            #     f"grenade run time: {state[2].get('grenade_run', 0.0):.2f} s",
            )
        return state, (test_result, params, duration)

    # train the net
    (opt_state, params, timing), (res, params_over_time, durations) = custom_lax.simple_scan(
        epoch, (opt_state, params, {}), np.arange(epochs)
    )
    loss, acc, t_spike, recording = res  # type: ignore

    time_string = dt.datetime.now().strftime("%H:%M:%S")
    Path(folder).mkdir(parents=True, exist_ok=True)
    if save_params:
        save_params_recurrent(params[0], folder)

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
            MOCK_HW,
        )

    # save experiment data
    max_acc = round(np.max(acc).item(), 3)  # type: ignore
    print(f"Max acc: {max_acc} after {np.argmax(acc)} epochs")  # type: ignore
    experiment = {
        "config": wafer_config.file,
        "weight_mapping": wafer_config.weight_scaling,
        "mock_hw": MOCK_HW,
        "max_accuracy": max_acc,
        "seed": seed,
        "epochs": epochs,
        "tau_mem": p.tau_mem,
        "tau_syn": p.tau_syn,
        "v_th": p.v_th,
        "v_reset": p.v_reset,
        "t_late": t_late,
        "bias_spike (t_late)": bias_spike,
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
    folder = f"jaxsnn/plots/hardware/yinyang_{'mock' if MOCK_HW else 'no_mock'}/{dt_string}"
    print(f"Running experiment, results in folder: {folder}")
    hxtorch.init_hardware()
    train(0, folder, plot=True, print_epoch=True, save_params=False)
    hxtorch.release_hardware()





 # non recurrent version
    # _, hw_mock = serial(
    #     EventPropLIF(
    #         n_hidden=hidden_size,
    #         n_spikes=n_spikes_input + n_spikes_hidden,
    #         t_max=t_max,
    #         p=p,
    #         solver=solver,
    #         duplication=duplication if duplicate_neurons else None
    #     ),
    #     EventPropLIF(
    #         n_hidden=output_size,
    #         n_spikes=n_spikes_input + n_spikes_hidden + n_spikes_output,
    #         t_max=t_max,
    #         p=p,
    #         solver=solver,
    #     ),
    # )
    # hw_mock_batched = jax.jit(jax.vmap(hw_mock, in_axes=(None, 0)))

    # init_fn, apply_fn = serial_spikes_known(
    #     HardwareLIF(
    #         n_hidden=hidden_size,
    #         n_spikes=n_spikes_input + n_spikes_hidden,
    #         t_max=t_max,
    #         p=p,
    #         mean=weight_mean[0],
    #         std=weight_std[0],
    #         duplication=duplication if duplicate_neurons else None
    #     ),
    #     HardwareLIF(
    #         n_hidden=output_size,
    #         n_spikes=n_spikes_input + n_spikes_hidden + n_spikes_output,
    #         t_max=t_max,
    #         p=p,
    #         mean=weight_mean[1],
    #         std=weight_std[1],
    #     ),
    # )

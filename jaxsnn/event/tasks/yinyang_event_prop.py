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
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset import yinyang_dataset
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.leaky_integrate_and_fire import (
    LIFParameters,
    RecurrentEventPropLIF,
)
from jaxsnn.event.loss import (
    loss_and_acc,
    loss_wrapper,
    mse_loss,
)
from jaxsnn.event.plot import plt_and_save
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.utils import save_params as save_params_fn
from jaxsnn.event import custom_lax
from jax.config import config

config.update("jax_debug_nans", True)


def train(
    seed: int,
    folder: str,
    plot: bool = True,
    print_epoch: bool = True,
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
    epochs = 300
    n_train_batches = int(train_samples / batch_size)
    n_test_batches = int(test_samples / batch_size)
    t_late = 2.0 * p.tau_syn
    t_max = 4.0 * p.tau_syn
    weight_mean = [3.0, 0.5]
    weight_std = [1.6, 0.8]
    bias_spike = 0.9 * p.tau_syn

    correct_target_time = 0.9 * p.tau_syn
    wrong_target_time = 1.5 * p.tau_syn

    # net
    input_size = 5
    hidden_size = 100
    output_size = 3
    n_spikes_hidden = input_size + hidden_size
    n_spikes_output = n_spikes_hidden + 3
    optimizer_fn = optax.adam

    rng = random.PRNGKey(seed)
    param_rng, train_rng, test_rng = random.split(rng, 3)
    trainset = yinyang_dataset(
        train_rng,
        t_late,
        [n_train_batches, batch_size],
        mirror=True,
        bias_spike=bias_spike,
        correct_target_time=correct_target_time,
        wrong_target_time=wrong_target_time,
    )
    testset = yinyang_dataset(
        test_rng,
        t_late,
        [n_test_batches, batch_size],
        mirror=True,
        bias_spike=bias_spike,
        correct_target_time=correct_target_time,
        wrong_target_time=wrong_target_time,
    )
    input_size = trainset[0].idx.shape[-1]
    solver = partial(ttfs_solver, p.tau_mem, p.v_th)

    # declare net
    init_fn, apply_fn = serial(
        RecurrentEventPropLIF(
            layers=[hidden_size, output_size],
            n_spikes=n_spikes_output,
            t_max=t_max,
            p=p,
            solver=solver,
            mean=weight_mean,
            std=weight_std,
            wrap_only_step=False,
        )
    )

    # init params and optimizer
    params = init_fn(param_rng, input_size)

    n_neurons = params[0].input.shape[0] + hidden_size + output_size

    scheduler = optax.exponential_decay(step_size, n_train_batches, lr_decay)

    optimizer = optimizer_fn(scheduler)
    opt_state = optimizer.init(params)

    loss_fn = batch_wrapper(
        partial(loss_wrapper, apply_fn, mse_loss, p.tau_mem, n_neurons, output_size)
    )

    def update(
        input: Tuple[optax.OptState, List[Weight]],
        batch: Tuple[Spike, Array],
    ):
        opt_state, params = input
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)

        grad = jax.tree_util.tree_map(
            lambda par, g: np.where(par == 0.0, 0.0, g / p.tau_syn), params, grad
        )

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), (value, grad)

    def epoch(state, i):
        # do testing before training for plot
        params = state[1]
        test_result = loss_and_acc(loss_fn, params, testset[:2])

        start = time.time()
        state, (recording, grad) = jax.lax.scan(update, state, trainset[:2])
        duration = time.time() - start

        if print_epoch:
            masked = onp.ma.masked_where(recording[1][0] == np.inf, recording[1][0])
            jax.debug.print(
                "Epoch {i}, loss: {loss:.6f}, acc: {acc:.3f}, spikes: {spikes:.1f}, grad: {grad:.9f}, gradient ratio: {grad_ratio:.4f}, time first output: {t_output:.2f} tau_s, in {duration:.2f} s",
                i=i,
                loss=round(test_result[0], 3),
                acc=round(test_result[1], 3),
                spikes=np.sum(recording[1][1][0].idx >= 0, axis=-1).mean(),
                grad=grad[0].input.mean(),
                grad_ratio=np.abs(grad[0].input).mean()
                / (
                    np.sum(np.abs(grad[0].recurrent))
                    / np.count_nonzero(grad[0].recurrent)
                ),
                t_output=masked.mean() / p.tau_syn,
                duration=duration,
            )
        return state, (test_result, params, duration)

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
        )

    # save experiment data
    max_acc = round(np.max(acc).item(), 3)  # type: ignore
    print(f"Max acc: {max_acc} after {np.argmax(acc)} epochs")  # type: ignore
    experiment = {
        "max_accuracy": max_acc,
        "seed": seed,
        "epochs": epochs,
        "tau_mem": p.tau_mem,
        "tau_syn": p.tau_syn,
        "v_th": p.v_th,
        "v_reset": p.v_reset,
        "t_late": t_late,
        "bias_spike (tau_syn)": round(bias_spike / p.tau_syn, 4),
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
    folder = f"jaxsnn/plots/event/yinyang/{dt_string}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"Running experiment, results in folder: {folder}")
    train(0, folder, plot=True, print_epoch=True, save_params=True)

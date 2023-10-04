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
from jaxsnn.event.leaky_integrate_and_fire import LIFParameters, LIF, RecurrentLIF
from jaxsnn.event.loss import (
    loss_and_acc,
    loss_wrapper,
    mse_loss,
    adapted_ttfs_loss,
    adapted_event_prop_loss,
)
from jaxsnn.event.plot import plt_and_save
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.utils import save_params as save_params_fn
from jaxsnn.event import custom_lax

# config.update("jax_debug_nans", True)


def train(
    seed: int,
    folder: str,
    plot: bool = True,
    print_epoch: bool = True,
    save_params: bool = False,
    correct_target_time: float = 0.9,
    wrong_target_time: float = 1.1,
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
    t_late = 2.0 * p.tau_syn
    t_max = 4.0 * p.tau_syn
    weight_mean = [3.0, 0.5]
    weight_std = [1.6, 0.8]

    # in units of t_late
    bias_spike = 0.0

    correct_target_time *= p.tau_syn
    wrong_target_time *= p.tau_syn

    # net
    input_size = 5
    hidden_size = 100
    output_size = 3
    n_spikes_hidden = 50  # input_size + hidden_size
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

    # init_fn, apply_fn = serial(
    #     RecurrentLIF(
    #         layers=[hidden_size, output_size],
    #         n_spikes=n_spikes_output,
    #         t_max=t_max,
    #         p=p,
    #         solver=solver,
    #         mean=weight_mean,
    #         std=weight_std,
    #     )
    # )

    # declare net
    init_fn, apply_fn = serial(
        LIF(
            hidden_size,
            n_spikes=n_spikes_hidden,
            t_max=t_max,
            p=p,
            solver=solver,
            mean=weight_mean[0],
            std=weight_std[0],
        ),
        LIF(
            output_size,
            n_spikes=n_spikes_output,
            t_max=t_max,
            p=p,
            solver=solver,
            mean=weight_mean[1],
            std=weight_std[1],
        ),
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
        state: Tuple[optax.OptState, List[Weight]],
        batch: Tuple[Spike, Array],
    ):
        opt_state, params = state

        # value = loss_fn(params, batch)
        # grad = params
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)

        grad = jax.tree_util.tree_map(lambda g: g / p.tau_syn, grad)

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
                grad_ratio=1,  # np.abs(grad[0].input).mean() / np.abs(grad[0].input).mean()
                # / np.maximum(np.abs(grad[1].input), 1e-7).mean(),
                # / np.maximum(np.abs(grad[0].recurrent), 1e-7).mean(),
                t_output=masked.mean() / p.tau_syn,
                duration=duration,
            )
        return state, (test_result, params, duration)

    # train the net
    (opt_state, params), (res, params_over_time, durations) = custom_lax.simple_scan(
        epoch, (opt_state, params), np.arange(epochs)
    )
    loss, acc, t_spike, recording = res

    # last_epoch_first_batch = recording[-1][-1, 0]
    # np.save(f"{folder}/t_spike.npy", t_spike, allow_pickle=False)
    # np.save(f"{folder}/recording_last_epoch_first_batch.npy", (last_epoch_first_batch.time, last_epoch_first_batch.idx), allow_pickle=False)

    time_string = dt.datetime.now().strftime("%H:%M:%S")
    if save_params:
        filenames = [f"{folder}/weights_1.npy", f"{folder}/weights_2.npy"]
        save_params_fn(params, filenames)

    # generate plots
    if plot:
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
            mock_hw=True,
        )

    # save experiment data
    max_acc = round(np.max(acc).item(), 3)
    print(f"Max acc: {max_acc} after {np.argmax(acc)} epochs")
    print(f"Last acc: {acc[-1]}")
    experiment = {
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
        "loss": [round(float(l), 5) for l in loss],
        "accuracy": [round(float(a), 5) for a in acc],
        "time per epoch": [round(float(d), 3) for d in durations],
    }

    with open(f"{folder}/params_{max_acc}_{time_string}.json", "w") as outfile:
        json.dump(experiment, outfile, indent=4)

    return acc[-1]


if __name__ == "__main__":
    dt_string = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = f"jaxsnn/plots/event/yinyang/{dt_string}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"Running experiment, results in folder: {folder}")
    # for i in range(5):
    for i in np.arange(0.6, 1.6, 0.1):
        accs = []
        for j in np.arange(1.7, 2.6, 0.1):
            print(f"Target times: {i}, {j}")
            print()
            acc = train(
                0,
                folder,
                plot=False,
                print_epoch=True,
                save_params=False,
                correct_target_time=i,
                wrong_target_time=j,
            )
            accs.append(acc)
        print(accs)

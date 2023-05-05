from jaxsnn.event.hardware.experiment import Experiment
import time
from jaxsnn.event.hardware.neuron import Neuron
from jaxsnn.event.hardware.input_neuron import InputNeuron
from jax import random
from functools import partial
import jax.numpy as np
import numpy as onp
from jaxsnn.base.types import Spike
from jaxsnn.event.leaky_integrate_and_fire import LIFParameters
from jaxsnn.event.plot import plt_and_save
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.dataset import linear_dataset
from jaxsnn.event.compose import serial_spikes_known
import optax
from typing import Tuple, List
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.base.types import Array, Spike, Weight
from jaxsnn.event import custom_lax
from jaxsnn.event.loss import first_spike
import hxtorch
import jax
from jaxsnn.event.compose import serial
from jaxsnn.event.leaky_integrate_and_fire import (
    LIFParameters,
    HardwareRecurrentLIF,
    HardwareLIF,
    EventPropLIF
)
from jaxsnn.event.loss import (
    loss_wrapper_known_spikes,
    mse_loss,
)

log = hxtorch.logger.get("hxtorch.snn.experiment")

# calib_path = "jaxsnn/event/hardware/calib/calibration_W69F0_leak80_th150_reset80_taus-6us_taum-6us_trefrac-2.0us_isyningm-500_sdbias-1000.pbin"
calib_path = "jaxsnn/event/hardware/calib/calibration_W66F3_leak80_th150_reset80_taus-6us_taum-6us_trefrac-2.0us_isyningm-500_sdbias-1000.pbin"

folder = "jaxsnn/plots/hardware/linear"

def train():

    # neuron params, low v_reset only allows one spike per neuron
    p = LIFParameters(v_reset=-0.0, v_th=1.0, tau_syn_inv=1.0 / 6e-6, tau_mem_inv=1.0 / 6e-6)

    # training params
    seed = 42
    learning_rate = 5e-3
    lr_decay = 0.99
    train_samples = 1_000
    test_samples = 1_000
    batch_size = 64
    epochs = 20
    n_train_batches = int(train_samples / batch_size)
    n_test_batches = int(test_samples / batch_size)
    t_late = 2.0 * p.tau_syn
    t_max = 4.0 * p.tau_syn
    t_max_us = t_max / 1e-6

    weight_mean = 3.0
    weight_std = 0.5
    bias_spike = 0.9 * p.tau_syn

    correct_target_time = 0.9 * p.tau_syn
    wrong_target_time = 1.5 * p.tau_syn

    # net
    duplication = 1
    input_size = 5 * duplication
    output_size = 2

    n_neurons = input_size + output_size

    # n spikes
    n_spikes = input_size + 2

    dataset_kwargs =  {
        "mirror": True,
        "bias_spike": bias_spike,
        "correct_target_time": correct_target_time,
        "wrong_target_time": wrong_target_time,
        "duplication": duplication,
    }

    rng = random.PRNGKey(seed)
    trainset = linear_dataset(
        rng,
        t_late,
        [n_train_batches, batch_size],
        **dataset_kwargs
    )
    testset = linear_dataset(
        rng,
        t_late,
        [test_samples],
        **dataset_kwargs
    )


    init_fn, apply_fn = serial_spikes_known(
        HardwareLIF(
            output_size,
            # layers=[output_size],
            n_spikes=n_spikes,
            t_max=t_max,
            p=p,
            mean=weight_mean,
            std=weight_std,
        )
    )

    # weights = [WeightInput(random.uniform(rng, (hidden_size, input_size)))]
    params = init_fn(rng, input_size)

    # init optimizer
    optimizer_fn = optax.adam
    scheduler = optax.exponential_decay(learning_rate, n_train_batches, lr_decay)
    optimizer = optimizer_fn(scheduler)
    opt_state = optimizer.init(params)
    state = (opt_state, params)

    ###
    ### MOCK MODE
    _, apply_fn_hardware = serial(
        EventPropLIF(
            n_hidden=output_size,
            n_spikes=n_spikes,
            t_max=t_max,
            p=p,
            solver=partial(ttfs_solver, p.tau_mem, p.v_th),
            mean=weight_mean,
            std=weight_std,
        )
    )
    ###

    # set up experiment and define neurons
    experiment = Experiment(calib_path=calib_path)
    InputNeuron(input_size, p, experiment)
    Neuron(output_size, p, experiment)

    train_loss_fn = batch_wrapper(
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

    @jax.jit
    def update_software(
        input: Tuple[optax.OptState, List[Weight]],
        batch: Tuple[Spike, Array],
        hw_spikes: List[Spike],
    ):
        opt_state, params = input
        value, grad = jax.value_and_grad(train_loss_fn, has_aux=True)(
            params, batch, hw_spikes
        )

        # TODO do we need scaling with tau syn?
        grad = jax.tree_util.tree_map(
            lambda par, g: np.where(par == 0.0, 0.0, g / p.tau_syn), params, grad
        )
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), (value, grad)


    def update(input, batch):
        input_spikes, _ = batch
        # hw_spikes = experiment.get_hw_results(input_spikes, params, t_max_us, n_spikes)
        hw_spikes = jax.vmap(apply_fn_hardware, in_axes=(None, 0))(params, input_spikes)
        res = update_software(input, batch, hw_spikes)
        return res

    def test(testset):
        input_spikes, target = testset
        # hw_spikes = experiment.get_hw_results(input_spikes, params, t_max_us, n_spikes)
        hw_spikes = jax.vmap(apply_fn_hardware, in_axes=(None, 0))(params, input_spikes)
        loss, accuracy, t_first_spike = jax.vmap(loss_and_acc, in_axes=(0, 0))(hw_spikes[-1], target)
        return np.mean(loss), np.mean(accuracy), t_first_spike

    def loss_and_acc(spikes, target):
        t_first_spike = first_spike(spikes, n_neurons)[-output_size:]
        loss = mse_loss(t_first_spike, target, p.tau_mem)
        accuracy = np.argmin(testset[1], axis=-1) == np.argmin(t_first_spike, axis=-1)
        return loss, accuracy, t_first_spike


    def epoch(state, i):
        start = time.time()
        state, (recording, grad) = custom_lax.simple_scan(update, state, trainset[:2])
        duration = time.time() - start
        loss, acc, t_first_spike = test(testset[: 2])
        opt_state, params = state

        masked = onp.ma.masked_where(recording[1][0] == np.inf, recording[1][0])
        jax.debug.print(
            "Epoch {i}, train loss: {train_loss:.6f}, test loss: {loss:.6f}, test acc: {acc:.3f}, spikes: {spikes:.1f}, weights: {weights} abs weight: {abs_weights}, grad: {grad:.9f}, time first output: {t_output:.2f} tau_s, in {duration:.2f} s",
            i=i,
            train_loss=np.mean(recording[0]),
            loss=round(loss, 3),
            acc=round(acc, 3),
            spikes=np.sum(recording[1][1][0].idx >= 0, axis=-1).mean(),
            weights=np.mean(params[0].input),
            abs_weights=np.mean(np.abs(params[0].input)),
            grad=grad[0].input.mean(),
            t_output=masked.mean() / p.tau_syn,
            duration=duration,
        )
        return state, (t_first_spike, params, loss, acc)

    state, (t_first_spike, params_over_time, loss, acc) = custom_lax.simple_scan(epoch, state, np.arange(epochs))

    plt_and_save(folder, testset, None, t_first_spike, params_over_time, loss, acc, p.tau_syn, None, epochs, duplication)

    # n_output = testset[1].shape[-1]
    # fig, ax1 = plt.subplots(1, n_output, figsize=(5 * n_output, 4))

    # plt_t_spike_neuron(fig, ax1, testset, t_first_spike, p.tau_syn, duplication=duplication)
    # fig.savefig(f"{folder}/spike_times.png", dpi=150)

    # # # weights
    # fig, axs = plt.subplots(3, 1, figsize=(7, 7))
    # plt_weights(fig, axs, params_over_time)
    # fig.savefig(f"{folder}/weights_over_time.png", dpi=150)

    # # classification
    # fig, axs = plt.subplots(1, 2, figsize=(7.5, 4))
    # plt_dataset(axs[0], testset, p.tau_syn, duplication=duplication)
    # plt_prediction(axs[1], testset, t_first_spike, p.tau_syn, duplication=duplication)
    # fig.tight_layout()
    # fig.savefig(f"{folder}/classification.png", dpi=150)



if __name__ == "__main__":
    hxtorch.init_hardware()
    train()
    hxtorch.release_hardware()

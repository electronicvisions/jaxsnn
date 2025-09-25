"""Train a two-layer feed-forward network with spike data from the BSS-2
system (Hardware-in-the-loop-training). A second forward run in software is
conducted to add missing information about the synaptic current at spike time.
In this run, the spikes from BSS-2 are used as solution for the root-solving.

Once the information about the synaptic current is also returned with the
event-based observations from BSS-2, this second forward pass in software
can be emitted.
"""
from typing import Callable, List, Tuple
import argparse
from functools import partial

import hxtorch
import jax
import jax.numpy as jnp
import optax
from jax import random
import jaxsnn
from jaxsnn.base.compose import serial
from jaxsnn.base.dataset import yinyang_dataset, data_loader
from jaxsnn.event import custom_lax
from jaxsnn.event.encode import (
    spatio_temporal_encode, target_temporal_encode, encode)
from jaxsnn.event.hardware.experiment import Experiment
from jaxsnn.event.hardware.input_neuron import InputNeuron
from jaxsnn.event.hardware.neuron import Neuron
from jaxsnn.event.hardware.utils import add_linear_noise, sort_batch
from jaxsnn.event.modules.leaky_integrate_and_fire import (
    HardwareRecurrentLIF, LIFParameters)
from jaxsnn.event.loss import mse_loss
from jaxsnn.event.types import Spike, Weight, EventPropSpike
from jaxsnn.event.hardware.calib import WaferConfig
from jaxsnn.examples.event.utils import loss_wrapper


def loss_and_acc_scan(
    loss_fn: Callable,
    weights: List[Weight],
    dataset: Tuple[EventPropSpike, jax.Array],
) -> Tuple:
    weights, (loss, (t_first_spike, recording)) = custom_lax.scan(
        loss_fn, weights, dataset
    )
    accuracy = jnp.argmin(dataset[1], axis=-1) == jnp.argmin(
        t_first_spike, axis=-1
    )
    return jnp.mean(loss), jnp.mean(accuracy), t_first_spike, recording


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="hxtorch spiking YinYang example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--duplicate-neurons", action="store_true", default=True)
    parser.add_argument(
        "--plot", action="store_true", default=True)
    # data
    parser.add_argument("--testset-size", type=int, default=2944)
    parser.add_argument("--trainset-size", type=int, default=4992)
    # encoding
    parser.add_argument("--t-late", type=float, default=2.0 * 6e-6)
    parser.add_argument("--correct-target-time", type=float, default=0.9 * 6e-6)
    parser.add_argument("--wrong-target-time", type=float, default=1.1 * 6e-6)
    # model
    parser.add_argument("--tau-mem", type=float, default=12e-6)
    parser.add_argument("--tau-syn", type=float, default=6e-6)
    # training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="<num samples>",
        help="input batch size for training")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    # hw
    parser.add_argument("--duplication", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--hw-correction", type=int, default=-20)
    parser.add_argument("--max-runtime", type=int, default=50)
    parser.add_argument("--weight-scale", type=float, default=43)
    parser.add_argument("--calib-path", type=str)
    parser.add_argument("--calib-name", type=str)
    return parser


def main(args: argparse.Namespace):
    log = jaxsnn.get_logger("jaxsnn.examples.event.hardware.yinyang")

    # neuron params, low v_reset only allows one spike per neuron
    params = LIFParameters(
        v_reset=-1000.0, v_th=1.0, tau_syn=args.tau_syn, tau_mem=args.tau_mem)

    n_train_batches = args.trainset_size // args.batch_size
    n_test_batches = args.testset_size // args.batch_size
    train_samples = args.batch_size * n_train_batches
    test_samples = args.batch_size * n_test_batches

    t_max = 4.0 * params.tau_syn
    runtime = int(max(t_max / 1e-6, args.max_runtime))

    # How many input neurons do we have?
    input_size = 5
    output_size = 3
    if args.duplicate_neurons:
        input_size *= args.duplication

    # Define trainset and testset
    rng = random.PRNGKey(args.seed)
    param_rng, train_rng, test_rng = random.split(rng, 3)
    xy_trainset = yinyang_dataset(
        train_rng, train_samples, mirror=True, bias_spike=0.0)
    xy_testset = yinyang_dataset(
        test_rng, test_samples, mirror=True, bias_spike=0.0)

    # Encoding
    input_encoder_batched = jax.jit(jax.vmap(partial(
        spatio_temporal_encode,
        t_late=args.t_late,
        duplication=args.duplication,
        duplicate_neurons=args.duplicate_neurons)))
    target_encoder_batched = jax.jit(jax.vmap(partial(
        target_temporal_encode,
        n_classes=output_size,
        correct_target_time=args.correct_target_time,
        wrong_target_time=args.wrong_target_time)))

    # Datasets
    trainset = encode(
        xy_trainset, input_encoder_batched, target_encoder_batched)
    testset = encode(
        xy_testset, input_encoder_batched, target_encoder_batched)

    # Software net which adds the current in a second pass
    # and calculates the gradients with EventProp
    init_fn, apply_fn = serial(
        HardwareRecurrentLIF(
            layers=[args.hidden_size, output_size],
            n_spikes=input_size + args.hidden_size + output_size,
            t_max=t_max,
            params=params,
            mean=[3.0 / args.duplication, 0.5],
            std=[1.6 / args.duplication, 0.8],
            duplication=args.duplication if args.duplicate_neurons else None))

    _, weights = init_fn(param_rng, input_size)

    # Optimizer
    scheduler = optax.exponential_decay(
        args.lr, n_train_batches, args.lr_decay)
    optimizer = optax.chain(
        optax.clip(0.01),
        optax.adam(scheduler))
    opt_state = optimizer.init(weights)

    # define loss and update function
    loss_fn = jax.jit(partial(
        loss_wrapper, apply_fn, mse_loss, params.tau_mem,
        input_size + args.hidden_size + 3, 3))

    # set up neurons on BSS-2
    experiment = Experiment(
        WaferConfig(args.calib_path, args.calib_name, args.weight_scale))
    InputNeuron(input_size, params, experiment)
    Neuron(args.hidden_size, params, experiment)
    Neuron(3, params, experiment)

    @jax.jit
    def update_software(input, batch, hw_spikes):
        opt_state, weights = input
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(
            weights, batch, external=hw_spikes)
        # Kill recurrent grads
        grad = jax.tree_util.tree_map(
            lambda par, g: jnp.where(par == 0.0, 0.0, g), weights, grad)

        # Update weights
        updates, opt_state = optimizer.update(grad, opt_state, params=weights)
        weights = optax.apply_updates(weights, updates)

        return (opt_state, weights), (value, grad)

    def train_fn(input, batch):
        opt_state, weights, rng = input
        input_spikes, _ = batch

        hw_spikes, _ = experiment.get_hw_results(
            input_spikes, weights, runtime, n_spikes=[args.hidden_size, 3],
            time_data={}, hw_cycle_correction=args.hw_correction)

        # merge to one layer
        hw_spikes = [
            sort_batch(Spike(
                idx=jnp.concatenate(
                    (hw_spikes[0].idx, hw_spikes[1].idx), axis=-1),
                time=jnp.concatenate(
                    (hw_spikes[0].time, hw_spikes[1].time), axis=-1)))
        ]

        # add time noise, this is necessary to not have two spikes at the
        # same time
        hw_spikes = [add_linear_noise(hw_spikes[0])]
        (opt_state, weights), (value, grad) = update_software(
            (opt_state, weights), batch, hw_spikes)
        return (opt_state, weights, rng), (value, grad)

    def test_fn(weights, batch):
        input_spikes, _ = batch

        hw_spikes, _ = experiment.get_hw_results(
            input_spikes, weights, runtime, n_spikes=[args.hidden_size, 3],
            time_data={}, hw_cycle_correction=args.hw_correction)

        # merge to one layer
        hw_spikes = [
            sort_batch(Spike(
                idx=jnp.concatenate(
                    (hw_spikes[0].idx, hw_spikes[1].idx), axis=-1),
                time=jnp.concatenate(
                    (hw_spikes[0].time, hw_spikes[1].time), axis=-1)))
        ]

        # add time noise to not have multiple spikes at the same time
        hw_spikes = [add_linear_noise(hw_spikes[0])]
        loss_result = loss_fn(weights, batch, external=hw_spikes)

        return weights, loss_result

    def epoch(state, i):
        opt_state, weights, rng = state
        test_rng, train_rng, perm_rng, rng = jax.random.split(rng, 4)
        # Train
        trainset_batched = data_loader(
            trainset, args.batch_size, n_train_batches, rng=perm_rng)
        (opt_state, weights, _), _ = custom_lax.scan(
            train_fn, (opt_state, weights, train_rng), trainset_batched)
        # Test
        testset_batched = data_loader(
            testset, args.batch_size, n_test_batches, rng=test_rng)
        loss, acc, t_first_spike, _ = loss_and_acc_scan(
            test_fn, weights, testset_batched)
        # New state
        state = (opt_state, weights, rng)
        log.INFO(f"Epoch {i} - Loss: {loss:.4f} - Acc: {acc:.4f} - ")
        return state, ((loss, acc), weights, t_first_spike)

    # train the net
    res = custom_lax.scan(
        epoch, (opt_state, weights, rng), jnp.arange(args.epochs))
    _, ((losses, accuracies), weights, t_first_spike) = res

    # find best epoch
    acc = round(accuracies[-1], 3)
    log.INFO(f"Max acc: {acc} after {args.epochs} epochs")

    return acc


if __name__ == "__main__":
    hxtorch.init_hardware()
    main(get_parser().parse_args())
    hxtorch.release_hardware()

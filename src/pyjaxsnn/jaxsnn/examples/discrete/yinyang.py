import argparse
import time
from functools import partial

import jax
import optax
import jax.numpy as jnp
from jax import random
import jaxsnn
from jaxsnn.base.compose import serial
from jaxsnn.base.dataset import yinyang_dataset, data_loader
from jaxsnn.base.params import LIFParameters
from jaxsnn.discrete.modules.leaky_integrate import LI
from jaxsnn.discrete.modules.leaky_integrate_and_fire import LIF
from jaxsnn.discrete.decode import max_over_time_decode
from jaxsnn.discrete.encode import spatio_temporal_encode
from jaxsnn.discrete.loss import nll_loss, acc_and_loss


log = jaxsnn.get_logger("jaxsnn.examples.discrete.yinyang")


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="hxtorch spiking YinYang example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    # data
    parser.add_argument("--testset-size", type=int, default=2944)
    parser.add_argument("--trainset-size", type=int, default=4992)
    # model
    parser.add_argument("--tau-mem", type=float, default=1e-2)
    parser.add_argument("--tau-syn", type=float, default=5e-3)
    parser.add_argument("--v_th", type=float, default=0.6)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=120)
    # training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="<num samples>",
        help="input batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-decay", type=float, default=0.98)
    parser.add_argument("--expected-spikes", type=float, default=0.8)
    return parser


def train_step(optimizer, state, batch, loss_fn):
    """A single step in the training process.

    1. Run the sample and calculate the gradient
    2. Calculate parameter updates
    3. Update the parameters

    The step function is compliant with the syntax of `jaxlax.scan`,
    making it easy to be looped over.
    """
    opt_state, weights, i = state
    inputs, output = batch

    (loss, recording), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        weights, (inputs, output), max_over_time_decode)
    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)
    return (opt_state, weights, i + 1), recording


def main(args: argparse.Namespace):
    params = LIFParameters(
        tau_syn=args.tau_syn, tau_mem=args.tau_mem, v_th=args.v_th)

    n_train_batches = args.trainset_size // args.batch_size
    n_test_batches = args.testset_size // args.batch_size
    train_samples = args.batch_size * n_train_batches
    test_samples = args.batch_size * n_test_batches

    t_late = params.tau_syn + params.tau_mem
    time_steps = int(2 * t_late / args.dt)
    log.info(f"dt: {args.dt}, {time_steps} time steps, t_late: {t_late}")

    # Define RNGs
    rng = random.PRNGKey(args.seed)
    init_rng, train_rng, test_rng, shuffle_rng = random.split(rng, 4)

    # Setting up trainset and testset
    xy_trainset = yinyang_dataset(
        train_rng, train_samples, mirror=True, bias_spike=0.0)
    xy_testset = yinyang_dataset(
        test_rng, test_samples, mirror=True, bias_spike=0.0)

    # Encoding the inputs
    time_steps_encoding = int(time_steps * 2 / 3)
    input_encoder_batched = jax.vmap(
        spatio_temporal_encode, in_axes=(0, None, None, None))

    train_input_encoded = input_encoder_batched(
        xy_trainset[0], time_steps_encoding, t_late, args.dt)
    trainset = (train_input_encoded, xy_trainset[1])

    test_input_encoded = spatio_temporal_encode(
        xy_testset[0], time_steps_encoding, t_late, args.dt)
    testset = (test_input_encoded, xy_testset[1])

    # define the network
    snn_init, snn_apply = serial(LIF(args.hidden_size), LI(3))

    # define optimizer
    scheduler = optax.exponential_decay(
        args.lr, n_train_batches, args.lr_decay)
    optimizer = optax.adam(scheduler)

    # define loss and train function
    loss_fn = partial(
        nll_loss, snn_apply, expected_spikes=args.expected_spikes, rho=1e-5)
    train_step_fn = partial(train_step, optimizer, loss_fn=loss_fn)

    overall_time = time.time()
    _, weights = snn_init(init_rng, input_size=5)
    opt_state = optimizer.init(weights)

    accuracies, loss = [], []
    for epoch in range(args.epochs):
        start = time.time()
        # Generate randomly shuffled batches
        this_shuffle_rng, shuffle_rng = random.split(shuffle_rng)
        trainset_batched = data_loader(trainset, 64, this_shuffle_rng)

        # Swap axes because time axis needs to come before batch axis
        trainset_batched = (
            jnp.swapaxes(trainset_batched[0], 1, 2),
            trainset_batched[1])
        (opt_state, weights, i), recording = jax.lax.scan(
            train_step_fn, (opt_state, weights, 0), trainset_batched)
        end = time.time() - start

        spikes_per_item = jnp.count_nonzero(recording[0].z) / train_samples
        accuracy, test_loss = acc_and_loss(
            snn_apply,
            weights,
            (testset[0], testset[1]),
            max_over_time_decode)

        accuracies.append(accuracy)
        loss.append(test_loss)

        log.info(
            f"Epoch: {epoch}, Loss: {test_loss:3f}, "
            + f"Test accuracy: {accuracy:.3f}, Seconds: {end:.3f}, "
            + f"Spikes: {spikes_per_item:.1f}")

    acc = round(accuracies[-1], 3)
    log.info(f"Acc: {acc} after {args.epochs} epochs")
    log.info(
        f"Finished {args.epochs} epochs in {time.time() - overall_time:.3f} "
        + "seconds")

    return acc


if __name__ == "__main__":
    main(get_parser().parse_args())

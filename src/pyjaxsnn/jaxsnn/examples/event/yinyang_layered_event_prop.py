# pylint: disable=logging-not-lazy,logging-fstring-interpolation
import argparse
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import random
import jaxsnn
from jaxsnn.base.compose import serial
from jaxsnn.base.dataset import yinyang_dataset
from jaxsnn.event.encode import (
    spatio_temporal_encode, target_temporal_encode, encode)
from jaxsnn.event import custom_lax
from jaxsnn.event.modules.leaky_integrate_and_fire import (
    EventPropLIF, LIFParameters)
from jaxsnn.event.loss import loss_wrapper, mse_loss
from jaxsnn.event.training import epoch, update
from jaxsnn.event.types import OptState


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
    # encoding
    parser.add_argument("--t-late", type=float, default=2.0 * 5e-3)
    parser.add_argument("--correct-target-time", type=float, default=0.9 * 5e-3)
    parser.add_argument("--wrong-target-time", type=float, default=1.1 * 5e-3)
    # model
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--tau-mem", type=float, default=1e-2)
    parser.add_argument("--tau-syn", type=float, default=5e-3)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--n-spikes-hidden", type=int, default=50)
    parser.add_argument("--n-spikes-output", type=int, default=53)
    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="<num samples>",
        help="input batch size for training")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    return parser


def main(args: argparse.Namespace):
    log = jaxsnn.get_logger("jaxsnn.examples.event.yinyang_layered_event_prop")

    # neuron params, low v_reset only allows one spike per neuron
    params = LIFParameters(
        v_reset=-1000.0, v_th=args.threshold, tau_syn=args.tau_syn,
        tau_mem=args.tau_mem)

    n_train_batches = args.trainset_size // args.batch_size
    n_test_batches = args.testset_size // args.batch_size
    train_samples = args.batch_size * n_train_batches
    test_samples = args.batch_size * n_test_batches

    t_max = 4.0 * params.tau_syn
    input_size = 5
    output_size = 3

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
        t_late=args.t_late, duplication=1, duplicate_neurons=False)))
    target_encoder_batched = jax.jit(jax.vmap(partial(
        target_temporal_encode,
        n_classes=3, correct_target_time=args.correct_target_time,
        wrong_target_time=args.wrong_target_time)))

    # Datasets
    trainset = encode(
        xy_trainset, input_encoder_batched, target_encoder_batched)
    testset = encode(
        xy_testset, input_encoder_batched, target_encoder_batched)

    # define net
    init_fn, apply_fn = serial(
        EventPropLIF(
            args.hidden_size,
            n_spikes=args.n_spikes_hidden,
            t_max=t_max,
            params=params,
            mean=3.0,
            std=1.6),
        EventPropLIF(
            output_size,
            n_spikes=args.n_spikes_output,
            t_max=t_max,
            params=params,
            mean=0.5,
            std=0.8))

    # init weights
    _, weights = init_fn(param_rng, input_size)

    # define and init optimizer
    scheduler = optax.exponential_decay(
        args.lr, n_train_batches, args.lr_decay)
    optimizer = optax.chain(
        optax.clip(0.01),
        optax.adam(scheduler))
    opt_state = optimizer.init(weights)

    # define loss and update function
    n_neurons = input_size + args.hidden_size + output_size
    loss_fn = partial(
        loss_wrapper, apply_fn, mse_loss, params.tau_mem,
        n_neurons, output_size)

    update_fn = jax.jit(partial(update, optimizer, loss_fn, params))
    epoch_fn = partial(epoch, update_fn, loss_fn, trainset, testset)

    # iterate over epochs
    res = custom_lax.scan(
        epoch_fn, OptState(opt_state, weights, rng), jnp.arange(args.epochs))
    state, (test_result, weights_over_time, duration) = res

    # save experiment data
    acc = round(test_result.accuracy[-1], 3)
    log.info(f"Acc: {acc} after {args.epochs} epochs")

    return acc


if __name__ == "__main__":
    main(get_parser().parse_args())

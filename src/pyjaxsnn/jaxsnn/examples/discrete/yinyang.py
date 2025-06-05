from typing import Callable, Dict, Tuple
import argparse
import time
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import optax
import jax.numpy as jnp
from jax import random
import jaxsnn
from jaxsnn.base.dataset import yinyang_dataset, data_loader
from jaxsnn.discrete.topology import Topology
from jaxsnn.discrete.modules import Input, LIF, LI, Linear
from jaxsnn.discrete.functional import LIFParameters, LIParameters
from jaxsnn.discrete.decode import max_over_time_decode
from jaxsnn.discrete.encode import spatio_temporal_encode
from jaxsnn.discrete.loss import nll_loss
from jaxsnn.discrete.encode import one_hot


log = jaxsnn.get_logger("jaxsnn.examples.discrete.yinyang")


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="jaxsnn spiking YinYang example",
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=0.98)
    parser.add_argument("--expected-spikes", type=float, default=0.8)
    return parser


def train_step(
    apply_fn: Callable,
    decoder: Callable,
    optimizer: optax.GradientTransformation,
    rho: float,
    target_rate: float,
    state: Tuple[optax.OptState, optax.Params],
    batch: Tuple[Dict[str, jax.Array], jax.Array],
) -> Tuple[Tuple[optax.OptState, optax.Params],
           Tuple[jax.Array, Dict[str, jax.Array]]]:
    """
    Run one training step.

    1. Forward pass and loss (+ regularization) computation
    2. Gradient computation
    3. Parameter update

    :param apply_fn: Batched model apply function called as
        apply_fn(inputs, params).
    :param decoder: Batched decoder applied to the LI layer output.
    :param optimizer: Optax optimizer used to compute and apply updates.
    :param rho: Regularization strength for firing-rate penalty.
    :param target_rate: Target spikes per sample for the LIF-layer regularizer.
    :param state: Tuple of (opt_state, parameters).
    :param batch: Mini-batch as (inputs, targets).

    :returns: Updated (opt_state, parameters) and (loss, preds).
    """
    opt_state, parameters = state

    def loss_fn(params, batch, apply_fn=apply_fn, decoder=decoder, rho=rho):
        inputs, targets = batch
        _, preds = apply_fn(inputs, params)
        preds_decoded = decoder(preds["li"])
        loss = nll_loss(preds_decoded, targets)
        regularization = rho * jnp.sum(
            jnp.square(jnp.sum(preds["lif"], axis=1) - target_rate)
        )
        return loss + regularization, preds

    (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        parameters, batch
    )

    updates, opt_state = optimizer.update(grads, opt_state)
    parameters = optax.apply_updates(parameters, updates)

    return (opt_state, parameters), (loss, preds)


def test_step(
    apply_fn: Callable,
    decoder: Callable,
    parameters: optax.Params,
    testset: Tuple[Dict[str, jax.Array], jax.Array],
) -> Tuple[jax.Array, jax.Array]:
    """
    Computes the accuracy and loss for a single test step.

    :param apply_fn: The model's apply function.
    :param decoder: The decoder function to compute loss and predictions from
        model output.
    :param parameters: The trained model parameters.
    :param testset: A batch of test data, as a tuple of (inputs, targets).

    :return: A tuple containing the computed accuracy and loss.
    """
    inputs, targets = testset

    _, preds = apply_fn(inputs, parameters)
    preds_decoded = decoder(preds["li"])

    correct = (jnp.argmax(preds_decoded, axis=1) == targets).sum()
    accuracy = correct / len(targets)

    targets = one_hot(targets, preds_decoded.shape[1])
    loss = -jnp.mean(jnp.sum(targets * preds_decoded, axis=1))

    return jnp.mean(accuracy), jnp.mean(loss)


def main(args: argparse.Namespace):
    """
    Function to train a LIF network to solve the YinYang dataset
    """
    lif_params = LIFParameters(
        tau_syn=args.tau_syn,
        tau_mem=args.tau_mem,
        v_th=args.v_th,
    )
    li_params = LIParameters(
        tau_syn=args.tau_syn,
        tau_mem=args.tau_mem,
    )

    n_train_batches = args.trainset_size // args.batch_size
    n_test_batches = args.testset_size // args.batch_size
    train_samples = args.batch_size * n_train_batches
    test_samples = args.batch_size * n_test_batches

    t_late = lif_params.tau_syn + lif_params.tau_mem
    time_steps = int(2 * t_late / args.dt)

    log.info(f"dt: {args.dt}, {time_steps} time steps, t_late: {t_late}")

    # Define RNGs
    rng = random.PRNGKey(args.seed)
    init_rng, train_rng, test_rng, shuffle_rng = random.split(rng, 4)

    # Setting up trainset and testset
    xy_trainset = yinyang_dataset(
        train_rng, train_samples, mirror=True, bias_spike=0.0,
    )
    xy_testset = yinyang_dataset(
        test_rng, test_samples, mirror=True, bias_spike=0.0,
    )

    # Encoding the inputs
    input_encoder_batched = jax.vmap(
        spatio_temporal_encode, in_axes=(0, None, None, None),
    )
    train_input_encoded = input_encoder_batched(
        xy_trainset[0], time_steps, t_late, args.dt,
    )
    test_input_encoded = input_encoder_batched(
        xy_testset[0], time_steps, t_late, args.dt,
    )

    # define topology
    builder = Topology(dt=args.dt)

    # create modules
    builder.add(
        {
            "inp": Input(5),
            "lif": LIF(
                args.hidden_size,
                lif_params,
            ),
            "li": LI(
                3,
                li_params,
            ),
            "syn_ih": Linear(
                mean=0.0,
                std=0.7,
            ),
            "syn_hh": Linear(
                mean=0.0,
                std=0.2,
            ),
            "syn_ho": Linear(
                mean=0.0,
                std=0.2,
            ),
        }
    )

    # connect modules
    builder.connect(
        [
            ("inp", "syn_ih"),
            ("syn_ih", "lif"),
            ("lif", "syn_hh"),
            ("syn_hh", "lif"),
            ("lif", "syn_ho"),
            ("syn_ho", "li"),
        ]
    )

    # Datasets
    trainset = ({"inp": train_input_encoded}, xy_trainset[1])
    testset = ({"inp": test_input_encoded}, xy_testset[1])

    # build topology
    init_fn, apply_fn = builder.done()

    # define optimizer
    scheduler = optax.exponential_decay(
        args.lr, n_train_batches, args.lr_decay,
    )
    optimizer = optax.adam(scheduler)

    apply_fn = jax.vmap(apply_fn, in_axes=(0, None))

    train_step_fn = partial(
        train_step,
        apply_fn,
        jax.vmap(max_over_time_decode),
        optimizer,
        1e-5,
        args.expected_spikes,
    )

    overall_time = time.time()
    parameters = init_fn(init_rng)
    opt_state = optimizer.init(parameters)

    accuracies, loss = [], []
    for epoch in range(args.epochs):
        start = time.time()
        # Generate randomly shuffled batches
        this_shuffle_rng, shuffle_rng = random.split(shuffle_rng)

        # Training
        trainset_batched = data_loader(trainset, 64, rng=this_shuffle_rng)
        (opt_state, parameters), (_, preds) = jax.lax.scan(
            train_step_fn,
            (opt_state, parameters),
            trainset_batched,
        )
        end = time.time() - start

        spikes_per_item = jnp.count_nonzero(preds["lif"]) / train_samples

        # Testing
        accuracy, test_loss = test_step(
            apply_fn,
            jax.vmap(max_over_time_decode),
            parameters,
            testset,
        )

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

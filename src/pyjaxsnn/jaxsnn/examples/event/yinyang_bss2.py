"""Train a two-layer feed-forward network with spike data from the BSS-2
system (Hardware-in-the-loop-training). A second forward run in software is
conducted to add missing information about the synaptic current at spike time.
In this run, the spikes from BSS-2 are used as solution for the root-solving.

Once the information about the synaptic current is also returned with the
event-based observations from BSS-2, this second forward pass in software
can be emitted.
"""
import argparse
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import random

import jaxsnn
from jaxsnn.base.dataset import yinyang_dataset
from jaxsnn.event import custom_lax
from jaxsnn.event.encode import (
    spatio_temporal_encode,
    target_temporal_encode,
    encode,
)
from jaxsnn.event.hardware.parameter import (
    MixedHXModelParameter,
    HXParameter,
)
from jaxsnn.event.loss import mse_loss
from jaxsnn.examples.event.utils import (
    loss_wrapper,
    test_step,
)
from jaxsnn.event.modules.hx import (
    HXSource,
    HXLIF,
    NeuronParameters,
    HXLinear,
)
from jaxsnn.event.topology import Topology
from jaxsnn.event.training import (
    epoch,
    update,
)
from jaxsnn.event.types import OptState


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="jaxsnn spiking YinYang example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--duplicate-neurons", action="store_true", default=True
    )

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
    parser.add_argument("--n-spikes-hidden", type=int, default=100)

    # training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="<num samples>",
        help="input batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-decay", type=float, default=0.98)

    # hw
    parser.add_argument("--duplication", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--hw-correction", type=float, default=-8.037022780627012e-08)
    parser.add_argument("--max-runtime", type=float, default=30e-6)
    parser.add_argument("--weight-scale", type=float, default=58)

    return parser


def main(args: argparse.Namespace):
    log = jaxsnn.get_logger("jaxsnn.examples.event.hardware.yinyang")

    # neuron params, low v_reset only allows one spike per neuron
    lif_params = NeuronParameters(
        v_th=MixedHXModelParameter(1.0, 150),
        v_leak=MixedHXModelParameter(0.0, 80),
        v_reset=MixedHXModelParameter(0.0, 80),
        refractory_time=HXParameter(30e-6),
        i_synin_gm=HXParameter(500),
        synapse_dac_bias=HXParameter(1000),
        tau_syn=HXParameter(args.tau_syn),
        tau_mem=HXParameter(args.tau_mem),
    )

    n_train_batches = args.trainset_size // args.batch_size
    n_test_batches = args.testset_size // args.batch_size
    train_samples = args.batch_size * n_train_batches
    test_samples = args.batch_size * n_test_batches

    t_max = 4.0 * lif_params.tau_syn.model_value
    t_max = max(t_max, args.max_runtime)

    # How many input neurons do we have?
    input_size = 5
    output_size = 3
    if args.duplicate_neurons:
        input_size *= args.duplication

    # define trainset and testset
    rng = random.PRNGKey(args.seed)
    param_rng, train_rng, test_rng, rng = random.split(rng, 4)

    xy_trainset = yinyang_dataset(
        train_rng,
        train_samples,
        mirror=True,
        bias_spike=0.0,
    )
    xy_testset = yinyang_dataset(
        test_rng,
        test_samples,
        mirror=True,
        bias_spike=0.0,
    )

    # Encoding
    input_encoder_batched = jax.jit(jax.vmap(partial(
        spatio_temporal_encode,
        t_late=args.t_late,
        duplication=args.duplication,
        duplicate_neurons=args.duplicate_neurons))
    )
    target_encoder_batched = jax.jit(jax.vmap(partial(
        target_temporal_encode,
        n_classes=output_size,
        correct_target_time=args.correct_target_time,
        wrong_target_time=args.wrong_target_time)),
    )

    # Datasets
    trainset = encode(
        xy_trainset,
        input_encoder_batched,
        target_encoder_batched,
    )
    testset = encode(
        xy_testset,
        input_encoder_batched,
        target_encoder_batched,
    )
    trainset = (
        {"inp": trainset[0]},
        trainset[1]
    )
    testset = (
        {"inp": testset[0]},
        testset[1],
    )

    jaxsnn.init_hardware()

    # define topology
    builder = Topology(
        mock=False,
        t_max=t_max,
        backprop_method="eventprop",
    )

    # create modules
    builder.add(
        {
            "inp": HXSource(
                size=input_size,
                n_events=5 * args.duplication,
            ),
            "lif_h": HXLIF(
                size=args.hidden_size,
                n_steps=args.n_spikes_hidden + 5 * args.duplication,
                n_hw_spikes=args.n_spikes_hidden,
                params=lif_params,
                time_offset=args.hw_correction,
            ),
            "lif_o": HXLIF(
                size=output_size,
                n_steps=output_size + args.n_spikes_hidden,
                n_hw_spikes=output_size,
                params=lif_params,
                time_offset=args.hw_correction,
            ),
            "syn_ih": HXLinear(
                mean=0.6,
                std=0.3,
                min_delay=0.0,
                weight_scale=args.weight_scale,
            ),
            "syn_ho": HXLinear(
                mean=0.5,
                std=0.8,
                min_delay=0.0,
                weight_scale=args.weight_scale,
            )
        }
    )

    # connect modules
    builder.connect(
        [
            ("inp", "syn_ih"),
            ("syn_ih", "lif_h"),
            ("lif_h", "syn_ho"),
            ("syn_ho", "lif_o"),
        ]
    )

    # build topology
    init_fn, apply_fn = builder.done()

    # init weights
    params = init_fn(param_rng)

    # Optimizer
    scheduler = optax.exponential_decay(
        args.lr,
        n_train_batches,
        args.lr_decay
    )
    optimizer = optax.chain(
        optax.scale(1.0 / args.tau_syn),
        optax.clip(0.01),
        optax.adam(scheduler),
    )
    opt_state = optimizer.init(params)

    # define loss and update function
    loss_fn = partial(
        loss_wrapper,
        apply_fn,
        mse_loss,
        args.tau_mem,
        "lif_o",
        output_size,
    )

    update_fn = jax.jit(partial(update, optimizer, loss_fn))
    test_fn = partial(test_step, loss_fn)
    epoch_fn = partial(
        epoch,
        update_fn,
        test_fn,
        trainset,
        testset,
        args.batch_size,
        args.batch_size,
    )

    # train the net
    res = custom_lax.scan(
        epoch_fn,
        OptState(opt_state, params, rng),
        jnp.arange(args.epochs, dtype=int),
    )
    opt_state, (test_result, _, _) = res

    # find best epoch
    acc = round(test_result[1][-1], 3)
    log.info(f"Acc: {acc} after {args.epochs} epochs")

    jaxsnn.release_hardware()
    return acc


if __name__ == "__main__":
    main(get_parser().parse_args())

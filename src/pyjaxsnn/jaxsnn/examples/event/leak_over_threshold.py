import argparse

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxsnn.event.topology import Topology
from jaxsnn.event.modules import (
    Source,
    Linear,
    LIF,
    LIFParameters,
)
from jaxsnn.event.types import Spike


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="jaxsnn leak-over-threshold example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-max", type=float, default=200e-3)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--tau-mem", type=float, default=1e-2)
    parser.add_argument("--tau-syn", type=float, default=5e-3)
    parser.add_argument("--w-std", type=float, default=1e-2)
    parser.add_argument("--v-reset", type=float, default=0.0)
    parser.add_argument("--v-leak", type=float, default=3.0)
    parser.add_argument("--v-th", type=float, default=1.0)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument(
        "--plot", default=False, action="store_true",
    )
    return parser


def plot(
    times: jnp.array,
    args: argparse.Namespace,
):
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(times, [0] * len(times), alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title('Spike Times for LIF1 Layer')
    plt.grid(True, alpha=0.3)
    plt.savefig(args.plot_path)


def main(
    args: argparse.Namespace,
) -> jnp.array:
    params = LIFParameters(
        v_reset=args.v_reset,
        v_leak=args.v_leak,
        v_th=args.v_th,
        tau_syn=args.tau_syn,
        tau_mem=args.tau_mem,
    )

    input_size = 100

    # Define trainset and testset
    rng = jax.random.PRNGKey(args.seed)
    param_rng, input_rng = jax.random.split(rng, 2)

    times = jnp.sort(jax.random.uniform(input_rng, (1, 100), maxval=args.t_max))
    index = jax.random.randint(input_rng, (1, 100,), 0, input_size)

    input_spikes = {
        "inp": Spike(
            time=times,
            idx=index,
            current=jnp.zeros_like(index, dtype=float),
            layer_idx=jnp.zeros_like(index, dtype=int),
            internal=jnp.ones_like(index, dtype=bool),
        )
    }

    # Datasets
    builder = Topology(t_max=args.t_max)
    # create modules
    builder.add(
        {
            "inp": Source(input_size),
            "syn": Linear(
                mean=0.0,
                std=args.w_std,
                min_delay=0.000,
            ),
            "lif": LIF(
                1,
                1000,
                params,
            ),
        }
    )
    # connect modules
    builder.connect(
        [
            ("inp", "syn"),
            ("syn", "lif"),
        ]
    )

    # build topology
    init_fn, apply_fn = builder.done()

    # init weights
    weights = init_fn(param_rng)

    # Extract spike times for plotting
    spikes = apply_fn(input_spikes, weights)
    lif_spikes = spikes["lif"]
    spike_times = lif_spikes.time[lif_spikes.internal]

    if args.plot:
        plot(spike_times, args)

    return spike_times


if __name__ == "__main__":
    main(get_parser().parse_args())

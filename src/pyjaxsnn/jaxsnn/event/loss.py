import jax
import jax.numpy as jnp
from jaxsnn.event.types import Spike
from jaxsnn.event.states import LIFState


def max_over_time(output: LIFState) -> jax.Array:
    return jnp.max(output.V, axis=0)


def nll_loss(output: jax.Array, targets: jax.Array) -> jax.Array:
    n_classes = targets.shape[0]
    idx = jnp.argmin(targets)
    targets = jnp.array(idx == jnp.arange(n_classes))
    output = jnp.maximum(output, 0)
    preds = jax.nn.log_softmax(output)
    loss = -jnp.sum(targets * preds)
    return loss


def target_time_loss(
    first_spikes: jax.Array,
    target: jax.Array,
    tau_mem: float
) -> jax.Array:
    loss_value = -jnp.sum(
        jnp.log(1 + jnp.exp(-jnp.abs(first_spikes - target) / tau_mem))
    )
    return loss_value


def ttfs_loss(
    first_spikes: jax.Array,
    target: jax.Array,
    tau_mem: float,
) -> jax.Array:
    idx = jnp.argmin(target)
    # TODO: Exchange 2 * tau_mem with t_max
    first_spikes = jnp.minimum(jnp.abs(first_spikes), 2 * tau_mem)
    return -jnp.log(
        jnp.sum(jnp.exp((first_spikes[idx] - first_spikes) / tau_mem))
    )


def mse_loss(
    first_spikes: jax.Array,
    target: jax.Array,
    tau_mem: float,
) -> jax.Array:
    return jnp.sum(
        jnp.square((jnp.minimum(first_spikes, 2 * tau_mem) - target) / tau_mem)
    )


def first_spike(
    spikes: Spike,
    n_outputs: int
) -> jax.Array:
    spikes = spikes.where(spikes.internal)
    first_spikes = jnp.array(
        [
            jnp.min(jnp.where(spikes.idx == idx, spikes.time, jnp.inf))
            for idx in range(n_outputs)
        ]
    )
    return first_spikes

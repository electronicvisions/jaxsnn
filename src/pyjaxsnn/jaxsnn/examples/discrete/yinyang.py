import time
from functools import partial

import jax
import optax
from jax import numpy as np
from jax import random
import jaxsnn
from jaxsnn.discrete.compose import serial
from jaxsnn.discrete.leaky_integrate import LI
from jaxsnn.discrete.leaky_integrate_and_fire import LIF
from jaxsnn.discrete.decode import max_over_time_decode
from jaxsnn.discrete.encode import spatio_temporal_encode
from jaxsnn.discrete.loss import nll_loss, acc_and_loss
from jaxsnn.base.params import LIFParameters
from jaxsnn.discrete.dataset.yinyang import YinYangDataset, data_loader


log = jaxsnn.get_logger("jaxsnn.examples.discrete.yinyang")


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
        weights, (inputs, output), max_over_time_decode
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)
    return (opt_state, weights, i + 1), recording


def train(seed: int = 0, epochs: int = 100, DT: float = 5e-4):
    n_classes = 3
    input_shape = 5
    dataset_size = 5000
    batch_size = 64
    n_train_batches = dataset_size / batch_size
    bias_spike = 0.0

    hidden_features = 120
    expected_spikes = 0.8
    lr_decay = 0.98
    step_size = 5e-4

    t_late = (
        1.0 / LIFParameters().tau_syn_inv + 1.0 / LIFParameters().tau_mem_inv
    )
    time_steps = int(2 * t_late / DT)
    log.info(f"DT: {DT}, {time_steps} time steps, t_late: {t_late}")

    # define train and test data
    rng = random.PRNGKey(42)
    rng, train_key, test_key, init_key = random.split(rng, 4)
    trainset = YinYangDataset(train_key, 4992, bias_spike=bias_spike)
    test_dataset = YinYangDataset(test_key, 1000, bias_spike=bias_spike)

    trainset_batches = data_loader(trainset, batch_size, None)
    # Encoding the inputs
    input_encoder_batched = jax.vmap(
        spatio_temporal_encode,
        in_axes=(0, None, None, None)
    )
    train_input_encoded = input_encoder_batched(
        trainset_batches[0],
        time_steps,
        t_late,
        DT,
    )
    trainset = (train_input_encoded, trainset_batches[1])

    test_input_encoded = spatio_temporal_encode(
        test_dataset.vals,
        time_steps,
        t_late,
        DT,
    )
    test_dataset.vals = test_input_encoded

    # define the network
    snn_init, snn_apply = serial(
        LIF(hidden_features),
        LI(n_classes),
    )

    # define optimizer
    scheduler = optax.exponential_decay(step_size, n_train_batches, lr_decay)
    optimizer_fn = optax.adam
    optimizer = optimizer_fn(scheduler)

    # define loss and train function
    snn_apply = partial(snn_apply, recording=True)
    loss_fn = partial(
        nll_loss, snn_apply, expected_spikes=expected_spikes, rho=1e-5
    )
    train_step_fn = partial(train_step, optimizer, loss_fn=loss_fn)

    overall_time = time.time()
    init_key = random.PRNGKey(seed)
    _, weights = snn_init(init_key, input_shape=input_shape)
    opt_state = optimizer.init(weights)

    accuracies = []
    loss = []
    for epoch in range(epochs):
        start = time.time()
        (opt_state, weights, i), recording = jax.lax.scan(
            train_step_fn, (opt_state, weights, 0), trainset
        )
        end = time.time() - start

        spikes_per_item = np.count_nonzero(recording[0].z) / dataset_size
        accuracy, test_loss = acc_and_loss(
            snn_apply, weights,
            (test_dataset.vals, test_dataset.classes),
            max_over_time_decode
        )
        accuracies.append(accuracy)
        loss.append(test_loss)
        log.info(
            f"Epoch: {epoch}, Loss: {test_loss:3f}, Test accuracy: {accuracy:.3f}, Seconds: {end:.3f}, Spikes: {spikes_per_item:.1f}"
        )
    log.info(
        f"Finished {epochs} epochs in {time.time() - overall_time:.3f} seconds"
    )
    return accuracies, loss


if __name__ == "__main__":
    train()

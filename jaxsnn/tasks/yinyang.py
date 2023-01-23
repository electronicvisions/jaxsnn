import time
from functools import partial

import matplotlib.pyplot as plt
import optax
from jax import numpy as np
from jax import random, value_and_grad
from jax.lax import scan

import jaxsnn
from jaxsnn.dataset.yinyang import DataLoader, YinYangDataset
from jaxsnn.functional.lif import LIFParameters
from jaxsnn.functional.loss import acc_and_loss, nll_loss
from jaxsnn.functional.threshold import superspike


def train_step(state, batch, loss_fn):
    opt_state, params, i = state
    input, output = batch

    (loss, recording), grads = value_and_grad(loss_fn, has_aux=True)(
        params, (input, output)
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (opt_state, params, i + 1), recording


if __name__ == "__main__":
    n_classes = 3
    input_shape = 4
    dataset_size = 4992
    batch_size = 64
    n_train_batches = dataset_size / batch_size

    epochs = 200
    hidden_features = 120
    expected_spikes = 0.5
    step_size = 1e-3
    DT = 5e-4
    lr_decay = 0.99

    t_late = 1.0 / LIFParameters().tau_syn_inv + 1.0 / LIFParameters().tau_mem_inv
    T = int(2 * t_late / DT) * 5
    T = 400
    print(T)
    print(t_late)

    rng = random.PRNGKey(42)
    rng, train_key, test_key, init_key = random.split(rng, 4)
    trainset = YinYangDataset(train_key, 4992)

    test_dataset = YinYangDataset(test_key, 1000)

    snn_init, snn_apply = jaxsnn.serial(
        jaxsnn.functional.SpatioTemporalEncode(T, t_late, DT),
        jaxsnn.euler_integrate(
            jaxsnn.LIFStep(hidden_features, superspike), jaxsnn.LIStep(n_classes)
        ),
        jaxsnn.functional.MaxOverTimeDecode(),
    )

    output_shape, params = snn_init(init_key, input_shape=input_shape)

    scheduler = optax.exponential_decay(step_size, n_train_batches, lr_decay)
    optimizer_fn = optax.adam
    optimizer = optimizer_fn(scheduler)
    opt_state = optimizer.init(params)

    # define functions
    snn_apply = partial(snn_apply, recording=True)
    loss_fn = partial(nll_loss, snn_apply, expected_spikes=expected_spikes)
    train_step_fn = partial(train_step, loss_fn=loss_fn)

    def plot_neuron_voltage(recording):
        # recording[layer_idx].observable[step idx, time_idx, batch idx, neuron_idx]
        for neuron_idx in range(5):
            x = recording[1].v[0, :, 0, neuron_idx]
            plt.plot(np.arange(T), x)
        plt.savefig("./plots/voltage.png")

    def plot_spikes_per_step(recording):
        # recording[layer_idx].observable[step idx, time_idx, batch idx, neuron_idx]
        x = np.sum(recording[1].z, axis=(1, 2, 3))
        plt.plot(np.arange(len(x)), x)
        plt.savefig("./plots/spikes.png")

    overall_time = time.time()
    for epoch in range(epochs):
        trainloader = DataLoader(trainset, batch_size, rng=None)
        start = time.time()
        (opt_state, params, i), recording = scan(
            train_step_fn, (opt_state, params, 0), trainloader
        )
        end = time.time() - start

        spikes_per_item = np.count_nonzero(recording[1].z) / len(trainset)
        accuracy, test_loss = acc_and_loss(
            snn_apply, params, (test_dataset.vals, test_dataset.classes)
        )
        print(
            f"Epoch: {epoch}, Loss: {test_loss:3f}, Test accuracy: {accuracy:.3f}, Seconds: {end:.3f}, Spikes: {spikes_per_item:.1f}"
        )
    print(f"Finished {epochs} epochs in {time.time() - overall_time:.3f} seconds")

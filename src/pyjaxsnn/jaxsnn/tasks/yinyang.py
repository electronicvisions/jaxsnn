from pathlib import Path
import time
from functools import partial
import datetime as dt
import matplotlib.pyplot as plt
import optax
from jax import numpy as np
from jax import random
import jax

import jaxsnn
from jaxsnn.dataset.yinyang import DataLoader, YinYangDataset
from jaxsnn.functional.lif import LIFParameters
from jaxsnn.functional.loss import acc_and_loss, nll_loss
from jaxsnn.functional.threshold import superspike


def train_step(optimizer, state, batch, loss_fn):
    opt_state, params, i = state
    input, output = batch

    (loss, recording), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, (input, output)
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (opt_state, params, i + 1), recording


def train(folder, seed, epochs: int = 100, DT=2e-4):
    n_classes = 3
    input_shape = 5
    dataset_size = 5000
    batch_size = 64
    n_train_batches = dataset_size / batch_size

    hidden_features = 120
    expected_spikes = 0.8
    lr_decay = 0.98
    bias_spike = 0.0

    # ssems like smaller time bins require a higher learning rate
    DT = 2e-4
    step_size = 5e-4

    # 2e-4 and 2e-4 works well
    # DT = 2e-4, step_size = 5e-4 => 96%
    # DT = 2e-4, step_size = 3e-4 => 96%

    t_late = 1.0 / LIFParameters().tau_syn_inv + 1.0 / LIFParameters().tau_mem_inv
    T = int(2 * t_late / DT)
    print(f"DT: {DT}, {T} time steps, t_late: {t_late}")

    rng = random.PRNGKey(42)
    rng, train_key, test_key, init_key = random.split(rng, 4)
    trainset = YinYangDataset(train_key, 4992, bias_spike=bias_spike)

    test_dataset = YinYangDataset(test_key, 1000, bias_spike=bias_spike)

    snn_init, snn_apply = jaxsnn.serial(
        jaxsnn.functional.SpatioTemporalEncode(T, t_late, DT),
        jaxsnn.euler_integrate(
            jaxsnn.LIFStep(hidden_features, superspike), jaxsnn.LIStep(n_classes)
        ),
        jaxsnn.functional.MaxOverTimeDecode(),
    )

    scheduler = optax.exponential_decay(step_size, n_train_batches, lr_decay)
    optimizer_fn = optax.adam
    optimizer = optimizer_fn(scheduler)

    # define functions
    snn_apply = partial(snn_apply, recording=True)
    loss_fn = partial(nll_loss, snn_apply, expected_spikes=expected_spikes, rho=1e-5)
    train_step_fn = partial(train_step, optimizer, loss_fn=loss_fn)

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
    init_key = random.PRNGKey(seed)
    _, params = snn_init(init_key, input_shape=input_shape)
    opt_state = optimizer.init(params)

    accuracies = []
    loss = []
    for epoch in range(epochs):
        trainloader = DataLoader(trainset, batch_size, rng=None)
        start = time.time()
        (opt_state, params, i), recording = jax.lax.scan(
            train_step_fn, (opt_state, params, 0), trainloader
        )
        end = time.time() - start

        spikes_per_item = np.count_nonzero(recording[1].z) / len(trainset)
        accuracy, test_loss = acc_and_loss(
            snn_apply, params, (test_dataset.vals, test_dataset.classes)
        )
        accuracies.append(accuracy)
        loss.append(test_loss)
        print(
            f"Epoch: {epoch}, Loss: {test_loss:3f}, Test accuracy: {accuracy:.3f}, Seconds: {end:.3f}, Spikes: {spikes_per_item:.1f}"
        )
    print(f"Finished {epochs} epochs in {time.time() - overall_time:.3f} seconds")
    return accuracies, loss


if __name__ == "__main__":
    dt_string = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = f"jaxsnn/plots/norse/yinyang/{dt_string}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"Running experiment, results in folder: {folder}")

    # different seeds
    epochs = 100
    seeds = 5
    acc_container = []
    loss_container = []
    for i in range(seeds):
        acc, loss = train(folder, seed=i, DT=5e-4, epochs=epochs)
        acc_container.append(acc)
        loss_container.append(loss)

    np.save(
        f"{folder}/acc_{seeds}seeds_{epochs}epochs.npy",
        np.array(acc_container),
        allow_pickle=True,
    )
    np.save(
        f"{folder}/loss_{seeds}seeds_{epochs}epochs.npy",
        np.array(loss_container),
        allow_pickle=True,
    )

    # different time bins and different seeds
    # epochs = 300
    # seeds = 5
    # acc_container = []
    # loss_container = []
    # for DT in np.logspace(-4, -2, 5):
    #     acc, loss = train(folder, seed=0, epochs=epochs, DT=DT)
    #     acc_container.append(acc)
    #     loss_container.append(loss)

    # np.save(f"{folder}/acc_dt_logspace_{epochs}epochs.npy", np.array(acc_container), allow_pickle=True)
    # np.save(f"{folder}/loss_dt_logspace_{epochs}epochs.npy", np.array(loss_container), allow_pickle=True)

from functools import partial

import jax
import optax
from jax import random
from jaxsnn import discrete
from jaxsnn.base.params import LIFParameters
from jaxsnn.discrete.dataset.yinyang import DataLoader, YinYangDataset
from jaxsnn.discrete.loss import acc_and_loss, nll_loss
from jaxsnn.discrete.threshold import superspike


def update(optimizer, state, batch, loss_fn):
    opt_state, weights, i = state
    input, output = batch

    (loss, recording), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        weights, (input, output)
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)
    return (opt_state, weights, i + 1), recording


def test_train():
    n_classes = 3
    input_shape = 4
    batch_size = 64
    epochs = 3

    hidden_features = 70
    expected_spikes = 0.5
    step_size = 1e-3
    DT = 5e-4

    t_late = (
        1.0 / LIFParameters().tau_syn_inv + 1.0 / LIFParameters().tau_mem_inv
    )
    T = int(2 * t_late / DT)

    rng = random.PRNGKey(42)
    rng, train_key, test_key, init_key = random.split(rng, 4)
    trainset = YinYangDataset(train_key, 6400)
    test_dataset = YinYangDataset(test_key, 1000)

    snn_init, snn_apply = discrete.serial(
        discrete.SpatioTemporalEncode(T, t_late, DT),
        discrete.euler_integrate(
            discrete.LIFStep(hidden_features, superspike),
            discrete.LIStep(n_classes),
        ),
        discrete.MaxOverTimeDecode(),
    )

    _, weights = snn_init(init_key, input_shape=input_shape)

    optimizer = optax.adam(step_size)
    opt_state = optimizer.init(weights)

    # define functions
    snn_apply = partial(snn_apply, recording=True)
    loss_fn = partial(nll_loss, snn_apply, expected_spikes=expected_spikes)
    train_step_fn = partial(partial(update, optimizer), loss_fn=loss_fn)

    trainloader = DataLoader(trainset, batch_size, rng=None)
    for _ in range(epochs):
        (opt_state, weights, _), _ = jax.lax.scan(
            train_step_fn, (opt_state, weights, 0), trainloader
        )

    accuracy, _ = acc_and_loss(
        snn_apply, weights, (test_dataset.vals, test_dataset.classes)
    )
    assert accuracy > 0.70

from functools import partial

import optax
from jax import random, value_and_grad
from jax.lax import scan

import jaxsnn
from jaxsnn.dataset.yinyang import DataLoader, YinYangDataset
from jaxsnn.functional.lif import LIFParameters
from jaxsnn.functional.loss import nll_loss, acc_and_loss
from jaxsnn.functional.threshold import superspike


def update(optimizer, state, batch, loss_fn):
    opt_state, params, i = state
    input, output = batch

    (loss, recording), grads = value_and_grad(loss_fn, has_aux=True)(
        params, (input, output)
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (opt_state, params, i + 1), recording


def test_train():
    n_classes = 3
    input_shape = 4
    batch_size = 64
    epochs = 3

    hidden_features = 70
    expected_spikes = 0.5
    step_size = 1e-3
    DT = 5e-4

    t_late = 1.0 / LIFParameters().tau_syn_inv + 1.0 / LIFParameters().tau_mem_inv
    T = int(2 * t_late / DT)

    rng = random.PRNGKey(42)
    rng, train_key, test_key, init_key = random.split(rng, 4)
    trainset = YinYangDataset(train_key, 6400)
    test_dataset = YinYangDataset(test_key, 1000)

    snn_init, snn_apply = jaxsnn.serial(
        jaxsnn.functional.SpatioTemporalEncode(T, t_late, DT),
        jaxsnn.euler_integrate(
            jaxsnn.LIFStep(hidden_features, superspike), jaxsnn.LIStep(n_classes)
        ),
        jaxsnn.functional.MaxOverTimeDecode(),
    )

    _, params = snn_init(init_key, input_shape=input_shape)

    optimizer = optax.adam(step_size)
    opt_state = optimizer.init(params)

    # define functions
    snn_apply = partial(snn_apply, recording=True)
    loss_fn = partial(nll_loss, snn_apply, expected_spikes=expected_spikes)
    train_step_fn = partial(partial(update, optimizer), loss_fn=loss_fn)

    trainloader = DataLoader(trainset, batch_size, rng=None)
    for _ in range(epochs):
        (opt_state, params, _), _ = scan(
            train_step_fn, (opt_state, params, 0), trainloader
        )

    accuracy, _ = acc_and_loss(
        snn_apply, params, (test_dataset.vals, test_dataset.classes)
    )
    assert accuracy > 0.70

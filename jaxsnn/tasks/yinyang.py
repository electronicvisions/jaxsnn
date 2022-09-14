import time
from functools import partial

import jax.numpy as jnp
import jaxsnn
from jax import random, value_and_grad
from jax.example_libraries.optimizers import adam
from jax.lax import scan
from jaxsnn.dataset.yinyang import DataLoader, YinYangDataset
from jaxsnn.functional.lif import LIFParameters
from jaxsnn.model.snn import acc_and_loss, nll_loss


# TODO pass and plot records
# TODO get_params adn opt_update functions are inside
def step(loss_fun, state, batch):
    opt_state, i = state
    net_params = get_params(opt_state)
    input, output = batch

    (loss, _recording), grads = value_and_grad(loss_fun, has_aux=True)(
        net_params, (input, output)
    )
    opt_state = opt_update(i, grads, opt_state)
    return (opt_state, i + 1), loss


if __name__ == "__main__":
    batch_size = 64
    epochs = 50
    n_classes = 3
    hidden_features = 50
    input_features = 4

    DT = 5e-4
    t_late = 1.0 / LIFParameters().tau_syn_inv + 1.0 / LIFParameters().tau_mem_inv  # type: ignore
    T = int(5 + t_late / DT)

    train_key = random.PRNGKey(42)
    trainset = YinYangDataset(train_key, 6400)
    trainloader = DataLoader(trainset, batch_size)

    test_key = random.PRNGKey(10)
    test_dataset = YinYangDataset(test_key, 1000)

    # TODO serial should jit the function?
    snn_init, snn_apply = jaxsnn.serial(
        jaxsnn.SpatioTemporalEncode(T, t_late, DT),
        jaxsnn.LIF(hidden_features),
        jaxsnn.LI(n_classes),
        jaxsnn.MaxOverTimeDecode(),
    )
    loss_fun = partial(nll_loss, snn_apply)
    step_fun = partial(step, loss_fun)

    net_key = random.PRNGKey(3)
    output_shape, params_initial = snn_init(net_key, input_shape=4)

    opt_init, opt_update, get_params = adam(1e-3)
    opt_state = opt_init(params_initial)

    for epoch in range(epochs):
        start = time.time()
        (opt_state, i), loss = scan(step_fun, (opt_state, 0), trainloader)
        accuracy, _ = acc_and_loss(
            snn_apply, get_params(opt_state), (test_dataset.vals, test_dataset.classes)
        )
        print(
            f"Epoch: {epoch}, Loss: {jnp.mean(loss):3f}, Test accuracy: {accuracy:.2f}, Seconds: {time.time() - start:.3f}"
        )

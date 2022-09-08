import copy
import time
from dataclasses import dataclass

import jax.numpy as jnp
from jax import random, value_and_grad
from jax.example_libraries.optimizers import adam
from jaxsnn.dataset.yingyang import YinYangDataset
from jaxsnn.functional.encode import one_hot, spatio_temporal_encode
from jaxsnn.functional.leaky_integrator import li_init_state, li_init_weights
from jaxsnn.functional.lif import LIFParameters, lif_init_state, lif_init_weights
from jaxsnn.model.snn import accuracy_and_loss, nll_loss


@dataclass
class Loader:
    # [time, batch, x, y]
    dataset: YinYangDataset
    batch_size: int
    idx: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < len(self):
            start = self.idx * self.batch_size
            stop = (self.idx + 1) * self.batch_size
            result = self.dataset.vals[start:stop], self.dataset.classes[start:stop]  # type: ignore
            self.idx += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.dataset.classes) // self.batch_size


if __name__ == "__main__":
    batch_size = 64
    epochs = 100
    step_size = 1e-3
    n_classes = 3
    hidden_features = 50
    input_features = 4  # make this more generic
    momentum = 0.9

    # this spatial resolution may not permit better results
    T = 50
    DT = 5e-4
    t_late = 1.0 / LIFParameters().tau_syn_inv + 1.0 / LIFParameters().tau_mem_inv  # type: ignore

    # TODO use scan over update for train function
    # train = scan(update, trainloader)
    # train()
    def train(opt_state, state, trainloader):
        losses = []
        # grads = tree_map(lambda p: jnp.zeros_like(p), params)
        for i, (input, output) in enumerate(trainloader):
            input = spatio_temporal_encode(input, T, t_late, DT)
            output = one_hot(output, n_classes)
            state = copy.deepcopy(state)
            net_params = get_params(opt_state)
            (loss, recording), grads = value_and_grad(nll_loss, has_aux=True)(
                net_params, state, (input, output)
            )
            opt_state = opt_update(i, grads, opt_state)
            losses.append(loss)
        return opt_state, jnp.mean(jnp.array(losses))

    # TODO vmap over accuracy and loss
    # test = vmap(accuracy_and_loss, testloader)
    # test()
    def test(params, testset):
        states = [
            lif_init_state((len(testset), params[0][0].shape[1])),
            li_init_state((len(testset), params[1][0].shape[1])),
        ]
        input = spatio_temporal_encode(testset.vals, T, t_late, DT)
        output = one_hot(testset.classes, n_classes)
        accuracy, loss = accuracy_and_loss(params, states, (input, output))
        return jnp.mean(accuracy), jnp.mean(loss)

    key = random.PRNGKey(42)
    dataset = YinYangDataset(key, 6400)

    test_key = random.PRNGKey(10)
    test_dataset = YinYangDataset(test_key, 1000)

    # define network with states and params
    # maybe just define forward function which we can pass to train?
    # TODO look how this is done in stax
    params = [
        lif_init_weights(key, input_features, hidden_features, scale=0.5),
        li_init_weights(key, hidden_features, n_classes, scale=0.2),
    ]
    states = [
        lif_init_state((batch_size, params[0][0].shape[1])),
        li_init_state((batch_size, params[1][0].shape[1])),
    ]

    # TODO we need adam optimizer or adaptive lr for better results
    opt_init, opt_update, get_params = adam(step_size)
    opt_state = opt_init(params)

    for epoch in range(epochs):
        start = time.time()
        opt_state, mean_loss = train(opt_state, states, Loader(dataset, batch_size))
        train_time = time.time() - start
        start = time.time()
        accuracy, loss = test(get_params(opt_state), test_dataset)
        test_time = time.time() - start
        print(
            f"Epoch: {epoch}, Loss: {mean_loss}, Test accuracy: {accuracy:.2f}, Seconds: {train_time:.3f}, {test_time:.3f}"
        )


# test_loss, accuracy = test(model, DEVICE, test_loader, epoch, rho_a=0.5)

from dataclasses import dataclass

import jax.numpy as jnp
from jax import random
from jaxsnn.dataset.yingyang import YinYangDataset
from jaxsnn.functional.encode import one_hot, spatio_temporal_encode
from jaxsnn.functional.leaky_integrator import li_init_weights
from jaxsnn.functional.lif import LIFParameters, lif_init_weights
from jaxsnn.model.snn import update


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
            result = self.dataset.vals[start:
                                       stop], self.dataset.classes[start: stop]
            self.idx += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.dataset.classes) // self.batch_size


if __name__ == "__main__":
    batch_size = 32
    epochs = 5
    step_size = 0.0005
    T = 80
    DT = 5e-4
    t_late = 1.0 / LIFParameters().tau_syn_inv + 1.0 / LIFParameters().tau_mem_inv

    def train(params, trainloader, step_size):
        losses = []
        for (input, output) in trainloader:
            input = spatio_temporal_encode(input, T, t_late, DT)
            output = one_hot(output, 3)
            params, loss = update(params, input, output, step_size=step_size)
            losses.append(loss)
        return losses, jnp.mean(jnp.array(losses))

    key = random.PRNGKey(42)
    dataset = YinYangDataset(key, 1000)

    params = [lif_init_weights(key, 4, 50, scale=0.5),
              li_init_weights(key, 50, 3, scale=0.2)]

    for epoch in range(epochs):
        training_losses, mean_loss = train(
            params, Loader(dataset, batch_size), step_size)
        print(f"Epoch: {epoch}, Loss: {mean_loss}")

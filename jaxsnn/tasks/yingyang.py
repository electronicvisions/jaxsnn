from jax import random
from jaxsnn.dataset.yingyang import YinYangDataset
from jaxsnn.functional.encode import spatio_temporal_encode
from jaxsnn.functional.leaky_integrator import li_init_state, li_init_weights
from jaxsnn.functional.lif import lif_init_state, lif_init_weights
from jaxsnn.model.snn import forward

if __name__ == "__main__":
    key = random.PRNGKey(42)
    dataset = YinYangDataset(key, 1000)
    states = [lif_init_state(50), li_init_state(3)]
    weights = [lif_init_weights(key, 4, 50), li_init_weights(key, 50, 3)]

    for input, output in dataset:
        input = spatio_temporal_encode(input, 10)
        forward(states, weights, input)

import unittest
import jax
import jax.numpy as jnp
from jaxsnn.base.params import LIFParameters
from jaxsnn.base.compose import serial
from jaxsnn.event.modules.leaky_integrate_and_fire import LIF
from jaxsnn.event.types import EventPropSpike
from jaxsnn.event.utils import load_weights


class TestYinYangLoss(unittest.TestCase):

    def test_loss(self):
        params = LIFParameters(v_reset=-1000.0)
        t_max = 4.0 * params.tau_syn

        # net
        hidden_size = 60
        output_size = 3
        n_spikes_hidden = 60
        n_spikes_output = 55

        # class 1
        input_spikes = EventPropSpike(
            time=jnp.array(
                [0.0000000e00, 0.00195968, 0.00479668, 0.00520332, 0.00804032]
            ),
            idx=jnp.array([4, 0, 3, 1, 2]),
            current=jnp.zeros(5),
        )

        weights = load_weights(
            ["tests/sw/event/tasks/weights3.npy",
            "tests/sw/event/tasks/weights4.npy"])

        # declare net
        _, apply_fn = serial(
            LIF(hidden_size, n_spikes_hidden, t_max, params=params),
            LIF(output_size, n_spikes_output, t_max, params=params))

        def first_spike(spikes: EventPropSpike, size: int):
            return jnp.array(
                [
                    jnp.min(jnp.where(spikes.idx == idx, spikes.time, jnp.inf))
                    for idx in range(size)
                ]
            )

        def loss_fn(
            weights,
            input_spikes,
        ):
            _, _, _, recording = apply_fn(weights, input_spikes, None, None)
            first_spikes = first_spike(recording[1], 3)
            loss = -jnp.log(
                jnp.sum(
                    1
                    + jnp.exp(-first_spikes[1] / params.tau_mem)
                    - jnp.exp(-first_spikes / params.tau_mem)
                )
            )
            return loss, recording

        (loss, recording), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            weights, input_spikes)
        self.assertFalse(jnp.isnan(jnp.mean(grad[0].input)))
        self.assertAlmostEqual(loss, -1.0986, 4)


if __name__ == '__main__':
    unittest.main()

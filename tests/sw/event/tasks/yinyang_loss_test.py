import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.compose import serial
from jaxsnn.event.leaky_integrate_and_fire import LIF
from jaxsnn.event.types import EventPropSpike
from jaxsnn.event.utils import load_params
from numpy.testing import assert_almost_equal


def test_nans():
    p = LIFParameters(v_reset=-1000.0)
    t_max = 4.0 * p.tau_syn

    # net
    hidden_size = 60
    output_size = 3
    n_spikes_hidden = 60
    n_spikes_output = 55

    # class 1
    input_spikes = EventPropSpike(
        time=np.array([0.0000000e00, 4.4690369e-05, 4.4967532e-03, 5.5032466e-03]),
        idx=np.array([4, 2, 3, 1, 0]),
        current=np.zeros(5),
    )

    params = load_params(
        [
            "tests/sw/event/tasks/weights3.npy",
            "tests/sw/event/tasks/weights4.npy",
        ]
    )

    # declare net
    _, apply_fn = serial(
        LIF(
            hidden_size,
            n_spikes_hidden,
            t_max,
            p,
        ),
        LIF(
            output_size,
            n_spikes_output,
            t_max,
            p,
        ),
    )

    def first_spike(spikes: EventPropSpike, size: int):
        return np.array(
            [
                np.min(np.where(spikes.idx == idx, spikes.time, np.inf))
                for idx in range(size)
            ]
        )

    def loss_fn(
        weights,
        input_spikes,
    ):
        recording = apply_fn(weights, input_spikes)
        first_spikes = first_spike(recording[1], 3)
        loss = -np.log(
            np.sum(
                1
                + np.exp(-first_spikes[1] / p.tau_mem)
                - np.exp(-first_spikes / p.tau_mem)
            )
        )
        return loss, recording

    (loss, recording), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        params, input_spikes
    )
    assert not np.isnan(np.mean(grad[0].input))
    assert_almost_equal(loss, -1.1499, 4)


if __name__ == "__main__":
    test_nans()

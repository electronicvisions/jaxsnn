from functools import partial

import unittest
import jax.numpy as np
from jaxsnn.event.root.next import next_queue
from jaxsnn.event.types import Spike


class TestTasksYinYang(unittest.TestCase):
    def test_next_queue(self):
        known_spikes = Spike(
            time=np.array([0.5, 1.0, 1.5, 2.0]), idx=np.array([0, 7, 3, 1])
        )
        layer_start = 1
        next_fn = partial(next_queue, known_spikes, layer_start, None)

        # test spike in previous layer is ignored
        res = next_fn(0.3, t_max=4.0)
        assert res == Spike(time=np.array([1.0]), idx=np.array([6]))

        # test next spike is found
        res = next_fn(0.9, t_max=4.0)
        assert res == Spike(time=np.array([1.0]), idx=np.array([6]))

        res = next_fn(1.7, t_max=4.0)
        assert res == Spike(time=np.array([2.0]), idx=np.array([0]))

        # test t_max is respected
        res = next_fn(1.7, t_max=1.9)
        assert res == Spike(time=np.array([1.9]), idx=np.array([-1]))


if __name__ == '__main__':
    unittest.main()

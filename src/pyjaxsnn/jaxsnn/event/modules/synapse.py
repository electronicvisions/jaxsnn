from typing import Callable, Dict, Tuple

import jax
from jaxsnn.event.hardware.transforms import linear_saturating


def Synapse(  # pylint: disable=invalid-name
    mean: float = 0.5,
    std: float = 2.0,
    min_delay: float = 0.0,
    weight_scale: float = 1.0,
    transform: Callable = linear_saturating
) -> Tuple[Callable, Dict]:
    """
    Creates a synapse initialization function and associated parameters.

    :param mean: Mean value for initializing synaptic weights.
    :param std: Standard deviation for initializing synaptic weights.
    :param min_delay: Minimum allowable synaptic delay.
    :param weight_scale: Scaling factor for synaptic weights.
    :param transform: Transformation function applied to synaptic weights.

    :returns: A tuple containing:
        - gen: A generator function that provides an init function and a module
               generator.
        - parameters: A dictionary containing the synapse configuration.
    """

    def gen(
        input_size: int,
        output_size: int,
    ):
        """
        Generates initialization and module creation functions for a synapse.

        :param input_size: The number of input features for the synapse.
        :param output_size: The number of output features for the synapse.
        :return: A tuple containing:
            - init: A function that initializes the synapse weights using a
                    random number generator.
            - gen_bss2_module: A function that generates a Synapse module based
                               on the provided experiment configuration.
        """
        # Maybe switch this to old construct_init functions
        def init(rng: jax.Array):
            weights = jax.random.normal(rng, (input_size, output_size))
            return weights * std + mean

        return init, None

    parameters = {
        "layer_type": "synapse",
        "mean": mean,
        "std": std,
        "min_delay": min_delay,
        "weight_scale": weight_scale,
        "transform": transform
    }

    return gen, parameters

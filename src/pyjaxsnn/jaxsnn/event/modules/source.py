import jax
from jaxsnn.event.types import SourcePopulation


# pylint: disable=invalid-name
def Source(size: int) -> SourcePopulation:
    """
    Creates a Source population layer representing external input.

    :param size: Number of neurons/channels in the source layer.

    :returns: A SourcePopulation object containing the generator and
        parameters.
    """
    def generator() -> SourcePopulation.Functions:
        """
        Generates the initialization and state functions for the source layer.

        :returns: A SourcePopulation.Functions object containing init, state,
            and event functions.
        """

        # pylint: disable=unused-argument
        def init_fn(rng: jax.Array) -> None:
            return None

        def state_fn() -> None:
            # TODO: Should set layer_idx to inputs because not known beforehand
            return None

        # pylint: disable=unused-argument
        def event_fn(*args) -> None:
            return None

        return SourcePopulation.Functions(init_fn, state_fn, event_fn)

    parameters = {
        "size": size,
    }

    return SourcePopulation(generator, parameters, size)

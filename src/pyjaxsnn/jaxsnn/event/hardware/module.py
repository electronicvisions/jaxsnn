from typing import Any, Dict, Optional, Tuple, Union

import pygrenade_vx.network as grenade


class Module:
    """
    Module supplying basic functionality for building SNNs on hardware.
    """

    def __init__(self, experiment) -> None:
        """
        :param experiment: Experiment to append layer to.
        """
        super().__init__()
        self._changed_since_last_run = True
        self.experiment = experiment
        self.extra_args: Tuple[Any] = tuple()
        self.extra_kwargs: Dict[str, Any] = {}
        self.size: Optional[int] = None

        # Grenade descriptor
        self.descriptor: Union[
            grenade.PopulationOnNetwork, grenade.ProjectionOnNetwork
        ]

    @property
    def changed_since_last_run(self) -> bool:
        """
        Getter for changed_since_last_run.

        :returns: Boolean indicating wether module changed since last run.
        """
        return self._changed_since_last_run

    def reset_changed_since_last_run(self) -> None:
        """
        Reset changed_since_last_run. Sets the corresponding flag to false.
        """
        self._changed_since_last_run = False

    def register_hw_entity(self) -> None:
        raise NotImplementedError

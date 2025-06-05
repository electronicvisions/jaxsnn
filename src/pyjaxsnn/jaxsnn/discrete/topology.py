from typing import Dict, List, Optional, Tuple

import jax

from jaxsnn import get_logger
from jaxsnn.base.topology import BaseTopology
from jaxsnn.base.types import BaseState
from jaxsnn.discrete.types import (
    DenseData,
    Projection,
    Population,
    SourcePopulation,
    States,
    IOData,
    ModelApplyFn,
    Parameters,
)


class Topology(BaseTopology):
    """
    Represents a discrete-time spiking neural network (SNN) topology.

    Manages the graph of layers, their connections, and the execution of the
    network simulation over discrete time steps.

    :param time_steps: Number of simulation time steps.
    :param dt: Simulation time step size (in seconds).
    :param record_states: Whether to record and return the full state history.
    """

    def __init__(
        self,
        time_steps: int = 100,
        dt: float = 1e-6,
        record_states: bool = False
    ) -> None:
        """
        Initialize the discrete topology.

        :param time_steps: Number of simulation time steps.
        :param dt: Simulation time step size (in seconds).
        :param record_states: Whether to record and return the full state
            history.
        """
        super().__init__()
        self.time_steps = time_steps
        self.dt = dt  # pylint: disable=invalid-name
        self.record_states = record_states
        self.log = get_logger("jaxsnn.discrete.Topology")

    def generate_apply_fn(
        self,
        sccs_ordered: List[Tuple[str, ...]],
    ) -> ModelApplyFn:
        return self._build_trajectory(sccs_ordered)

    def attach_layer_fns(self) -> None:
        self.log.TRACE("Generate and attach layer functions to graph")
        for node, attrs in self.graph.nodes(data=True):
            module = attrs["module"]
            self.log.TRACE(f"Processing node: {node}")

            if isinstance(module, Projection):
                source_pop = list(self.graph.predecessors(node)).pop()
                target_pop = list(self.graph.successors(node)).pop()
                module.fns = module.generator(
                    self.graph.nodes[source_pop]["module"].size,
                    self.graph.nodes[target_pop]["module"].size,
                )

            if isinstance(module, (SourcePopulation, Population)):
                module.fns = module.generator(self.dt)

    def _build_trajectory(
        self,
        sccs_ordered: List[Tuple[str, ...]],
    ) -> ModelApplyFn:
        """
        Builds the trajectory function that executes the network simulation.

        :param sccs_ordered: List of strongly connected components in
            topological order.

        :return: The trajectory function.
        """
        def trajectory(
            inputs: IOData,
            parameters: Parameters,
        ) -> Tuple[Optional[States], IOData]:
            """
            Executes the network simulation for the given inputs and
            parameters.

            :param inputs: Input data for the simulation (time-major).
            :param parameters: Network parameters (weights, etc.).

            :return: A tuple containing the recorded states (if enabled) and
                the outputs.
            """
            def model_step_fn(
                carry: Tuple[States, IOData, Parameters],
                time_slice: Dict[str, jax.Array],
            ) -> Tuple[Tuple[States, IOData, Parameters],
                       Tuple[States, IOData]]:
                """Execute one time step of the network dynamics."""
                states, prev_outputs, parameters = carry

                # Initialize step outputs and state containers
                new_states: Dict[str, Optional[BaseState]] = states.copy()
                step_result: Dict[str, DenseData] = {}

                # Process nodes in topological order (SCC-based)
                for scc in sccs_ordered:
                    for node in scc:
                        # Select module
                        module = self.graph.nodes[node]["module"]

                        # Find input to current module
                        node_in = {}
                        for pred in self.graph.predecessors(node):
                            if pred in time_slice:
                                node_in[pred] = time_slice[pred]
                            elif pred in step_result:
                                node_in[pred] = step_result[pred]
                            elif pred in prev_outputs:
                                node_in[pred] = prev_outputs[pred]

                        # Special handling for SourcePopulation nodes
                        if isinstance(module, SourcePopulation):
                            assert not list(self.graph.predecessors(node)), (
                                "SourcePopulation nodes should not have "
                                "predecessors"
                            )
                            node_in = {node: time_slice[node]}

                        # Execute node dynamics for one time step
                        new_state, output = module.fns.step(
                            node_in,
                            states[node],
                            parameters[node]
                        )

                        # Update state and output containers
                        step_result[node] = output
                        new_states[node] = new_state

                return (
                    (new_states, step_result, parameters),
                    (new_states, step_result),
                )

            # Inits
            init_states, init_outputs = {}, {}
            for scc in sccs_ordered:
                for node in scc:
                    init_states[node], init_outputs[node] = \
                        self.graph.nodes[node]["module"].fns.state()

            # Scan over leading input axis (time)
            _, (states, outputs) = jax.lax.scan(
                model_step_fn,
                (init_states, init_outputs, parameters),
                inputs,
            )

            if self.record_states:
                return states, outputs

            return None, outputs

        return trajectory

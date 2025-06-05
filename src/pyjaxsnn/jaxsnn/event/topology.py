from typing import Any, Dict, List, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jaxsnn import get_logger
from jaxsnn.base.topology import BaseTopology
try:
    from jaxsnn.event.hardware.experiment import Experiment
except ImportError:
    Experiment = Any  # Fallback if the module is not available

try:
    from jaxsnn.event.hardware.modules.population import (
        Population as HXPopulation,
        InputPopulation as HXInputPopulation
    )
except ImportError:
    HXPopulation = Any  # Fallback if the module is not available
    HXInputPopulation = Any  # Fallback if the module is not available
from jaxsnn.event.adjoint.trajectory import adjoint_trajectory
from jaxsnn.event.adjoint.step import multi_layer_adjoint_step
from jaxsnn.event.functional import trajectory
from jaxsnn.event.stepping import multi_layer_step
from jaxsnn.event.types import (
    Spike,
    AdjointStepFn,
    Parameters,
    IOData,
    States,
    Projection,
    Population,
    SourcePopulation,
    EventStepFn,
    ModelApplyFn,
)

import networkx as nx


class Topology(BaseTopology):
    """
    Represents a spiking neural network (SNN) topology as a directed
    graph of layers.

    Stores layer generators with associated parameters (nodes) and the
    connections (edges) between them. Handles construction of cleaned
    graphs, strongly connected components (SCCs), and simulation
    trajectories for efficient event-based computation. The backward
    pass can be configured to use EventProp or analytical gradients.

    :params t_max: Maximum time for the simulation.
    :params backprop_method: Either "eventprop" for the EventProp algorithm or
        "analytical" for the time-to-first-spike solver.
    :params mock: If True, the topology will not be connected to a BSS2
        experiment and will only run in software simulation mode.
    :params inter_batch_entry_wait: Wait time between batch entries in
            FPGA cycles.
    :params has_external_events: If True, the recordings of an externally
        performed forward pass are expected to be provided.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        t_max: float,
        backprop_method: str = 'analytical',
        mock: bool = True,
        inter_batch_entry_wait: int = 0,
        has_external_events: bool = False,
    ):
        """Initialize an empty topology."""
        super().__init__()
        self.graph: nx.DiGraph = nx.DiGraph()
        # TODO: Make this layer specific
        self.backprop_method = backprop_method
        # TODO: Maybe remove t_max all together
        self.t_max = t_max
        self.experiment: Optional[Experiment] = None
        if not mock:
            self.experiment = Experiment(
                self,
                inter_batch_entry_wait=inter_batch_entry_wait,
            )
            self.experiment.runtime_in_s = t_max
        self.mock = mock
        self.has_external_events = has_external_events
        self.log = get_logger("jaxsnn.event.Topology")

    def extract_bss2_data(self, node):
        """Extract BSS2 data from the given node's HX module."""
        return self.graph.nodes[node]["hx_module"].hw_observables

    def generate_apply_fn(
        self,
        sccs_ordered: List[Tuple[str, ...]],
    ) -> ModelApplyFn:
        """
        Generate the main apply function for the SNN model.

        Constructs a function that executes the network simulation by iterating
        through the ordered strongly connected components (SCCs). It handles
        both software simulation (via JAX scan) and hardware execution (via
        custom VJP and callback).

        :param sccs_ordered: List of SCCs in topological order.
        :returns: A callable ModelApplyFn that takes inputs and parameters
            and returns the network output.
        """
        # Build trajectories for each SCC
        trajectories = self._build_trajectories(sccs_ordered)

        # Define apply function for the entire topology
        def apply_fn(
            inputs: IOData,
            parameters: Parameters,
            external_events: Optional[IOData],
        ) -> Tuple[Optional[States], IOData]:
            # Generate layer states
            states = {}
            events: Dict[str, Optional[Spike]] = {}
            for scc in sccs_ordered:
                for node in scc:
                    module = self.graph.nodes[node]["module"]
                    states[node] = module.fns.state()
                    events[node] = module.fns.event(module.n_steps)

            # Insert input events at corresponding places
            for node, input_events in inputs.items():
                events[node] = input_events

            # Loop over SCCs while skipping input layers
            for scc in sccs_ordered:
                if trajectories[scc] is None:
                    continue

                # TODO: Fix queue head size
                # +1 to increment self, not just input queues
                # TODO: Move to build trajectories?
                queue_heads = {
                    node: jnp.array(
                        [0] * (len(list(self.graph.predecessors(node))) + 1))
                    for node in scc
                }

                events, states, _, _ = trajectories[scc](
                    parameters,
                    events,
                    external_events,
                    states,
                    queue_heads,
                )

            return events

        if self.mock:
            if self.has_external_events:
                return jax.vmap(apply_fn, in_axes=(0, None, 0))
            return jax.vmap(
                partial(apply_fn, external_events=None), in_axes=(0, None)
            )

        apply_fn_batched = jax.vmap(apply_fn, in_axes=(0, None, 0))
        hx_run = self._construct_hx_run()

        @jax.custom_vjp
        def hx_dispatch(
            input_spikes: IOData,
            params: Parameters,
        ) -> IOData:
            expected_return_type = self.experiment.expected_return_type(
                input_spikes
            )
            hx_result = jax.pure_callback(
                hx_run,
                expected_return_type,
                input_spikes,
                params,
            )
            hx_observables = apply_fn_batched(
                input_spikes,
                params,
                hx_result,
            )
            return hx_observables

        def hx_dispatch_fwd(
            input_spikes: IOData,
            params: Parameters,
        ) -> Tuple[IOData, Tuple[IOData, Parameters, IOData]]:
            expected_return_type = self.experiment.expected_return_type(
                input_spikes
            )
            hx_result = jax.pure_callback(
                hx_run,
                expected_return_type,
                input_spikes,
                params,
            )
            hx_observables, vjp_fn = jax.vjp(
                apply_fn_batched,
                input_spikes,
                params,
                hx_result,
            )
            return hx_observables, vjp_fn

        def hx_dispatch_bwd(
            res: Tuple[IOData, Parameters, IOData],
            g: Tuple[Parameters, IOData]  # pylint: disable=invalid-name
        ) -> Tuple[Parameters, IOData]:
            grad_spikes = g
            vjp_fn = res

            grad_model_spikes, grad_model_params, _ = vjp_fn(grad_spikes)

            return grad_model_spikes, grad_model_params

        hx_dispatch.defvjp(hx_dispatch_fwd, hx_dispatch_bwd)

        return hx_dispatch

    def attach_layer_fns(self) -> None:
        """
        Generate and attach functional closures to each graph node.

        Iterates through the graph nodes and calls the generator function
        of each module to create its specific initialization, state, and
        step functions. These functions are then attached to the module
        instance stored in the graph node.
        """
        self.log.info("Generate and attach layer functions to graph")
        for node, attrs in self.graph.nodes(data=True):
            module = attrs["module"]
            self.log.TRACE(f"Processing node: {node}")

            if isinstance(module, Projection):
                source_pop = list(self.graph.predecessors(node)).pop()
                target_pop = list(self.graph.successors(node)).pop()
                # TODO: Maybe dont assign to module but graph node
                module.fns = module.generator(
                    self.graph.nodes[source_pop]["module"].size,
                    self.graph.nodes[target_pop]["module"].size,
                )

            if isinstance(module, Population):
                min_delays = {}
                for pop_node, param_node in zip(
                        self.pre_population_nodes(node), self.pre_nodes(node)):
                    pre_proj = self.graph.nodes[param_node]["module"]
                    assert isinstance(pre_proj, Projection)
                    min_delays[pop_node] = pre_proj.min_delay

                module.fns = module.generator(
                    self.pre_population_nodes(node),
                    self.pre_nodes(node),
                    min_delays,
                    self.node_index_mapping,
                    self.t_max,
                    node,
                    list(self.node_scc_mapping[node]),
                    self.backprop_method,
                )

            if isinstance(module, SourcePopulation):
                module.fns = module.generator()

    def _build_trajectories(
        self,
        ordered_sccs: List[Tuple[str, ...]],
    ) -> Dict[Tuple[str, ...], EventStepFn]:
        """
        Construct trajectories for each strongly connected component (SCC).

        For SCCs with multiple nodes, builds a multi-layer step function and
        optionally a custom backward pass for 'eventprop' backprop method.
        For single-node SCCs, returns the apply function directly except for
        input layers where it returns None.

        :param ordered_sccs: List of strongly connected components in
            topological order.

        :returns: Dictionary mapping SCC tuples to their corresponding
            trajectory functions.
        """
        trajectories = {}
        for scc in ordered_sccs:
            if (len(scc) == 1 and not isinstance(
                    self.graph.nodes[scc[0]]["module"], Population)):
                trajectories[scc] = None
                continue

            pop_modules: Dict[str, Population] = {}
            for node in scc:
                module = self.graph.nodes[node]["module"]
                if isinstance(module, Population):
                    pop_modules[node] = module

            n_steps = max(module.n_steps for module in pop_modules.values())

            step_fns: Dict[str, EventStepFn] = {}
            pop_nodes: List[str] = []
            for node, module in pop_modules.items():
                module.n_steps = n_steps
                step_fns[node] = module.fns.step
                pop_nodes.append(node)

            this_trajectory = partial(
                trajectory,
                partial(
                    multi_layer_step,
                    step_fns,
                    pop_nodes,
                    self.node_index_mapping,
                ),
                n_steps,
            )

            if self.backprop_method == "eventprop":
                if len(pop_nodes) > 1:
                    self.log.warning(
                        "For multi-layer SCCs, the EventProp backward pass "
                        "might result in inaccurate gradients."
                    )
                adjoint_step_fns: Dict[str, AdjointStepFn] = {}
                for node, module in pop_modules.items():
                    if module.fns.adjoint_step is not None:
                        adjoint_step_fns[node] = module.fns.adjoint_step

                this_adjoint_trajectory = partial(
                    adjoint_trajectory,
                    partial(
                        multi_layer_adjoint_step,
                        adjoint_step_fns,
                        pop_nodes,
                    ),
                    n_steps,
                )

                def make_fwd_trajectory(traj_fn):
                    def fwd(*args, **kwargs):
                        res = traj_fn(*args, **kwargs)
                        return res, res
                    return fwd

                this_trajectory_fwd = make_fwd_trajectory(this_trajectory)
                this_trajectory = jax.custom_vjp(this_trajectory)

                this_trajectory.defvjp(
                    this_trajectory_fwd,
                    this_adjoint_trajectory,
                )

            trajectories[scc] = this_trajectory

        return trajectories

    def _construct_hx_run(self):
        """
        Construct the BSS2 run function for hardware simulation.

        This function is used to run the experiment on the hardware backend
        and is wrapped in a JAX custom VJP function for gradient computation.

        :returns: A callable that runs the BSS2 experiment.
        """
        self.log.TRACE("Generate HX experiment run function")

        # Add all populations first
        for node, attrs in self.graph.nodes(data=True):
            module = attrs["module"]
            if not isinstance(module, (SourcePopulation, Population)):
                continue
            assert module.fns.hx_module is not None
            self.graph.nodes[node]["hx_module"] = module.fns.hx_module(
                self.node_index_mapping[node],
                self.experiment,
                None,
                None
            )

        # Add all projections
        for node, attrs in self.graph.nodes(data=True):
            module = attrs["module"]
            if not isinstance(module, Projection):
                continue
            source_node = list(self.graph.predecessors(node)).pop()
            target_node = list(self.graph.successors(node)).pop()
            source_pop = self.graph.nodes[source_node]["hx_module"]
            target_pop = self.graph.nodes[target_node]["hx_module"]
            assert isinstance(source_pop, (HXInputPopulation, HXPopulation))
            assert isinstance(target_pop, (HXInputPopulation, HXPopulation))
            assert module.fns.hx_module is not None
            self.graph.nodes[node]["hx_module"] = module.fns.hx_module(
                self.node_index_mapping[node],
                self.experiment,
                source_pop,
                target_pop,
            )

        return self.experiment.run

from __future__ import annotations
from typing import (
    List,
    Dict,
    Optional,
    TYPE_CHECKING,
)
from functools import partial

from dlens_vx_v3 import halco
from pyhalco_hicann_dls_vx_v3 import DLSGlobal

import jax
import jax.numpy as jnp

from jaxsnn.event.types import (
    EventStepFn,
    Spike,
    Population,
    StepState,
)
from jaxsnn.event.hardware.experiment import Experiment
from jaxsnn.event.hardware.modules.neuron import Neuron
from jaxsnn.event.hardware.modules.base_module import BaseModule
from jaxsnn.event.states import LIFState
from jaxsnn.event.stepping.step_existing_events import step_existing
from jaxsnn.event.modules.lif.lif import build_adjoint_step_function
from jaxsnn.event.modules.lif.parameters import LIFParameters
from jaxsnn.event.modules.hx.lif.parameters import NeuronParameters
from jaxsnn.event.functional.lif.transition import transition
from jaxsnn.event.functional.lif.dynamics import lif_exponential_flow

if TYPE_CHECKING:
    from hxtorch.core.morphology import Morphology


# pylint: disable=too-many-arguments
def HXLIF(  # pylint: disable=invalid-name, too-many-locals
    size: int,
    n_steps: int,
    params: NeuronParameters,
    time_offset: float = 0.0,
    n_hw_spikes: Optional[int] = None,
    chip_coordinate: Optional[DLSGlobal] = None,
    enable_spike_recording: bool = True,
    enable_cadc_recording: bool = False,
    enable_cadc_recording_placement_in_dram: bool = False,
    enable_madc_recording: bool = False,
    record_neuron_id: Optional[int] = None,
    placement_constraint: Optional[List[halco.LogicalNeuronOnDLS]] = None,
    neuron_structure: Optional[Morphology] = None,
    **extra_params
) -> Population:
    """
    Create a LIF neuron population for execution on BrainScaleS.

    :param size: Number of neurons in the population.
    :param n_steps: Number of simulation steps.
    :param params: Neuron parameters (hardware-compatible).
    :param time_offset: Optional time offset for all neurons.
    :param n_hw_spikes: Optional hardware spike limit.
    :param chip_coordinate: Optional chip coordinate for placement.
    :param enable_spike_recording: Enable spike recording.
    :param enable_cadc_recording: Enable CADC recording.
    :param enable_cadc_recording_placement_in_dram: Place CADC recording in
        DRAM.
    :param enable_madc_recording: Enable MADC recording.
    :param record_neuron_id: Restrict recording to a specific neuron.
    :param placement_constraint: Placement constraints for neurons.
    :param neuron_structure: Optional neuron morphology.
    :param extra_params: Additional hardware-specific parameters.

    :return: Population object for hardware LIF neurons.
    """

    def generator(  # pylint: disable=too-many-locals, too-many-arguments
        pre_layer_pop_nodes: List[str],
        pre_layer_param_nodes: List[str],
        pre_layer_params: Dict[str, float],
        node_index_mapping: Dict[str, int],
        t_max: float,
        node: str,
        scc_mask: List[str],  # pylint: disable=unused-argument
        backprop_method: str
    ) -> Population.Functions:
        """
        Generate all functional closures for the hardware LIF neuron
        population.

        :param pre_layer_pop_nodes: Names of presynaptic population nodes.
        :param pre_layer_param_nodes: Names of presynaptic parameter nodes.
        :param pre_layer_params: Parameters for presynaptic connections.
        :param node_index_mapping: Mapping of node names to indices.
        :param t_max: Maximum simulation time.
        :param node: Name of this node.
        :param scc_mask: Strongly connected component mask.
        :param backprop_method: Backpropagation method (must be 'eventprop').

        :return: Population.Functions with all closures for simulation and
            hardware.
        """

        def init_fn(
            rng: jax.Array,  # pylint: disable=unused-argument
        ) -> None:
            """
            Initialize the neuron layer state for hardware LIF neurons.

            :param rng: JAX random key (unused).

            :return: None (no learnable parameters on hardware supported yet).
            """
            return None

        def state_fn() -> StepState:
            """
            Create the initial StepState for the neuron layer.

            :return: StepState with zeroed LIFState and time.
            """
            state = StepState(
                LIFState(jnp.zeros(size), jnp.zeros(size)),
                jnp.array(0.0)
            )
            return state

        def event_fn(
            n_steps: int,
        ) -> Spike:
            """
            Generate an empty Spike object for the given number of event steps.

            :param n_steps: Number of event steps.

            :return: Empty Spike object.
            """
            return Spike.empty(n_steps)

        def generate_hx_module_fn(
            layer_idx: int,
            experiment: Experiment,
            source: Optional[BaseModule] = None,  # pylint: disable=unused-argument
            target: Optional[BaseModule] = None,  # pylint: disable=unused-argument
        ) -> Neuron:
            """
            Generate the HX neuron module for hardware simulation.

            :param experiment: Experiment context or configuration.

            :returns: Instantiated Neuron module for HX backend.
            """
            hx_module = Neuron(
                layer_idx=layer_idx,
                n_events=n_steps,
                n_hw_spikes=n_hw_spikes,
                size=size,
                time_offset=time_offset,
                experiment=experiment,
                chip_coordinate=chip_coordinate,
                **params.as_dict(),
                enable_spike_recording=enable_spike_recording,
                enable_cadc_recording=enable_cadc_recording,
                enable_cadc_recording_placement_in_dram=(
                    enable_cadc_recording_placement_in_dram),
                enable_madc_recording=enable_madc_recording,
                record_neuron_id=record_neuron_id,
                placement_constraint=placement_constraint,
                neuron_structure=neuron_structure,
                **extra_params,
            )
            return hx_module

        if backprop_method != 'eventprop':
            raise NotImplementedError(
                f"Backpropagation method {backprop_method} is not implemented."
            )

        lif_params = LIFParameters(
            tau_syn=params.tau_syn.model_value,
            tau_mem=params.tau_mem.model_value,
            v_th=params.v_th.model_value,
            v_leak=params.v_leak.model_value,
            v_reset=params.v_reset.model_value,
        )

        step_fn = build_step_function(
            pre_layer_pop_nodes,
            pre_layer_param_nodes,
            pre_layer_params,
            lif_params,
            node_index_mapping,
            t_max,
            node,
        )

        # Build adjoint step function for eventprop
        adjoint_step_fn = build_adjoint_step_function(
            pre_layer_pop_nodes,
            pre_layer_param_nodes,
            node_index_mapping,
            lif_params,
            t_max
        )

        return Population.Functions(
            init=init_fn,
            state=state_fn,
            step=step_fn,
            event=event_fn,
            adjoint_step=adjoint_step_fn,
            hx_module=generate_hx_module_fn,
        )

    parameters = {
        "size": size,
        "n_steps": n_steps,
        "time_offset": time_offset,
        "chip_coordinate": chip_coordinate,
        "enable_spike_recording": enable_spike_recording,
        "enable_cadc_recording": enable_cadc_recording,
        "enable_cadc_recording_placement_in_dram":
            enable_cadc_recording_placement_in_dram,
        "enable_madc_recording": enable_madc_recording,
        "record_neuron_id": record_neuron_id,
        "placement_constraint": placement_constraint,
        "neuron_structure": neuron_structure,
        **extra_params,
    }

    return Population(generator, parameters, size, n_steps)


# pylint: disable=too-many-arguments, too-many-locals
def build_step_function(
    pre_layer_pop_nodes: List[str],
    pre_layer_param_nodes: List[str],
    pre_layer_params: Dict[str, float],  # pylint: disable=unused-argument
    lif_params: LIFParameters,
    node_idx_mapping: Dict[str, int],
    t_max: float,
    node: str,
) -> EventStepFn:
    """
    Build the step function for a hardware LIF neuron layer.

    Assembles the solver, neuron dynamics, and transition functions for both
    recurrent and feedforward simulation contexts.

    :param pre_layer_pop_nodes: Names of presynaptic population nodes.
    :param pre_layer_param_nodes: Names of presynaptic parameter nodes.
    :param pre_layer_params: Parameters for presynaptic connections.
    :param lif_params: LIF neuron model parameters.
    :param node_idx_mapping: Mapping of node names to indices.
    :param t_max: Maximum simulation time.
    :param node: Name of this node.

    :return: EventStepFn for the LIF neuron layer.
    """
    # Vectorized flow dynamics
    single_flow = lif_exponential_flow(
        jnp.array(lif_params.tau_syn),
        jnp.array(lif_params.tau_mem),
    )
    dynamics_fn = jax.vmap(single_flow, in_axes=(0, None))

    # Transitions per input connection
    input_layer_idxs = [node_idx_mapping[x] for x in pre_layer_pop_nodes]
    max_input_idx = max(input_layer_idxs) if input_layer_idxs else -1
    num_nodes = max(max_input_idx, node_idx_mapping[node]) + 1
    transition_fns = [
        lambda s, w, i, l: s for _ in range(num_nodes)
    ]  # Default no-op transition

    # input transitions
    # TODO: Split transition fns
    for layer_idx, pre_pop_node in zip(
        input_layer_idxs, pre_layer_param_nodes
    ):
        transition_fns[layer_idx] = \
            partial(transition, lif_params.v_reset, pre_pop_node)

    # internal transition
    if node_idx_mapping[node] not in input_layer_idxs:
        transition_fns[node_idx_mapping[node]] = \
            partial(transition, lif_params.v_reset, pre_layer_param_nodes[0])

    # Recurrent and non-recurrent step functions
    step_fn = partial(
        step_existing,
        pre_layer_pop_nodes,
        dynamics_fn,
        transition_fns,
        node,
        t_max,
    )

    return step_fn

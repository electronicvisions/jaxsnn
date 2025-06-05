from typing import (
    Dict,
    List,
    Tuple,
    TypeVar,
    Generic,
    cast,
)
from abc import (
    ABC,
    abstractmethod,
)

import networkx as nx
import jax

from jaxsnn import get_logger
from jaxsnn.base.types import BasePopulation


ModelInitFnT = TypeVar('ModelInitFnT')
ModelApplyFnT = TypeVar('ModelApplyFnT')
ModuleT = TypeVar("ModuleT")


class AbstractTopology(ABC, Generic[ModelInitFnT, ModelApplyFnT]):

    @abstractmethod
    def add(self, modules: Dict[str, ModuleT]) -> None:
        """
        Add modules to the topology graph.

        :param modules: Dictionary mapping module names to modules.
        """

    @abstractmethod
    def connect(self, edges: List[Tuple[str, str]]) -> None:
        """
        Add directed edges between nodes in the topology graph.

        :param edges: List of (source, target) tuples.
        """

    @abstractmethod
    def done(self) -> Tuple[ModelInitFnT, ModelApplyFnT]:
        """
        Finalize the topology and return (init_fn, apply_fn).

        :return: Tuple of initialization and application functions.
        """


class BaseTopology(AbstractTopology[ModelInitFnT, ModelApplyFnT]):
    """
    Represents a spiking neural network (SNN) topology as a directed
    graph of layers.

    Stores layer generators with associated parameters (nodes) and the
    connections (edges) between them. Handles construction of cleaned
    graphs, strongly connected components (SCCs), and simulation
    trajectories for efficient event-based computation. The backward
    pass can be configured to use EventProp or analytical gradients.
    """

    def __init__(self) -> None:
        """Initialize an empty topology."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self.log = get_logger("jaxsnn.base.Topology")
        self.node_index_mapping: Dict[str, int] = {}
        self.node_scc_mapping: Dict[str, Tuple[str, ...]] = {}

    def add(
        self,
        modules: Dict[str, ModuleT],
    ) -> None:
        """
        Add a dictionary of nodes to the topology.

        :param nodes: A dictionary mapping node names to layer definitions.
        """
        for name, module in modules.items():
            self.log.TRACE(f"Adding node {name} of type {type(module)}.")
            self.graph.add_node(name, module=module)

    def connect(
        self,
        edges: List[Tuple[str, str]]
    ) -> None:
        """
        Add directed edges between nodes in the topology.

        :param edges: List of (source, target) node name tuples.
        """
        for src, dst in edges:
            self.log.TRACE(f"Adding edge from {src} to {dst}.")
            self.graph.add_edge(src, dst)

    def pre_process(self) -> List[Tuple[str, ...]]:
        sccs = list(nx.strongly_connected_components(self.graph))
        sccs_graph = nx.condensation(self.graph, sccs)
        sccs_ordered = [
            tuple(sccs_graph.nodes[scc]["members"])
            for scc in nx.topological_sort(sccs_graph)
        ]

        i = 0
        self.node_index_mapping: Dict[str, int] = {}
        for scc in sccs_ordered:
            for node in scc:
                self.node_index_mapping[node] = i
                i += 1
        self.node_scc_mapping: Dict[str, Tuple[str, ...]] = {
            node: scc for scc in sccs_ordered for node in scc
        }
        self.attach_layer_fns()

        return sccs_ordered

    def _get_pre_population_nodes(self, node: str) -> List[str]:
        """
        Recursively collect all presynaptic population nodes for a given node.

        :param node: Node name to trace predecessors for.

        :return: List of presynaptic population node names.
        """
        stack = list(self.graph.predecessors(node))

        pops = []
        while True:
            if not stack:
                return pops
            node = stack.pop()
            if isinstance(
                self.graph.nodes[node]["module"], BasePopulation
            ):
                pops.append(node)
                continue
            stack += list(self.graph.predecessors(node))
        return pops

    def pre_population_nodes(self, node: str) -> List[str]:
        return sorted(
            self._get_pre_population_nodes(node),
            key=lambda x: self.node_index_mapping[x]
        )

    def pre_nodes(self, node: str) -> List[str]:
        return sorted(
            self.graph.predecessors(node),
            key=lambda x: self.node_index_mapping[x]
        )

    def generate_init_fn(
        self,
        sccs_ordered: List[Tuple[str, ...]],
    ) -> ModelInitFnT:
        """
        Generate the model initialization function for the topology.

        :param sccs_ordered: List of strongly connected components in
            topological order.

        :return: Model initialization function.
        """
        init_fns = {
            node: self.graph.nodes[node]["module"].fns.init
            for scc in sccs_ordered for node in scc
        }

        def init(rng: jax.Array):
            params = {}
            for node, init_fn in init_fns.items():
                if init_fn is None:
                    params[node] = None
                else:
                    this_rng, rng = jax.random.split(rng)
                    params[node] = init_fn(this_rng)
            return params
        return cast(ModelInitFnT, init)

    @abstractmethod
    def generate_apply_fn(
        self,
        sccs_ordered: List[Tuple[str, ...]],
    ) -> ModelApplyFnT:
        """
        Generate the model application (forward pass) function for the
        topology.

        :param sccs_ordered: List of strongly connected components in
            topological order.

        :return: Model application function.
        """

    @abstractmethod
    def attach_layer_fns(self) -> None:
        """
        Attach functional closures (init/state/step) to each graph node.
        """

    def done(self) -> Tuple[ModelInitFnT, ModelApplyFnT]:
        """
        Finalize the graph topology into one init/apply pair.

        Constructs initialization and forward pass functions based on the
        topology and user-defined node generators. This includes automatic
        SCC decomposition, and handling of the forward and backward dynamics.

        :returns: A tuple containing:
            - init: function to initialize weights
            - apply: function to apply the topology to inputs
        """
        sccs_ordered = self.pre_process()
        init_fn = self.generate_init_fn(sccs_ordered)
        apply_fn = self.generate_apply_fn(sccs_ordered)
        return init_fn, apply_fn

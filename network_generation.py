"""
Network Topology Generation and Candidate Path Computation

This module generates quantum repeater network topologies with controlled
bottleneck structures to force resource contention, and computes K-shortest
candidate paths for multi-flow entanglement routing.

Scientific Background:
- Quantum repeater networks consist of nodes with finite memory qubits
- Links have entanglement generation rates and decoherence properties
- Multi-flow routing creates resource contention that classical greedy methods struggle with
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import itertools


@dataclass
class NodeProperties:
    """Quantum repeater node properties"""

    node_id: int
    num_memory_qubits: int  # Finite memory capacity
    t1_relaxation: float  # Amplitude damping time (microseconds)
    t2_coherence: float  # Dephasing time (microseconds)


@dataclass
class LinkProperties:
    """Quantum link properties"""

    source: int
    target: int
    distance: float  # Physical distance in km
    alpha_db_per_km: float  # Fiber attenuation coefficient
    generation_rate: float  # Bell pairs per second
    initial_fidelity: float  # Initial Bell pair fidelity
    bsm_success_prob: float  # Bell state measurement success probability
    bsm_error_rate: float  # BSM depolarizing error rate


@dataclass
class CommunicationDemand:
    """Source-destination pair requiring entanglement"""

    demand_id: int
    source: int
    destination: int
    priority: float  # For greedy baseline ordering
    min_fidelity: float  # Quality of service requirement
    max_latency: float  # Latency constraint in milliseconds


@dataclass
class CandidatePath:
    """A candidate path for routing a demand"""

    demand_id: int
    path_id: int
    nodes: List[int]  # Ordered list of nodes from source to destination
    edges: List[Tuple[int, int]]  # Ordered list of edges

    def num_links(self) -> int:
        return len(self.edges)

    def num_swaps(self) -> int:
        """Number of entanglement swapping operations required"""
        return max(0, len(self.edges) - 1)


class QuantumRepeaterNetwork:
    """
    Quantum repeater network with realistic hardware parameters.

    Supports multiple topology types designed to create resource contention:
    - Barbell: Two dense clusters connected by narrow bridge
    - Grid: Rectangular lattice with cut vertices
    - Sparse random: Low connectivity forcing path overlap
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.graph = None
        self.node_props = {}
        self.link_props = {}

    def generate_barbell_network(
        self, cluster_size: int = 4, bridge_width: int = 1, avg_memory_qubits: int = 4
    ) -> nx.Graph:
        """
        Generate barbell topology: two dense clusters with narrow bridge.

        This creates contention when demands must traverse the bridge,
        forcing trade-offs between path quality and resource availability.

        Args:
            cluster_size: Number of nodes in each cluster
            bridge_width: Number of nodes in the connecting bridge
            avg_memory_qubits: Average memory capacity per node

        Returns:
            NetworkX graph with node and edge properties
        """
        # Create two complete graphs (clusters)
        left_cluster = nx.complete_graph(cluster_size)
        right_cluster = nx.complete_graph(cluster_size)

        # Relabel right cluster to avoid node ID collision
        right_cluster = nx.relabel_nodes(
            right_cluster,
            {i: i + cluster_size + bridge_width for i in range(cluster_size)},
        )

        # Create bridge nodes
        bridge_nodes = list(range(cluster_size, cluster_size + bridge_width))

        # Combine into single graph
        G = nx.Graph()
        G.add_nodes_from(left_cluster.nodes())
        G.add_nodes_from(right_cluster.nodes())
        G.add_nodes_from(bridge_nodes)
        G.add_edges_from(left_cluster.edges())
        G.add_edges_from(right_cluster.edges())

        # Connect left cluster to bridge
        for i in range(min(2, cluster_size)):
            G.add_edge(i, bridge_nodes[0])

        # Connect bridge nodes sequentially
        for i in range(len(bridge_nodes) - 1):
            G.add_edge(bridge_nodes[i], bridge_nodes[i + 1])

        # Connect bridge to right cluster
        for i in range(min(2, cluster_size)):
            G.add_edge(bridge_nodes[-1], cluster_size + bridge_width + i)

        self._assign_network_properties(G, avg_memory_qubits)
        self.graph = G
        return G

    def generate_grid_network(
        self, rows: int = 3, cols: int = 4, avg_memory_qubits: int = 4
    ) -> nx.Graph:
        """
        Generate 2D grid topology with cut vertices.

        Grid topologies create natural bottlenecks at central nodes
        where multiple shortest paths converge.

        Args:
            rows: Number of rows in grid
            cols: Number of columns in grid
            avg_memory_qubits: Average memory capacity per node

        Returns:
            NetworkX grid graph with properties
        """
        G = nx.grid_2d_graph(rows, cols)

        # Relabel nodes to integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

        self._assign_network_properties(G, avg_memory_qubits)
        self.graph = G
        return G

    def generate_sparse_random_network(
        self, n_nodes: int = 10, edge_prob: float = 0.3, avg_memory_qubits: int = 4
    ) -> nx.Graph:
        """
        Generate sparse random graph with controlled connectivity.

        Low edge probability ensures limited path diversity, forcing
        demands to share resources.

        Args:
            n_nodes: Number of nodes
            edge_prob: Probability of edge between any two nodes
            avg_memory_qubits: Average memory capacity per node

        Returns:
            Connected random graph with properties
        """
        # Generate random graph, ensure connectivity
        while True:
            G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=self.seed)
            if nx.is_connected(G):
                break
            self.seed += 1  # Try different seed if not connected

        self._assign_network_properties(G, avg_memory_qubits)
        self.graph = G
        return G

    def _assign_network_properties(self, G: nx.Graph, avg_memory_qubits: int):
        """
        Assign realistic hardware properties to nodes and links.

        Node properties:
        - Memory qubits: Poisson distributed around average
        - T1: 100-500 microseconds (typical for solid-state qubits)
        - T2: 50-200 microseconds (always T2 <= T1)

        Link properties:
        - Distance: 5-50 km (metropolitan network scale)
        - Attenuation: 0.16-0.25 dB/km (standard fiber)
        - Generation rate: 1-10 MHz (modern sources)
        - Initial fidelity: 0.90-0.99 (state-of-the-art)
        - BSM success: 0.7-0.9 (detector efficiency limited)
        - BSM error: 0.01-0.05 (imperfect measurements)
        """
        # Assign node properties
        for node in G.nodes():
            # Memory qubits: Poisson distribution, minimum 2
            num_qubits = max(2, self.rng.poisson(avg_memory_qubits))

            # T1 relaxation time: 100-500 microseconds
            t1 = self.rng.uniform(100, 500)

            # T2 coherence time: 50-200 microseconds, constrained T2 <= T1
            t2 = min(self.rng.uniform(50, 200), t1)

            self.node_props[node] = NodeProperties(
                node_id=node,
                num_memory_qubits=num_qubits,
                t1_relaxation=t1,
                t2_coherence=t2,
            )

        # Assign link properties
        for edge in G.edges():
            source, target = edge

            # Physical distance: 5-50 km
            distance = self.rng.uniform(5, 50)

            # Fiber attenuation: 0.16-0.25 dB/km
            alpha = self.rng.uniform(0.16, 0.25)

            # Generation rate: 1-10 MHz
            gen_rate = self.rng.uniform(1e6, 10e6)

            # Initial fidelity: 0.90-0.99
            init_fid = self.rng.uniform(0.90, 0.99)

            # BSM success probability: 0.7-0.9
            bsm_success = self.rng.uniform(0.7, 0.9)

            # BSM error rate: 0.01-0.05
            bsm_error = self.rng.uniform(0.01, 0.05)

            self.link_props[edge] = LinkProperties(
                source=source,
                target=target,
                distance=distance,
                alpha_db_per_km=alpha,
                generation_rate=gen_rate,
                initial_fidelity=init_fid,
                bsm_success_prob=bsm_success,
                bsm_error_rate=bsm_error,
            )

            # Store symmetric edge
            if (target, source) not in self.link_props:
                self.link_props[(target, source)] = self.link_props[edge]

    def generate_communication_demands(
        self, num_demands: int, contention_level: float = 0.5
    ) -> List[CommunicationDemand]:
        """
        Generate communication demands with controlled contention.

        Contention is controlled by selecting source-destination pairs
        that force paths through network bottlenecks.

        Args:
            num_demands: Number of source-destination pairs
            contention_level: 0.0 (low overlap) to 1.0 (high overlap)

        Returns:
            List of communication demands
        """
        if self.graph is None:
            raise ValueError("Must generate network topology first")

        demands = []
        nodes = list(self.graph.nodes())

        # For barbell topology, force contention by routing across bridge
        if self._is_barbell_topology():
            demands = self._generate_barbell_demands(num_demands)
        else:
            # Random source-destination pairs
            for i in range(num_demands):
                # Sample two distinct nodes
                src, dst = self.rng.choice(nodes, size=2, replace=False)

                # Priority: uniform for fairness testing
                priority = self.rng.uniform(0, 1)

                # QoS requirements
                min_fidelity = self.rng.uniform(0.7, 0.9)
                max_latency = self.rng.uniform(10, 100)  # milliseconds

                demands.append(
                    CommunicationDemand(
                        demand_id=i,
                        source=src,
                        destination=dst,
                        priority=priority,
                        min_fidelity=min_fidelity,
                        max_latency=max_latency,
                    )
                )

        return demands

    def _is_barbell_topology(self) -> bool:
        """Heuristic to detect barbell topology"""
        # Check for low connectivity nodes (bridge)
        degrees = dict(self.graph.degree())
        avg_degree = np.mean(list(degrees.values()))
        min_degree = min(degrees.values())
        return min_degree < 0.5 * avg_degree

    def _generate_barbell_demands(self, num_demands: int) -> List[CommunicationDemand]:
        """Generate demands that cross the barbell bridge"""
        demands = []
        nodes = list(self.graph.nodes())

        # Identify left and right clusters by connectivity
        # Nodes with low degree are in bridge, others in clusters
        degrees = dict(self.graph.degree())
        avg_degree = np.mean(list(degrees.values()))

        left_cluster = []
        right_cluster = []

        for node in nodes:
            if degrees[node] >= 0.7 * avg_degree:
                # High degree node in a cluster
                # Use connectivity to partition
                neighbors = set(self.graph.neighbors(node))
                left_count = sum(1 for n in left_cluster if n in neighbors)
                right_count = sum(1 for n in right_cluster if n in neighbors)

                if len(left_cluster) == 0 or left_count > right_count:
                    left_cluster.append(node)
                else:
                    right_cluster.append(node)

        # Generate demands crossing clusters
        for i in range(num_demands):
            if len(left_cluster) > 0 and len(right_cluster) > 0:
                src = self.rng.choice(left_cluster)
                dst = self.rng.choice(right_cluster)
            else:
                # Fallback to random
                src, dst = self.rng.choice(nodes, size=2, replace=False)

            demands.append(
                CommunicationDemand(
                    demand_id=i,
                    source=src,
                    destination=dst,
                    priority=self.rng.uniform(0, 1),
                    min_fidelity=self.rng.uniform(0.7, 0.9),
                    max_latency=self.rng.uniform(10, 100),
                )
            )

        return demands


class CandidatePathGenerator:
    """
    Compute K-shortest candidate paths for each demand using Yen's algorithm.

    Candidate paths provide the discrete solution space for QAOA optimization.
    We precompute K diverse paths per demand to balance solution quality with
    problem size (M demands × K candidates = M*K qubits).
    """

    def __init__(self, network: QuantumRepeaterNetwork):
        self.network = network
        self.graph = network.graph

    def compute_candidate_paths(
        self,
        demands: List[CommunicationDemand],
        k: int = 3,
        weight_metric: str = "hops",
    ) -> Dict[int, List[CandidatePath]]:
        """
        Compute K candidate paths for each demand using Yen's algorithm.

        Args:
            demands: List of communication demands
            k: Number of candidate paths per demand
            weight_metric: 'hops' (shortest) or 'distance' (physical length)

        Returns:
            Dictionary mapping demand_id to list of candidate paths
        """
        candidate_paths = {}

        # Set edge weights based on metric
        if weight_metric == "distance":
            edge_weights = {
                edge: self.network.link_props[edge].distance
                for edge in self.graph.edges()
            }
        else:
            edge_weights = {edge: 1.0 for edge in self.graph.edges()}

        nx.set_edge_attributes(self.graph, edge_weights, "weight")

        for demand in demands:
            paths = []

            try:
                # Yen's K-shortest paths algorithm
                k_shortest = list(
                    nx.shortest_simple_paths(
                        self.graph, demand.source, demand.destination, weight="weight"
                    )
                )[:k]

                for path_id, node_path in enumerate(k_shortest):
                    # Convert node path to edge path
                    edge_path = [
                        (node_path[i], node_path[i + 1])
                        for i in range(len(node_path) - 1)
                    ]

                    paths.append(
                        CandidatePath(
                            demand_id=demand.demand_id,
                            path_id=path_id,
                            nodes=node_path,
                            edges=edge_path,
                        )
                    )
            except nx.NetworkXNoPath:
                print(
                    f"Warning: No path found for demand {demand.demand_id} "
                    f"from {demand.source} to {demand.destination}"
                )
                continue

            candidate_paths[demand.demand_id] = paths

        return candidate_paths

    def get_path_resource_usage(self, path: CandidatePath) -> Dict[int, int]:
        """
        Compute memory qubit usage at each node for a given path.

        Each path requires 2 memory qubits at intermediate nodes
        (one for each adjacent link) and 1 at endpoints.

        Args:
            path: Candidate path

        Returns:
            Dictionary mapping node_id to number of qubits required
        """
        usage = {}

        for i, node in enumerate(path.nodes):
            if i == 0 or i == len(path.nodes) - 1:
                # Endpoint: 1 qubit
                usage[node] = 1
            else:
                # Intermediate node: 2 qubits (one per adjacent link)
                usage[node] = 2

        return usage


def main():
    """Demonstration of network generation and path computation"""

    # Generate barbell network
    network = QuantumRepeaterNetwork(seed=42)
    G = network.generate_barbell_network(cluster_size=4, bridge_width=2)

    print(f"Generated barbell network:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Average degree: {np.mean([d for n, d in G.degree()]):.2f}")

    # Print node properties
    print("\nNode properties:")
    for node, props in sorted(network.node_props.items()):
        print(
            f"  Node {node}: {props.num_memory_qubits} qubits, "
            f"T1={props.t1_relaxation:.1f}μs, T2={props.t2_coherence:.1f}μs"
        )

    # Generate demands
    demands = network.generate_communication_demands(num_demands=4)
    print(f"\nGenerated {len(demands)} communication demands:")
    for d in demands:
        print(
            f"  Demand {d.demand_id}: {d.source} → {d.destination}, "
            f"priority={d.priority:.2f}, min_fid={d.min_fidelity:.2f}"
        )

    # Compute candidate paths
    path_gen = CandidatePathGenerator(network)
    candidate_paths = path_gen.compute_candidate_paths(demands, k=3)

    print("\nCandidate paths:")
    for demand_id, paths in candidate_paths.items():
        print(f"  Demand {demand_id}:")
        for path in paths:
            print(
                f"    Path {path.path_id}: {path.nodes} "
                f"({path.num_links()} links, {path.num_swaps()} swaps)"
            )
            usage = path_gen.get_path_resource_usage(path)
            print(f"      Resource usage: {usage}")


if __name__ == "__main__":
    main()

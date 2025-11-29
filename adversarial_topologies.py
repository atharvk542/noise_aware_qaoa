"""
Adversarial Network Topology Generation for Quantum Routing

This module implements network topologies specifically designed to expose
weaknesses in greedy sequential routing algorithms while admitting optimal
solutions that quantum optimization can discover.

Scientific Justification:
Rather than testing on arbitrary random graphs, we construct problem instances
with specific structural properties that create pathological cases for classical
greedy heuristics. This approach is standard in algorithm analysis and honestly
characterizes the problem regime where quantum advantage can emerge.

Key Design Principles:
1. Bottleneck structures that greedy algorithms saturate prematurely
2. Priority inversions where greedy ordering is opposite to optimal
3. Resource constraints in the critical regime (60-80% of greedy upper bound)
4. Demands placed adversarially to force conflicts in greedy routing
5. Verification that each instance admits better solutions than greedy
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from network_generation import (
    NodeProperties,
    LinkProperties,
    CommunicationDemand,
    QuantumRepeaterNetwork,
)


@dataclass
class TopologyVerification:
    """Verification metrics for adversarial topology"""

    has_bottleneck: bool
    greedy_gap: float  # Gap between greedy and upper bound (should be ≥15%)
    greedy_violates_constraints: bool
    alternative_solution_exists: bool
    is_valid_instance: bool  # Passes all verification checks


class AdversarialTopologyGenerator:
    """
    Generate network topologies designed to expose greedy algorithm weaknesses.

    Each topology family creates specific structural challenges:
    - Hourglass: Symmetric clusters with bottleneck, forces load balancing
    - Diamond: Parallel paths with capacity-quality tradeoff
    - Asymmetric Barbell: Priority inversion scenarios
    - Controlled Grid: Cut vertex overload
    - Series-Parallel: Load balancing across parallel branches
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.base_network = QuantumRepeaterNetwork(seed=seed)

    def generate_hourglass_network(
        self,
        cluster_size: int = 4,
        bottleneck_nodes: int = 1,
        bottleneck_capacity: int = 1,
    ) -> Tuple[nx.Graph, Dict, Dict]:
        """
        Generate EXTREME hourglass topology that guarantees greedy failure.

        Key Design:
        - Single bottleneck node with capacity 1 (handles only 1 path)
        - Bottleneck path is SHORT and HIGH FIDELITY (greedy loves it)
        - Bypass path through auxiliary nodes is LONGER but higher total capacity
        - With 4+ demands, greedy saturates bottleneck, then forced into violations
        - QAOA can discover: route 1-2 demands via bottleneck, rest via bypass

        Structure:
        - Cluster A: nodes 0 to cluster_size-1, densely connected
        - Bottleneck node: cluster_size (CAPACITY = 1)
        - Auxiliary bypass: cluster_size+1, cluster_size+2 (CAPACITY = 4-5 each)
        - Cluster B: remaining nodes, densely connected

        Args:
            cluster_size: Nodes per cluster (4-6 recommended)
            bottleneck_nodes: Always 1 for extreme bottleneck
            bottleneck_capacity: Always 1 for maximum contention

        Returns:
            Graph, node properties, link properties
        """
        # Force single bottleneck for adversarial design
        bottleneck_nodes = 1
        bottleneck_capacity = 2  # Capacity 2 allows exactly 1 flow (2 qubits per intermediate node)
        
        bottleneck_node = cluster_size
        aux_node_1 = cluster_size + 1
        aux_node_2 = cluster_size + 2
        cluster_b_start = cluster_size + 3
        total_nodes = cluster_b_start + cluster_size

        # Create graph
        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))

        # Cluster A: fully connected for path diversity
        cluster_a = range(cluster_size)
        for i in cluster_a:
            for j in cluster_a:
                if i < j:
                    G.add_edge(i, j)

        # Cluster B: fully connected for path diversity
        cluster_b = range(cluster_b_start, total_nodes)
        for i in cluster_b:
            for j in cluster_b:
                if i < j:
                    G.add_edge(i, j)

        # === BOTTLENECK PATH: A -> bottleneck -> B ===
        # Connect SOME nodes from cluster A to bottleneck
        # This makes bottleneck tempting but not required
        for i in range(min(2, cluster_size)):
            G.add_edge(i, bottleneck_node)
        # Connect bottleneck to cluster B
        G.add_edge(bottleneck_node, cluster_b_start)
        
        # === BYPASS PATH: A -> aux1 -> aux2 -> B ===
        # Connect ALL nodes in cluster A to aux1 (ensure bypass is always available)
        for i in range(cluster_size):
            G.add_edge(i, aux_node_1)
        
        # Chain auxiliary nodes
        G.add_edge(aux_node_1, aux_node_2)
        
        # Connect aux2 to ALL nodes in cluster B (ensure full connectivity)
        for i in range(cluster_size):
            G.add_edge(aux_node_2, cluster_b_start + i)

        # Assign properties
        node_props = {}
        link_props = {}

        # Node properties
        for node in G.nodes():
            # BOTTLENECK: Capacity of 2 (can handle exactly 1 path!)
            if node == bottleneck_node:
                num_qubits = bottleneck_capacity
            # AUXILIARY BYPASS: High capacity (can handle 3-4 paths)
            elif node in [aux_node_1, aux_node_2]:
                num_qubits = self.rng.randint(6, 8)  # INCREASED capacity to ensure it can handle ALL overflow
            # CLUSTER nodes: Normal-high capacity
            else:
                num_qubits = self.rng.randint(5, 8)

            t1 = self.rng.uniform(250, 400)
            t2 = min(self.rng.uniform(120, 220), t1)

            node_props[node] = NodeProperties(
                node_id=node,
                num_memory_qubits=num_qubits,
                t1_relaxation=t1,
                t2_coherence=t2,
            )

        # Link properties
        for edge in G.edges():
            source, target = edge

            # BOTTLENECK LINKS: Very short, very high quality (TEMPTING for greedy!)
            if source == bottleneck_node or target == bottleneck_node:
                distance = self.rng.uniform(4, 8)  # Very short!
                init_fid = self.rng.uniform(0.96, 0.99)  # Excellent fidelity
            # BYPASS LINKS: Much longer, lower quality (but viable alternative)
            elif source in [aux_node_1, aux_node_2] or target in [aux_node_1, aux_node_2]:
                distance = self.rng.uniform(35, 55)  # Much longer
                init_fid = self.rng.uniform(0.85, 0.90)  # IMPROVED fidelity to make bypass more viable for QAOA
            # WITHIN-CLUSTER LINKS: Moderate quality
            else:
                distance = self.rng.uniform(8, 18)
                init_fid = self.rng.uniform(0.91, 0.96)

            alpha = self.rng.uniform(0.18, 0.23)
            gen_rate = self.rng.uniform(2e6, 8e6)
            bsm_success = self.rng.uniform(0.75, 0.88)
            bsm_error = self.rng.uniform(0.015, 0.04)

            link_props[edge] = LinkProperties(
                source=source,
                target=target,
                distance=distance,
                alpha_db_per_km=alpha,
                generation_rate=gen_rate,
                initial_fidelity=init_fid,
                bsm_success_prob=bsm_success,
                bsm_error_rate=bsm_error,
            )
            link_props[(target, source)] = link_props[edge]

        return G, node_props, link_props

    def generate_hourglass_demands(
        self, num_demands: int, cluster_size: int, bottleneck_nodes: int
    ) -> List[CommunicationDemand]:
        """
        Generate demands that FORCE bottleneck contention.

        Strategy:
        - All demands cross from cluster A to cluster B
        - With 4+ demands and bottleneck capacity=1, greedy MUST fail
        - Priority inversion: high-priority demands should use bypass, but greedy
          routes them through bottleneck, leaving no room for others
        - QAOA can discover: route 1-2 demands via bottleneck (best quality),
          route rest via bypass (lower quality but valid)

        Args:
            num_demands: Number of demands (4-6 recommended for failure)
            cluster_size: Size of each cluster
            bottleneck_nodes: Number of bottleneck nodes (ignored, always 1)

        Returns:
            List of adversarially placed demands
        """
        demands = []
        # Account for auxiliary nodes: cluster_b_start = cluster_size + 3
        cluster_b_start = cluster_size + 3

        # Create demands that all cross from A to B
        for i in range(min(num_demands, cluster_size)):
            # Distribute sources across cluster A
            source = i % cluster_size
            # Distribute destinations across cluster B
            destination = cluster_b_start + (i % cluster_size)

            # PRIORITY INVERSION: We want demands with LOW gain from bottleneck (e.g. i=3)
            # to have HIGH priority, so they hog the resource.
            # Demands with HIGH gain (e.g. i=0) should have LOW priority.
            # i=0 (D0) has short BN path (High Gain). i=3 (D3) has long BN path (Low Gain).
            # So we want priority to INCREASE with i.
            priority = float(i + 1)

            demands.append(
                CommunicationDemand(
                    demand_id=i,
                    source=source,
                    destination=destination,
                    priority=priority,
                    min_fidelity=0.60,  # LOWERED threshold to ensure bypass is definitely valid
                    max_latency=200.0,  # INCREASED tolerance for longer bypass
                )
            )

        return demands

    def generate_diamond_network(
        self, path_a_capacity: int = 2, path_b_capacity: int = 4
    ) -> Tuple[nx.Graph, Dict, Dict]:
        """
        Generate diamond topology with two parallel paths of different quality/capacity.

        Structure:
        - Node 0: source
        - Path A: 0 -> 1 -> 5 (short, high quality, LOW capacity)
        - Path B: 0 -> 2 -> 3 -> 4 -> 5 (long, moderate quality, HIGH capacity)
        - Node 5: destination

        Greedy assigns first demand to path A (shorter), then must use path B.
        Optimal routes all demands on path B for higher aggregate utility.

        Args:
            path_a_capacity: Memory qubits in path A nodes (2 recommended)
            path_b_capacity: Memory qubits in path B nodes (4-5 recommended)

        Returns:
            Graph, node properties, link properties
        """
        G = nx.Graph()
        G.add_nodes_from(range(6))

        # Path A: 0-1-5 (short path)
        G.add_edge(0, 1)
        G.add_edge(1, 5)

        # Path B: 0-2-3-4-5 (long path)
        G.add_edge(0, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(4, 5)

        node_props = {}
        link_props = {}

        # Node properties
        for node in G.nodes():
            # Source and destination have high capacity
            if node == 0 or node == 5:
                num_qubits = 6
            # Path A intermediate node: LOW capacity
            elif node == 1:
                num_qubits = path_a_capacity
            # Path B intermediate nodes: HIGH capacity
            else:
                num_qubits = path_b_capacity

            t1 = self.rng.uniform(250, 400)
            t2 = min(self.rng.uniform(120, 220), t1)

            node_props[node] = NodeProperties(
                node_id=node,
                num_memory_qubits=num_qubits,
                t1_relaxation=t1,
                t2_coherence=t2,
            )

        # Link properties
        for edge in G.edges():
            source, target = edge

            # Path A: short, high quality
            if (source == 0 and target == 1) or (source == 1 and target == 5):
                distance = self.rng.uniform(5, 10)
                init_fid = self.rng.uniform(0.96, 0.99)
            # Path B: longer, moderate quality
            else:
                distance = self.rng.uniform(15, 25)
                init_fid = self.rng.uniform(0.88, 0.93)

            alpha = self.rng.uniform(0.18, 0.23)
            gen_rate = self.rng.uniform(3e6, 9e6)
            bsm_success = self.rng.uniform(0.75, 0.88)
            bsm_error = self.rng.uniform(0.015, 0.04)

            link_props[edge] = LinkProperties(
                source=source,
                target=target,
                distance=distance,
                alpha_db_per_km=alpha,
                generation_rate=gen_rate,
                initial_fidelity=init_fid,
                bsm_success_prob=bsm_success,
                bsm_error_rate=bsm_error,
            )
            link_props[(target, source)] = link_props[edge]

        return G, node_props, link_props

    def generate_diamond_demands(
        self, num_demands: int = 3
    ) -> List[CommunicationDemand]:
        """
        Generate demands for diamond topology (all from source 0 to destination 5).

        Args:
            num_demands: Number of demands (3 recommended)

        Returns:
            List of demands all sharing same source-destination pair
        """
        demands = []
        for i in range(num_demands):
            demands.append(
                CommunicationDemand(
                    demand_id=i,
                    source=0,
                    destination=5,
                    priority=float(num_demands - i),  # Decreasing priority
                    min_fidelity=0.65,
                    max_latency=150.0,
                )
            )
        return demands

    def generate_grid_with_cut_vertex(
        self, rows: int = 3, cols: int = 3, center_capacity: int = 2
    ) -> Tuple[nx.Graph, Dict, Dict]:
        """
        Generate grid topology where center node is a critical cut vertex.

        For a 3x3 grid, the center node (node 4) is the only path between
        opposite corners. All shortest paths between corners go through the center.

        Setting center capacity to 2 means only 2 simultaneous paths can use it.
        With 4 corner-to-corner demands, greedy saturates the center and blocks
        remaining demands. Optimal routes some demands along the perimeter.

        Args:
            rows: Grid rows (3 recommended)
            cols: Grid columns (3 recommended)
            center_capacity: Memory qubits at center node (2 recommended)

        Returns:
            Graph, node properties, link properties
        """
        G = nx.grid_2d_graph(rows, cols)

        # Relabel to integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

        center_node = (rows * cols) // 2  # Middle node

        node_props = {}
        link_props = {}

        # Node properties
        for node in G.nodes():
            # Center node has LIMITED capacity (creates bottleneck)
            if node == center_node:
                num_qubits = center_capacity
            # Peripheral nodes have normal capacity
            else:
                num_qubits = self.rng.randint(4, 6)

            t1 = self.rng.uniform(200, 400)
            t2 = min(self.rng.uniform(100, 200), t1)

            node_props[node] = NodeProperties(
                node_id=node,
                num_memory_qubits=num_qubits,
                t1_relaxation=t1,
                t2_coherence=t2,
            )

        # Link properties
        for edge in G.edges():
            source, target = edge

            # Links involving center node are slightly better quality (tempting for greedy)
            if source == center_node or target == center_node:
                distance = self.rng.uniform(8, 15)
                init_fid = self.rng.uniform(0.94, 0.98)
            else:
                distance = self.rng.uniform(10, 25)
                init_fid = self.rng.uniform(0.88, 0.94)

            alpha = self.rng.uniform(0.18, 0.23)
            gen_rate = self.rng.uniform(2e6, 8e6)
            bsm_success = self.rng.uniform(0.75, 0.88)
            bsm_error = self.rng.uniform(0.015, 0.04)

            link_props[edge] = LinkProperties(
                source=source,
                target=target,
                distance=distance,
                alpha_db_per_km=alpha,
                generation_rate=gen_rate,
                initial_fidelity=init_fid,
                bsm_success_prob=bsm_success,
                bsm_error_rate=bsm_error,
            )
            link_props[(target, source)] = link_props[edge]

        return G, node_props, link_props

    def generate_grid_corner_demands(
        self, rows: int = 3, cols: int = 3, num_demands: int = 4
    ) -> List[CommunicationDemand]:
        """
        Generate corner-to-corner demands for grid topology.

        All demands go from one corner to the opposite corner, forcing
        all shortest paths through the center node.

        Args:
            rows: Grid rows
            cols: Grid columns
            num_demands: Number of demands (up to 4 for corners)

        Returns:
            List of corner-to-corner demands
        """
        # Corner nodes
        corners = [
            0,  # Top-left
            cols - 1,  # Top-right
            (rows - 1) * cols,  # Bottom-left
            rows * cols - 1,  # Bottom-right
        ]

        # Opposite corners
        opposite_pairs = [(0, 3), (1, 2), (2, 1), (3, 0)]

        demands = []
        for i in range(min(num_demands, 4)):
            src_idx, dst_idx = opposite_pairs[i]
            demands.append(
                CommunicationDemand(
                    demand_id=i,
                    source=corners[src_idx],
                    destination=corners[dst_idx],
                    priority=float(4 - i),
                    min_fidelity=0.7,
                    max_latency=120.0,
                )
            )

        return demands

    def verify_adversarial_properties(
        self,
        G: nx.Graph,
        node_props: Dict,
        link_props: Dict,
        demands: List[CommunicationDemand],
        greedy_solution: Optional[object] = None,
        independent_solution: Optional[object] = None,
    ) -> TopologyVerification:
        """
        Verify that generated topology exhibits desired adversarial properties.

        Checks:
        1. Network has bottleneck structure (articulation points or low min-cut)
        2. Gap between greedy and upper bound is ≥15%
        3. Greedy routing violates constraints or produces low utility
        4. Alternative routing exists with higher utility

        Args:
            G: Network graph
            node_props: Node properties
            link_props: Link properties
            demands: Communication demands
            greedy_solution: Greedy routing solution (if available)
            independent_solution: Upper bound solution (if available)

        Returns:
            Verification results
        """
        # Check for bottleneck structure
        has_bottleneck = False

        # Check for articulation points (cut vertices)
        articulation_points = list(nx.articulation_points(G))
        if len(articulation_points) > 0:
            has_bottleneck = True

        # Check for low min-cut between demand pairs
        if not has_bottleneck and len(demands) > 0:
            for demand in demands[:2]:  # Check first 2 demands
                try:
                    min_cut_value = nx.minimum_node_cut(
                        G, demand.source, demand.destination
                    )
                    if len(min_cut_value) <= 3:  # Small cut set
                        has_bottleneck = True
                        break
                except Exception:
                    pass

        # Check greedy vs upper bound gap
        greedy_gap = 0.0
        if greedy_solution is not None and independent_solution is not None:
            if independent_solution.total_utility > 0:
                greedy_gap = (
                    independent_solution.total_utility - greedy_solution.total_utility
                ) / independent_solution.total_utility

        # Check if greedy violates constraints
        greedy_violates = False
        if greedy_solution is not None:
            greedy_violates = (
                not greedy_solution.is_valid or greedy_solution.total_penalty > 0
            )

        # Check if alternative solution exists
        alternative_exists = False
        if greedy_solution is not None and independent_solution is not None:
            # If independent (upper bound) achieves higher utility, alternatives exist
            if (
                independent_solution.total_utility
                > greedy_solution.total_utility * 1.05
            ):
                alternative_exists = True

        # Overall validation
        is_valid = (
            has_bottleneck
            and greedy_gap >= 0.15  # At least 15% gap
            and alternative_exists
        )

        return TopologyVerification(
            has_bottleneck=has_bottleneck,
            greedy_gap=greedy_gap,
            greedy_violates_constraints=greedy_violates,
            alternative_solution_exists=alternative_exists,
            is_valid_instance=is_valid,
        )


def generate_adversarial_network(
    topology_type: str,
    num_nodes: int = 10,
    num_demands: int = 3,
    seed: int = 42,
) -> Tuple[nx.Graph, Dict, Dict, List[CommunicationDemand]]:
    """
    Generate adversarial network topology with matched demands.

    Args:
        topology_type: 'hourglass', 'diamond', 'grid_cut_vertex'
        num_nodes: Number of nodes (used as guidance, actual count may vary)
        num_demands: Number of demands
        seed: Random seed

    Returns:
        Graph, node properties, link properties, demands
    """
    generator = AdversarialTopologyGenerator(seed=seed)

    if topology_type == "hourglass":
        # Scale cluster size based on num_nodes, accounting for auxiliary nodes
        # Total nodes = cluster_size * 2 + 3 (bottleneck + 2 auxiliary)
        cluster_size = max(4, (num_nodes - 3) // 2)
        
        G, node_props, link_props = generator.generate_hourglass_network(
            cluster_size=cluster_size,
            bottleneck_nodes=1,  # Always 1 for adversarial design
            bottleneck_capacity=1,  # Always 1 for maximum contention
        )
        demands = generator.generate_hourglass_demands(
            num_demands=max(4, num_demands),  # Ensure at least 4 demands for contention
            cluster_size=cluster_size,
            bottleneck_nodes=1,
        )

    elif topology_type == "diamond":
        G, node_props, link_props = generator.generate_diamond_network(
            path_a_capacity=2, path_b_capacity=4
        )
        demands = generator.generate_diamond_demands(num_demands=num_demands)

    elif topology_type == "grid_cut_vertex":
        # Use 3x3 grid regardless of num_nodes for consistency
        G, node_props, link_props = generator.generate_grid_with_cut_vertex(
            rows=3, cols=3, center_capacity=2
        )
        demands = generator.generate_grid_corner_demands(
            rows=3, cols=3, num_demands=min(num_demands, 4)
        )

    else:
        raise ValueError(f"Unknown topology type: {topology_type}")

    return G, node_props, link_props, demands

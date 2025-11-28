"""
Classical Baseline Algorithms for Multi-Flow Entanglement Routing

This module implements classical heuristics for comparison with QAOA:
1. Sequential Greedy: Route demands in priority order, each gets best available path
2. Independent Shortest Path: Upper bound assuming infinite resources
3. Random Path Selection: Lower bound sanity check

These baselines are essential for scientific validation. We must demonstrate
statistically significant improvements over greedy to claim quantum advantage.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
import time
from copy import deepcopy

from network_generation import (
    QuantumRepeaterNetwork,
    CandidatePath,
    CommunicationDemand,
)
from noise_models import EntanglementQualitySimulator, ResourceConflictDetector
from qaoa_optimizer import RoutingConfiguration


class SequentialGreedyRouter:
    """
    Sequential greedy heuristic for entanglement routing.

    Algorithm:
    1. Sort demands by priority (highest first)
    2. For each demand in order:
        a. Find best available path (highest utility, no conflicts)
        b. Allocate that path
        c. Update available resources
    3. Return complete routing configuration

    This is the primary classical baseline. It's efficient and widely used,
    but tends to favor early demands at expense of later ones, missing
    global optima where balanced allocation performs better.
    """

    def __init__(
        self,
        network: QuantumRepeaterNetwork,
        demands: List[CommunicationDemand],
        candidate_paths: Dict[int, List[CandidatePath]],
        fidelity_sim: EntanglementQualitySimulator,
    ):
        self.network = network
        self.demands = demands
        self.candidate_paths = candidate_paths
        self.fidelity_sim = fidelity_sim
        self.conflict_detector = ResourceConflictDetector(network)

    def solve(self) -> RoutingConfiguration:
        """
        Run greedy routing algorithm.

        Returns:
            Routing configuration
        """
        start_time = time.time()

        print("\n" + "=" * 70)
        print("SEQUENTIAL GREEDY BASELINE")
        print("=" * 70)

        # Sort demands by priority (descending)
        sorted_demands = sorted(self.demands, key=lambda d: d.priority, reverse=True)

        # Track resource usage
        node_usage = {node: 0 for node in self.network.graph.nodes()}
        link_usage = {}

        # Selected paths
        path_selections = {}
        selected_paths = []

        # Process each demand
        for demand in sorted_demands:
            demand_id = demand.demand_id
            paths = self.candidate_paths[demand_id]

            # Find best available path
            best_path = None
            best_utility = float("-inf")

            for path in paths:
                # Check if path is available (no resource conflicts)
                if self._is_path_available(path, node_usage, link_usage):
                    utility = self.fidelity_sim.compute_path_utility(path)

                    if utility > best_utility:
                        best_utility = utility
                        best_path = path

            # Allocate best path (or first path if none available)
            if best_path is None:
                # No path satisfies constraints, take first one anyway
                best_path = paths[0]
                print(
                    f"  Warning: No available path for demand {demand_id}, "
                    f"forcing path {best_path.path_id}"
                )

            path_selections[demand_id] = best_path.path_id
            selected_paths.append(best_path)

            # Update resource usage
            self._allocate_path(best_path, node_usage, link_usage)

            print(
                f"  Demand {demand_id} (priority={demand.priority:.2f}): "
                f"Selected path {best_path.path_id}, utility={best_utility:.4f}"
            )

        # Evaluate final configuration
        total_utility = sum(
            self.fidelity_sim.compute_path_utility(path) for path in selected_paths
        )

        total_penalty = self.conflict_detector.compute_total_violation_penalty(
            selected_paths, penalty_weight=10.0
        )

        objective = total_utility - total_penalty
        is_valid = total_penalty == 0

        elapsed_time = time.time() - start_time

        print(f"\nGreedy routing complete in {elapsed_time:.3f} seconds")
        print(
            f"Valid: {is_valid}, Utility: {total_utility:.4f}, "
            f"Penalty: {total_penalty:.4f}, Objective: {objective:.4f}"
        )

        return RoutingConfiguration(
            path_selections=path_selections,
            total_utility=total_utility,
            total_penalty=total_penalty,
            objective_value=objective,
            is_valid=is_valid,
        )

    def _is_path_available(
        self,
        path: CandidatePath,
        node_usage: Dict[int, int],
        link_usage: Dict[Tuple[int, int], int],
    ) -> bool:
        """
        Check if path can be allocated without exceeding capacities.

        Args:
            path: Candidate path
            node_usage: Current node memory usage
            link_usage: Current link bandwidth usage

        Returns:
            True if path is available
        """
        # Check node capacities
        for i, node in enumerate(path.nodes):
            # Determine qubit requirement
            if i == 0 or i == len(path.nodes) - 1:
                qubits_needed = 1
            else:
                qubits_needed = 2

            new_usage = node_usage.get(node, 0) + qubits_needed
            capacity = self.network.node_props[node].num_memory_qubits

            if new_usage > capacity:
                return False

        # Check link capacities (max 3 users per link)
        max_users = 3
        for edge in path.edges:
            canonical_edge = tuple(sorted(edge))
            new_usage = link_usage.get(canonical_edge, 0) + 1

            if new_usage > max_users:
                return False

        return True

    def _allocate_path(
        self,
        path: CandidatePath,
        node_usage: Dict[int, int],
        link_usage: Dict[Tuple[int, int], int],
    ):
        """
        Update resource usage after allocating a path.

        Args:
            path: Allocated path
            node_usage: Node memory usage dictionary (modified in-place)
            link_usage: Link bandwidth usage dictionary (modified in-place)
        """
        # Update node usage
        for i, node in enumerate(path.nodes):
            if i == 0 or i == len(path.nodes) - 1:
                qubits_needed = 1
            else:
                qubits_needed = 2

            node_usage[node] = node_usage.get(node, 0) + qubits_needed

        # Update link usage
        for edge in path.edges:
            canonical_edge = tuple(sorted(edge))
            link_usage[canonical_edge] = link_usage.get(canonical_edge, 0) + 1


class IndependentShortestPathRouter:
    """
    Upper bound: Route each demand on its optimal path ignoring constraints.

    This represents the best possible performance if resources were unlimited.
    Any practical algorithm must perform at or below this bound.

    Useful for computing approximation ratios.
    """

    def __init__(
        self,
        network: QuantumRepeaterNetwork,
        demands: List[CommunicationDemand],
        candidate_paths: Dict[int, List[CandidatePath]],
        fidelity_sim: EntanglementQualitySimulator,
    ):
        self.network = network
        self.demands = demands
        self.candidate_paths = candidate_paths
        self.fidelity_sim = fidelity_sim
        self.conflict_detector = ResourceConflictDetector(network)

    def solve(self) -> RoutingConfiguration:
        """
        Select best path for each demand independently.

        Returns:
            Routing configuration (likely has constraint violations)
        """
        start_time = time.time()

        print("\n" + "=" * 70)
        print("INDEPENDENT SHORTEST PATH (UPPER BOUND)")
        print("=" * 70)

        path_selections = {}
        selected_paths = []

        for demand in self.demands:
            demand_id = demand.demand_id
            paths = self.candidate_paths[demand_id]

            # Find path with highest utility
            best_path = max(
                paths, key=lambda p: self.fidelity_sim.compute_path_utility(p)
            )

            path_selections[demand_id] = best_path.path_id
            selected_paths.append(best_path)

            utility = self.fidelity_sim.compute_path_utility(best_path)
            print(
                f"  Demand {demand_id}: Path {best_path.path_id}, utility={utility:.4f}"
            )

        # Evaluate configuration
        total_utility = sum(
            self.fidelity_sim.compute_path_utility(path) for path in selected_paths
        )

        total_penalty = self.conflict_detector.compute_total_violation_penalty(
            selected_paths, penalty_weight=10.0
        )

        objective = total_utility - total_penalty
        is_valid = total_penalty == 0

        elapsed_time = time.time() - start_time

        print(f"\nIndependent routing complete in {elapsed_time:.3f} seconds")
        print(
            f"Valid: {is_valid}, Utility: {total_utility:.4f}, "
            f"Penalty: {total_penalty:.4f}, Objective: {objective:.4f}"
        )

        return RoutingConfiguration(
            path_selections=path_selections,
            total_utility=total_utility,
            total_penalty=total_penalty,
            objective_value=objective,
            is_valid=is_valid,
        )


class RandomPathRouter:
    """
    Lower bound: Randomly select one path per demand.

    Provides a sanity check. Any reasonable algorithm should beat random.
    """

    def __init__(
        self,
        network: QuantumRepeaterNetwork,
        demands: List[CommunicationDemand],
        candidate_paths: Dict[int, List[CandidatePath]],
        fidelity_sim: EntanglementQualitySimulator,
        seed: int = None,
    ):
        self.network = network
        self.demands = demands
        self.candidate_paths = candidate_paths
        self.fidelity_sim = fidelity_sim
        self.conflict_detector = ResourceConflictDetector(network)
        self.rng = np.random.RandomState(seed)

    def solve(self) -> RoutingConfiguration:
        """
        Randomly select path for each demand.

        Returns:
            Routing configuration
        """
        start_time = time.time()

        path_selections = {}
        selected_paths = []

        for demand in self.demands:
            demand_id = demand.demand_id
            paths = self.candidate_paths[demand_id]

            # Random selection
            path_id = self.rng.randint(0, len(paths))
            path = paths[path_id]

            path_selections[demand_id] = path_id
            selected_paths.append(path)

        # Evaluate configuration
        total_utility = sum(
            self.fidelity_sim.compute_path_utility(path) for path in selected_paths
        )

        total_penalty = self.conflict_detector.compute_total_violation_penalty(
            selected_paths, penalty_weight=10.0
        )

        objective = total_utility - total_penalty
        is_valid = total_penalty == 0

        elapsed_time = time.time() - start_time

        return RoutingConfiguration(
            path_selections=path_selections,
            total_utility=total_utility,
            total_penalty=total_penalty,
            objective_value=objective,
            is_valid=is_valid,
        )


def main():
    """Demonstration of classical baseline algorithms"""
    from network_generation import QuantumRepeaterNetwork, CandidatePathGenerator
    from noise_models import NoiseParameters

    # Setup test problem
    print("Setting up test problem...")
    network = QuantumRepeaterNetwork(seed=42)
    network.generate_barbell_network(cluster_size=3, bridge_width=2)

    demands = network.generate_communication_demands(num_demands=4)

    path_gen = CandidatePathGenerator(network)
    candidate_paths = path_gen.compute_candidate_paths(demands, k=3)

    noise_params = NoiseParameters()
    fidelity_sim = EntanglementQualitySimulator(network, noise_params)

    # Run baselines
    print("\n" + "=" * 70)
    print("CLASSICAL BASELINE COMPARISON")
    print("=" * 70)

    # Sequential Greedy
    greedy = SequentialGreedyRouter(network, demands, candidate_paths, fidelity_sim)
    greedy_solution = greedy.solve()

    # Independent Shortest Path
    independent = IndependentShortestPathRouter(
        network, demands, candidate_paths, fidelity_sim
    )
    independent_solution = independent.solve()

    # Random (average over 10 runs)
    print("\n" + "=" * 70)
    print("RANDOM PATH SELECTION (LOWER BOUND)")
    print("=" * 70)

    random_objectives = []
    random_utilities = []

    for run in range(10):
        random_router = RandomPathRouter(
            network, demands, candidate_paths, fidelity_sim, seed=run
        )
        random_solution = random_router.solve()
        random_objectives.append(random_solution.objective_value)
        random_utilities.append(random_solution.total_utility)

    print(f"Random (10 runs):")
    print(
        f"  Avg objective: {np.mean(random_objectives):.4f} ± {np.std(random_objectives):.4f}"
    )
    print(
        f"  Avg utility: {np.mean(random_utilities):.4f} ± {np.std(random_utilities):.4f}"
    )

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<30} {'Objective':>12} {'Utility':>12} {'Valid':>8}")
    print("-" * 70)
    print(
        f"{'Independent (upper bound)':<30} "
        f"{independent_solution.objective_value:>12.4f} "
        f"{independent_solution.total_utility:>12.4f} "
        f"{str(independent_solution.is_valid):>8}"
    )
    print(
        f"{'Sequential Greedy':<30} "
        f"{greedy_solution.objective_value:>12.4f} "
        f"{greedy_solution.total_utility:>12.4f} "
        f"{str(greedy_solution.is_valid):>8}"
    )
    print(
        f"{'Random (mean)':<30} "
        f"{np.mean(random_objectives):>12.4f} "
        f"{np.mean(random_utilities):>12.4f} "
        f"{'N/A':>8}"
    )

    # Approximation ratio
    if independent_solution.total_utility > 0:
        greedy_ratio = (
            greedy_solution.total_utility / independent_solution.total_utility
        )
        print(f"\nGreedy approximation ratio: {greedy_ratio:.4f}")


if __name__ == "__main__":
    main()

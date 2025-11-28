"""
Noise Models and Entanglement Fidelity Calculations

This module implements realistic quantum noise channels and computes end-to-end
entanglement fidelity for multi-hop paths in quantum repeater networks.

Scientific Background:
- Amplitude damping: spontaneous emission with rate 1/T1
- Phase damping: environmental dephasing with rate 1/T2
- Photon loss: distance-dependent fiber attenuation
- BSM errors: imperfect Bell state measurements during swapping

Key formulas:
- Fidelity under amplitude damping: F' = F + (1-F)(1-γ)/2 where γ = 1-exp(-t/T1)
- Fidelity after entanglement swapping: Werner state formalism
- End-to-end fidelity: composition of link fidelities through swapping chain
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from network_generation import (
    QuantumRepeaterNetwork,
    CandidatePath,
    NodeProperties,
    LinkProperties,
    CommunicationDemand,
)


@dataclass
class NoiseParameters:
    """Global noise parameters for the quantum hardware"""

    gate_error_1q: float = 0.001  # Single-qubit gate error rate
    gate_error_2q: float = 0.01  # Two-qubit gate error rate
    readout_error: float = 0.02  # Measurement error probability
    thermal_excitation: float = 0.01  # Thermal population in excited state


class EntanglementQualitySimulator:
    """
    Simulate entanglement distribution with realistic noise.

    This is the core of the noise-aware QAOA approach. Instead of assuming
    ideal quantum operations, we simulate the actual entanglement fidelity
    that would be achieved on physical hardware.
    """

    def __init__(
        self, network: QuantumRepeaterNetwork, noise_params: NoiseParameters = None
    ):
        self.network = network
        self.noise_params = noise_params or NoiseParameters()

    def compute_path_fidelity(
        self, path: CandidatePath, storage_time_us: float = 10.0
    ) -> float:
        """
        Compute end-to-end entanglement fidelity for a path.

        This is the key metric for entanglement routing quality. The calculation
        accounts for:
        1. Initial Bell pair fidelity on each link (from source imperfections)
        2. Photon loss during transmission (distance-dependent)
        3. Memory decoherence while waiting for swapping (T1, T2)
        4. Imperfect Bell state measurements during swapping

        Args:
            path: Candidate path through network
            storage_time_us: Time qubits spend in memory before swapping

        Returns:
            End-to-end entanglement fidelity (0 to 1)
        """
        if len(path.edges) == 0:
            return 0.0

        # Single-link path: no swapping needed
        if len(path.edges) == 1:
            return self._compute_link_fidelity(path.edges[0], storage_time_us)

        # Multi-link path: apply entanglement swapping
        # Start with first link fidelity
        fidelity = self._compute_link_fidelity(path.edges[0], storage_time_us)

        # Iteratively swap with each subsequent link
        for i in range(1, len(path.edges)):
            # Get fidelity of next link
            link_fidelity = self._compute_link_fidelity(path.edges[i], storage_time_us)

            # Apply entanglement swapping with BSM errors
            intermediate_node = path.nodes[i]
            bsm_success, bsm_error = self._get_bsm_properties(path.edges[i - 1])

            # Swapping fidelity formula for Werner states
            # F_out = F1 * F2 * (1 - bsm_error) + (1-F1)*(1-F2)*bsm_error/3
            # This accounts for the fact that swapping two Werner states
            # produces a Werner state with fidelity that depends on inputs
            fidelity = self._swap_fidelity(
                fidelity, link_fidelity, bsm_success, bsm_error
            )

        return max(0.0, min(1.0, fidelity))

    def _compute_link_fidelity(
        self, edge: Tuple[int, int], storage_time_us: float
    ) -> float:
        """
        Compute fidelity of entanglement on a single link.

        Combines:
        - Initial Bell pair fidelity from source
        - Photon loss during fiber transmission
        - Amplitude damping during storage (T1)
        - Phase damping during storage (T2)

        Args:
            edge: Link (source, target) tuple
            storage_time_us: Storage time in microseconds

        Returns:
            Link entanglement fidelity
        """
        link_props = self.network.link_props[edge]

        # Start with initial Bell pair fidelity
        fidelity = link_props.initial_fidelity

        # Apply photon loss (reduces fidelity)
        # Transmission probability: P = exp(-α * L)
        # Loss reduces fidelity toward completely mixed state
        alpha_natural = (
            link_props.alpha_db_per_km * 0.23026
        )  # Convert dB to natural units
        transmission = np.exp(-alpha_natural * link_props.distance)
        fidelity = fidelity * transmission + (1 - transmission) * 0.25

        # Get memory properties of endpoint nodes
        source_node = link_props.source
        target_node = link_props.target

        # Average decoherence parameters (both endpoints store qubits)
        avg_t1 = (
            self.network.node_props[source_node].t1_relaxation
            + self.network.node_props[target_node].t1_relaxation
        ) / 2
        avg_t2 = (
            self.network.node_props[source_node].t2_coherence
            + self.network.node_props[target_node].t2_coherence
        ) / 2

        # Apply amplitude damping (T1 process)
        fidelity = self._apply_amplitude_damping(fidelity, storage_time_us, avg_t1)

        # Apply phase damping (T2 process)
        fidelity = self._apply_phase_damping(fidelity, storage_time_us, avg_t2)

        return fidelity

    def _apply_amplitude_damping(
        self, fidelity: float, time_us: float, t1_us: float
    ) -> float:
        """
        Apply amplitude damping noise to Bell state fidelity.

        Amplitude damping models spontaneous emission from excited to ground state.
        For a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2, amplitude damping on one qubit
        transforms it toward |00⟩, reducing fidelity.

        Damping parameter: γ = 1 - exp(-t/T1)
        Fidelity evolution: F' = F + (1-F)·(1-γ)/2

        Args:
            fidelity: Current Bell state fidelity
            time_us: Evolution time in microseconds
            t1_us: T1 relaxation time in microseconds

        Returns:
            Fidelity after amplitude damping
        """
        gamma = 1.0 - np.exp(-time_us / t1_us)
        # Bell state fidelity decreases toward 1/2 (partially mixed)
        new_fidelity = fidelity + (1 - fidelity) * (1 - gamma) / 2
        return new_fidelity

    def _apply_phase_damping(
        self, fidelity: float, time_us: float, t2_us: float
    ) -> float:
        """
        Apply phase damping (dephasing) noise to Bell state fidelity.

        Phase damping models environmental phase noise that destroys coherence
        without energy relaxation. For Bell states, this reduces off-diagonal
        density matrix elements.

        Dephasing parameter: λ = 1 - exp(-t/T2)
        Fidelity evolution: F' = F - (F - 1/2)·λ

        Args:
            fidelity: Current Bell state fidelity
            time_us: Evolution time in microseconds
            t2_us: T2 coherence time in microseconds

        Returns:
            Fidelity after phase damping
        """
        lambda_param = 1.0 - np.exp(-time_us / t2_us)
        # Dephasing drives toward maximally mixed state (F = 1/2)
        new_fidelity = fidelity - (fidelity - 0.5) * lambda_param
        return new_fidelity

    def _get_bsm_properties(self, edge: Tuple[int, int]) -> Tuple[float, float]:
        """Get Bell state measurement properties for a link"""
        link_props = self.network.link_props[edge]
        return link_props.bsm_success_prob, link_props.bsm_error_rate

    def _swap_fidelity(
        self, f1: float, f2: float, bsm_success: float, bsm_error: float
    ) -> float:
        """
        Compute output fidelity from entanglement swapping.

        Entanglement swapping combines two Bell pairs A-B and B-C into A-C
        through a Bell state measurement on the B qubits. If both input pairs
        are Werner states with fidelities F1 and F2, the output is a Werner
        state with fidelity:

        F_out = F1 · F2 · (1 - ε) + (1-F1)·(1-F2)·ε/3

        where ε is the BSM error rate. The first term is successful swapping,
        the second is accidental success from noise.

        BSM success probability affects whether swapping succeeds at all,
        but given success, the fidelity follows the above formula.

        Args:
            f1: Fidelity of first Bell pair
            f2: Fidelity of second Bell pair
            bsm_success: BSM success probability
            bsm_error: BSM depolarizing error rate

        Returns:
            Fidelity of swapped Bell pair (given successful BSM)
        """
        # Account for BSM success probability
        # If BSM fails, we get no entanglement (F=0)
        # This is handled at higher level; here we compute conditional fidelity

        # Werner state swapping formula
        f_out = f1 * f2 * (1 - bsm_error) + (1 - f1) * (1 - f2) * bsm_error / 3

        # BSM success probability scales effective fidelity
        # If success prob < 1, we sometimes fail and get F=0
        # Expected fidelity: E[F] = p_success * F_out + (1-p_success) * 0
        effective_fidelity = bsm_success * f_out

        return effective_fidelity

    def compute_path_latency(self, path: CandidatePath) -> float:
        """
        Compute end-to-end latency for entanglement distribution.

        Latency includes:
        1. Entanglement generation time on each link (probabilistic)
        2. Classical communication time for BSM results
        3. Sequential swapping operations (cannot parallelize)

        Args:
            path: Candidate path

        Returns:
            Expected latency in milliseconds
        """
        latency_ms = 0.0

        # Entanglement generation on all links (can be parallel)
        # Time = 1 / (generation_rate * success_probability)
        max_gen_time = 0.0
        for edge in path.edges:
            link_props = self.network.link_props[edge]

            # Success probability from photon loss
            alpha_natural = link_props.alpha_db_per_km * 0.23026
            transmission = np.exp(-alpha_natural * link_props.distance)

            # Expected attempts until success: 1 / p_success
            expected_attempts = 1.0 / transmission

            # Time per attempt: 1 / generation_rate
            time_per_attempt_us = 1e6 / link_props.generation_rate

            # Total generation time
            gen_time_us = expected_attempts * time_per_attempt_us
            max_gen_time = max(max_gen_time, gen_time_us)

        latency_ms += max_gen_time / 1000.0  # Convert to milliseconds

        # Classical communication time for each swap
        # Light speed in fiber: ~200,000 km/s
        c_fiber = 200000.0  # km/s

        for i in range(path.num_swaps()):
            # Communication between adjacent nodes
            edge = path.edges[i]
            link_props = self.network.link_props[edge]

            # Round-trip time for BSM result
            comm_time_ms = 2 * link_props.distance / c_fiber * 1000
            latency_ms += comm_time_ms

        # Swapping operation time (local processing)
        # Assume ~1 microsecond per swap (fast compared to communication)
        swapping_time_ms = path.num_swaps() * 0.001
        latency_ms += swapping_time_ms

        return latency_ms

    def compute_path_utility(
        self,
        path: CandidatePath,
        storage_time_us: float = 10.0,
        fidelity_weight: float = 0.7,
        latency_weight: float = 0.3,
    ) -> float:
        """
        Compute utility of a path combining fidelity and latency.

        Utility is a weighted combination of normalized fidelity and latency.
        Higher utility is better. This is the objective function that QAOA
        maximizes.

        Args:
            path: Candidate path
            storage_time_us: Memory storage time
            fidelity_weight: Weight for fidelity term (0 to 1)
            latency_weight: Weight for latency term (0 to 1)

        Returns:
            Path utility (higher is better)
        """
        fidelity = self.compute_path_fidelity(path, storage_time_us)
        latency_ms = self.compute_path_latency(path)

        # Normalize latency to [0, 1] range
        # Assume maximum acceptable latency is 100 ms
        max_latency = 100.0
        normalized_latency = max(0, 1 - latency_ms / max_latency)

        # Weighted combination
        utility = fidelity_weight * fidelity + latency_weight * normalized_latency

        return utility


class ResourceConflictDetector:
    """
    Detect resource conflicts when multiple paths are selected.

    Conflicts occur when total resource usage exceeds node capacity:
    - Memory qubit usage exceeds available qubits at a node
    - Link bandwidth exceeds generation capacity
    """

    def __init__(self, network: QuantumRepeaterNetwork):
        self.network = network

    def check_node_capacity_violations(
        self, selected_paths: List[CandidatePath]
    ) -> Dict[int, int]:
        """
        Check which nodes have capacity violations.

        Args:
            selected_paths: List of paths in the routing configuration

        Returns:
            Dictionary mapping node_id to excess qubit usage (violation amount)
        """
        # Compute total qubit usage at each node
        node_usage = {}

        for path in selected_paths:
            for node in path.nodes:
                # Each path uses qubits at nodes it traverses
                # Endpoints: 1 qubit, intermediate: 2 qubits
                if node == path.nodes[0] or node == path.nodes[-1]:
                    qubits_needed = 1
                else:
                    qubits_needed = 2

                node_usage[node] = node_usage.get(node, 0) + qubits_needed

        # Check violations
        violations = {}
        for node, usage in node_usage.items():
            capacity = self.network.node_props[node].num_memory_qubits
            if usage > capacity:
                violations[node] = usage - capacity

        return violations

    def check_link_capacity_violations(
        self, selected_paths: List[CandidatePath]
    ) -> Dict[Tuple[int, int], int]:
        """
        Check which links have capacity violations.

        Multiple paths using the same link must share entanglement generation
        bandwidth. We assume each link can support a limited number of
        simultaneous users.

        Args:
            selected_paths: List of paths in routing configuration

        Returns:
            Dictionary mapping edge to excess usage count
        """
        # Count link usage
        link_usage = {}

        for path in selected_paths:
            for edge in path.edges:
                # Normalize edge to canonical form
                canonical_edge = tuple(sorted(edge))
                link_usage[canonical_edge] = link_usage.get(canonical_edge, 0) + 1

        # Assume each link supports max 3 simultaneous users
        # (This is a simplification; real systems have bandwidth limits)
        max_users_per_link = 3

        violations = {}
        for edge, usage in link_usage.items():
            if usage > max_users_per_link:
                violations[edge] = usage - max_users_per_link

        return violations

    def compute_total_violation_penalty(
        self, selected_paths: List[CandidatePath], penalty_weight: float = 10.0
    ) -> float:
        """
        Compute total penalty for resource violations.

        This penalty is added to the cost function to enforce constraints.

        Args:
            selected_paths: Routing configuration
            penalty_weight: Penalty per unit violation

        Returns:
            Total penalty value (higher is worse)
        """
        node_violations = self.check_node_capacity_violations(selected_paths)
        link_violations = self.check_link_capacity_violations(selected_paths)

        # Sum all violations
        total_node_excess = sum(node_violations.values())
        total_link_excess = sum(link_violations.values())

        penalty = penalty_weight * (total_node_excess + total_link_excess)

        return penalty


def main():
    """Demonstration of fidelity calculations and noise models"""
    from network_generation import QuantumRepeaterNetwork, CandidatePathGenerator

    # Create test network
    network = QuantumRepeaterNetwork(seed=42)
    G = network.generate_barbell_network(cluster_size=3, bridge_width=2)

    # Generate demands and paths
    demands = network.generate_communication_demands(num_demands=3)
    path_gen = CandidatePathGenerator(network)
    candidate_paths = path_gen.compute_candidate_paths(demands, k=3)

    # Initialize fidelity simulator
    noise_params = NoiseParameters(
        gate_error_1q=0.001, gate_error_2q=0.01, readout_error=0.02
    )
    simulator = EntanglementQualitySimulator(network, noise_params)

    print("Entanglement Fidelity Analysis\n")
    print("=" * 60)

    # Analyze each path
    for demand_id, paths in candidate_paths.items():
        print(
            f"\nDemand {demand_id} ({demands[demand_id].source} → {demands[demand_id].destination}):"
        )

        for path in paths:
            fidelity = simulator.compute_path_fidelity(path, storage_time_us=10.0)
            latency = simulator.compute_path_latency(path)
            utility = simulator.compute_path_utility(path)

            print(f"  Path {path.path_id}: {' → '.join(map(str, path.nodes))}")
            print(f"    Links: {path.num_links()}, Swaps: {path.num_swaps()}")
            print(f"    Fidelity: {fidelity:.4f}")
            print(f"    Latency: {latency:.2f} ms")
            print(f"    Utility: {utility:.4f}")

    # Test resource conflicts
    print("\n" + "=" * 60)
    print("Resource Conflict Analysis\n")

    # Select first path for each demand
    selected = [paths[0] for paths in candidate_paths.values()]

    conflict_detector = ResourceConflictDetector(network)
    node_violations = conflict_detector.check_node_capacity_violations(selected)
    link_violations = conflict_detector.check_link_capacity_violations(selected)

    print(f"Selected paths: {len(selected)}")
    print(f"Node capacity violations: {len(node_violations)}")
    if node_violations:
        for node, excess in node_violations.items():
            print(f"  Node {node}: {excess} qubits over capacity")

    print(f"Link capacity violations: {len(link_violations)}")
    if link_violations:
        for edge, excess in link_violations.items():
            print(f"  Link {edge}: {excess} users over capacity")

    total_penalty = conflict_detector.compute_total_violation_penalty(selected)
    print(f"\nTotal violation penalty: {total_penalty:.2f}")


if __name__ == "__main__":
    main()

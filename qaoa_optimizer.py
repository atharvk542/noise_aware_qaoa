"""
Noise-Aware QAOA for Multi-Flow Entanglement Routing

This module implements the core QAOA algorithm with noise-aware cost function
evaluation. The key innovation is that we simulate realistic entanglement
distribution with hardware noise models rather than assuming ideal operations.

Scientific Contribution:
Classical greedy heuristics route demands sequentially, missing global optima.
QAOA can find configurations where slightly suboptimal individual paths yield
better aggregate performance. By incorporating noise directly into the cost
function, we optimize for real-world performance rather than idealized metrics.

QAOA Structure:
- Problem Hamiltonian: Encodes routing costs and capacity constraints
- Mixer Hamiltonian: Enables transitions between valid configurations
- Variational parameters: Beta (problem) and gamma (mixer) per layer
- Adaptive depth: Balance solution quality vs gate errors
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
)

from network_generation import (
    QuantumRepeaterNetwork,
    CandidatePath,
    CommunicationDemand,
    CandidatePathGenerator,
)
from noise_models import (
    EntanglementQualitySimulator,
    ResourceConflictDetector,
    NoiseParameters,
)


@dataclass
class QAOAParameters:
    """QAOA algorithm hyperparameters"""

    depth: int = 2  # Number of QAOA layers (p)
    num_shots: int = 1024  # Measurements per circuit evaluation
    max_iterations: int = 30  # SPSA optimizer iterations (reduced for speed)
    spsa_a: float = 0.15  # SPSA step size parameter (increased for faster convergence)
    spsa_c: float = 0.015  # SPSA gradient estimation parameter
    penalty_weight: float = 10.0  # Constraint violation penalty
    regularization_weight: float = 0.1  # Robustness regularization
    adaptive_depth: bool = True  # Enable adaptive depth selection
    max_depth: int = 4  # Maximum depth for adaptive mode (reduced)


@dataclass
class RoutingConfiguration:
    """A complete routing solution"""

    path_selections: Dict[int, int]  # demand_id -> path_id
    total_utility: float
    total_penalty: float
    objective_value: float  # utility - penalty
    is_valid: bool  # No constraint violations


class QAOARoutingOptimizer:
    """
    QAOA optimizer for multi-flow entanglement routing.

    Encodes the routing problem as a QAOA instance where qubits represent
    path selection decisions. The cost function evaluates realistic
    entanglement quality using noise simulations.
    """

    def __init__(
        self,
        network: QuantumRepeaterNetwork,
        demands: List[CommunicationDemand],
        candidate_paths: Dict[int, List[CandidatePath]],
        noise_params: NoiseParameters,
        qaoa_params: QAOAParameters,
    ):
        self.network = network
        self.demands = demands
        self.candidate_paths = candidate_paths
        self.noise_params = noise_params
        self.qaoa_params = qaoa_params

        # Initialize simulators
        self.fidelity_sim = EntanglementQualitySimulator(network, noise_params)
        self.conflict_detector = ResourceConflictDetector(network)

        # Problem encoding
        self.num_demands = len(demands)
        self.num_candidates_per_demand = {
            d.demand_id: len(candidate_paths[d.demand_id]) for d in demands
        }
        self.total_qubits = sum(self.num_candidates_per_demand.values())

        # Qubit mapping: (demand_id, path_id) -> qubit_index
        self.qubit_map = self._build_qubit_mapping()
        self.inverse_qubit_map = {v: k for k, v in self.qubit_map.items()}

        # Build noise model for quantum circuits
        self.circuit_noise_model = self._build_circuit_noise_model()

        # Initialize backend with GPU acceleration if available, else fallback to CPU
        try:
            self.backend = AerSimulator(
                noise_model=self.circuit_noise_model, method="statevector", device="GPU"
            )
            # Test if GPU is actually working by running a dummy circuit
            # Some versions don't throw error until execution
            dummy_qc = QuantumCircuit(1)
            dummy_qc.h(0)
            self.backend.run(transpile(dummy_qc, self.backend)).result()
            print("Initialized AerSimulator with GPU acceleration")
        except Exception as e:
            print(f"Warning: GPU acceleration not available ({str(e)}). Falling back to CPU.")
            self.backend = AerSimulator(
                noise_model=self.circuit_noise_model, method="statevector", device="CPU"
            )

        # Optimization tracking
        self.cost_history = []
        self.best_solution = None
        self.best_cost = float("-inf")

    def _build_qubit_mapping(self) -> Dict[Tuple[int, int], int]:
        """
        Create mapping from (demand_id, path_id) to qubit index.

        We use a one-hot encoding per demand: K qubits for K candidate paths.
        Total qubits = sum over all demands of their candidate counts.
        """
        qubit_map = {}
        qubit_idx = 0

        for demand in self.demands:
            demand_id = demand.demand_id
            num_paths = self.num_candidates_per_demand[demand_id]

            for path_id in range(num_paths):
                qubit_map[(demand_id, path_id)] = qubit_idx
                qubit_idx += 1

        return qubit_map

    def _build_circuit_noise_model(self) -> NoiseModel:
        """
        Build Qiskit noise model for QAOA circuit simulation.

        Includes:
        - Depolarizing errors on single and two-qubit gates
        - Amplitude damping on idle qubits
        - Phase damping on idle qubits
        - Readout errors
        """
        noise_model = NoiseModel()

        # Single-qubit gate errors
        error_1q = depolarizing_error(self.noise_params.gate_error_1q, 1)
        noise_model.add_all_qubit_quantum_error(
            error_1q, ["u1", "u2", "u3", "rx", "ry", "rz"]
        )

        # Two-qubit gate errors
        error_2q = depolarizing_error(self.noise_params.gate_error_2q, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz"])

        # Readout errors (symmetric)
        from qiskit_aer.noise import ReadoutError

        p_error = self.noise_params.readout_error
        readout_error = ReadoutError([[1 - p_error, p_error], [p_error, 1 - p_error]])
        noise_model.add_all_qubit_readout_error(readout_error)

        return noise_model

    def build_qaoa_circuit(self, depth: int) -> QuantumCircuit:
        """
        Construct QAOA circuit with parameterized cost and mixer layers.

        Circuit structure:
        1. Initial state: Equal superposition over all basis states
        2. For each layer p:
            a. Cost Hamiltonian: Phase rotations encoding objective
            b. Mixer Hamiltonian: X rotations for exploration
        3. Measurement in computational basis

        Args:
            depth: Number of QAOA layers (p parameter)

        Returns:
            Parameterized quantum circuit
        """
        qr = QuantumRegister(self.total_qubits, "q")
        qc = QuantumCircuit(qr)

        # Initial state: Hadamard on all qubits (equal superposition)
        qc.h(qr)

        # Create parameters for each layer
        beta_params = [Parameter(f"β_{i}") for i in range(depth)]
        gamma_params = [Parameter(f"γ_{i}") for i in range(depth)]

        for layer in range(depth):
            # Cost Hamiltonian layer
            self._add_cost_hamiltonian(qc, qr, gamma_params[layer])

            # Mixer Hamiltonian layer
            self._add_mixer_hamiltonian(qc, qr, beta_params[layer])

        # Measurement
        qc.measure_all()

        return qc

    def _add_cost_hamiltonian(
        self, qc: QuantumCircuit, qr: QuantumRegister, gamma: Parameter
    ):
        """
        Add cost Hamiltonian layer to circuit.

        The cost Hamiltonian encodes:
        1. Reward for selecting high-utility paths (diagonal terms)
        2. Penalties for constraint violations (interaction terms)
        3. One-hot constraint enforcement per demand

        We use phase rotations Z(γ·cost) where cost depends on which
        paths are selected (encoded in computational basis states).

        Since we can't evaluate the full classical cost function in the
        quantum circuit, we use a simplified Hamiltonian that captures
        the structure:
        - Diagonal: Individual path utilities (pre-computed)
        - Pairwise: Penalties for paths that conflict on resources
        """
        # For each demand, add penalty terms for selecting multiple paths
        # (enforce one-hot constraint)
        for demand in self.demands:
            demand_id = demand.demand_id
            num_paths = self.num_candidates_per_demand[demand_id]

            # Penalty for selecting 0 or >1 paths
            # Add ZZ interactions between all pairs of qubits for this demand
            qubit_indices = [
                self.qubit_map[(demand_id, path_id)] for path_id in range(num_paths)
            ]

            for i in range(len(qubit_indices)):
                for j in range(i + 1, len(qubit_indices)):
                    # ZZ interaction: penalty when both qubits are |1⟩
                    qi, qj = qubit_indices[i], qubit_indices[j]
                    qc.cx(qr[qi], qr[qj])
                    qc.rz(2 * gamma * self.qaoa_params.penalty_weight, qr[qj])
                    qc.cx(qr[qi], qr[qj])

        # Add resource conflict penalties
        # For paths that share nodes/links, add ZZ interactions
        for d1_idx, demand1 in enumerate(self.demands):
            for d2_idx in range(d1_idx + 1, len(self.demands)):
                demand2 = self.demands[d2_idx]

                # Check all path pairs between these demands
                for path1 in self.candidate_paths[demand1.demand_id]:
                    for path2 in self.candidate_paths[demand2.demand_id]:
                        # Compute resource overlap
                        overlap = self._compute_path_overlap(path1, path2)

                        if overlap > 0:
                            # Add penalty proportional to overlap
                            q1 = self.qubit_map[(demand1.demand_id, path1.path_id)]
                            q2 = self.qubit_map[(demand2.demand_id, path2.path_id)]

                            penalty = overlap * self.qaoa_params.penalty_weight

                            qc.cx(qr[q1], qr[q2])
                            qc.rz(2 * gamma * penalty, qr[q2])
                            qc.cx(qr[q1], qr[q2])

        # Add single-qubit phase rotations for path utilities
        # (This encodes the reward for selecting each path)
        for demand in self.demands:
            for path in self.candidate_paths[demand.demand_id]:
                # Pre-compute path utility
                utility = self.fidelity_sim.compute_path_utility(path)

                # Z rotation: phase proportional to utility
                # Negative sign because we want to maximize utility
                qubit_idx = self.qubit_map[(demand.demand_id, path.path_id)]
                qc.rz(-2 * gamma * utility, qr[qubit_idx])

    def _add_mixer_hamiltonian(
        self, qc: QuantumCircuit, qr: QuantumRegister, beta: Parameter
    ):
        """
        Add mixer Hamiltonian layer to circuit.

        The mixer enables exploration of the solution space through
        X rotations. We use a simple transverse field mixer:
        H_mixer = sum_i X_i

        This allows transitions between different path selections.
        For better constraint preservation, we could use an XY mixer
        that only flips within each demand's qubit group, but the
        simple X mixer is sufficient with penalty terms.
        """
        # Apply X rotation to all qubits
        for i in range(self.total_qubits):
            qc.rx(2 * beta, qr[i])

    def _compute_path_overlap(
        self, path1: CandidatePath, path2: CandidatePath
    ) -> float:
        """
        Compute resource overlap between two paths.

        Returns a penalty value based on:
        - Number of shared nodes
        - Number of shared links
        - Memory capacity consumed at shared nodes

        Args:
            path1, path2: Candidate paths to compare

        Returns:
            Overlap penalty value
        """
        # Shared nodes
        nodes1 = set(path1.nodes)
        nodes2 = set(path2.nodes)
        shared_nodes = nodes1.intersection(nodes2)

        # Shared links (normalized)
        edges1 = set(tuple(sorted(e)) for e in path1.edges)
        edges2 = set(tuple(sorted(e)) for e in path2.edges)
        shared_edges = edges1.intersection(edges2)

        # Simple overlap metric: count shared resources
        overlap = len(shared_nodes) + 2 * len(shared_edges)

        return overlap

    def decode_bitstring(self, bitstring: str) -> RoutingConfiguration:
        """
        Decode measurement bitstring into routing configuration.

        Each bitstring represents a path selection for each demand.
        We enforce the one-hot constraint by selecting the path with
        lowest index that has qubit=1, or if none, select path 0.

        Args:
            bitstring: Measurement outcome (e.g., "0110101...")

        Returns:
            Routing configuration with utility and penalty
        """
        # Bitstring is in reverse order (qubit 0 is rightmost)
        bits = bitstring[::-1]

        path_selections = {}
        selected_paths = []

        for demand in self.demands:
            demand_id = demand.demand_id
            num_paths = self.num_candidates_per_demand[demand_id]

            # Extract bits for this demand
            demand_bits = []
            for path_id in range(num_paths):
                qubit_idx = self.qubit_map[(demand_id, path_id)]
                demand_bits.append(int(bits[qubit_idx]))

            # Select path: prefer first with bit=1, else select path 0
            selected_path_id = 0
            for path_id, bit in enumerate(demand_bits):
                if bit == 1:
                    selected_path_id = path_id
                    break

            path_selections[demand_id] = selected_path_id
            selected_paths.append(self.candidate_paths[demand_id][selected_path_id])

        # Evaluate this configuration
        total_utility = sum(
            self.fidelity_sim.compute_path_utility(path) for path in selected_paths
        )

        total_penalty = self.conflict_detector.compute_total_violation_penalty(
            selected_paths, penalty_weight=self.qaoa_params.penalty_weight
        )

        objective = total_utility - total_penalty
        is_valid = total_penalty == 0

        return RoutingConfiguration(
            path_selections=path_selections,
            total_utility=total_utility,
            total_penalty=total_penalty,
            objective_value=objective,
            is_valid=is_valid,
        )

    def evaluate_cost_function(self, params: np.ndarray, depth: int) -> float:
        """
        Evaluate QAOA cost function for given parameters.

        This is the core of noise-aware optimization:
        1. Build QAOA circuit with parameters
        2. Simulate with noise model
        3. Measure to get bitstrings
        4. Decode each bitstring to routing configuration
        5. Simulate realistic entanglement distribution with noise
        6. Compute average objective value (utility - penalties)

        Args:
            params: Array of [beta_0, gamma_0, beta_1, gamma_1, ...]
            depth: QAOA depth (number of layers)

        Returns:
            Expected objective value (higher is better)
        """
        # Build circuit
        qc = self.build_qaoa_circuit(depth)

        # Bind parameters
        param_dict = {}
        for i in range(depth):
            param_dict[f"β_{i}"] = params[2 * i]
            param_dict[f"γ_{i}"] = params[2 * i + 1]

        bound_circuit = qc.assign_parameters(param_dict)

        # Transpile and execute circuit
        transpiled_circuit = transpile(bound_circuit, self.backend)
        job = self.backend.run(transpiled_circuit, shots=self.qaoa_params.num_shots)
        result = job.result()

        # Get measurement counts
        counts = result.get_counts()

        # Evaluate cost for each measured bitstring
        total_cost = 0.0
        total_counts = 0

        for bitstring, count in counts.items():
            # Decode to routing configuration
            config = self.decode_bitstring(bitstring)

            # Weight by measurement count
            total_cost += config.objective_value * count
            total_counts += count

        # Average cost
        avg_cost = total_cost / max(1, total_counts)

        return avg_cost

    def optimize_spsa(self, depth: int) -> Tuple[np.ndarray, float]:
        """
        Optimize QAOA parameters using SPSA algorithm.

        Simultaneous Perturbation Stochastic Approximation is a gradient-free
        optimizer well-suited for noisy cost functions. It estimates gradients
        by perturbing all parameters simultaneously.

        SPSA update rule:
        θ_{k+1} = θ_k - a_k * ĝ_k

        where ĝ_k is estimated by:
        ĝ_k = [f(θ_k + c_k*Δ_k) - f(θ_k - c_k*Δ_k)] / (2*c_k) * Δ_k

        Δ_k is random ±1 vector (Rademacher distribution)

        Args:
            depth: QAOA circuit depth

        Returns:
            Tuple of (optimal_parameters, best_cost)
        """
        num_params = 2 * depth  # beta and gamma for each layer

        # Initialize parameters randomly
        theta = np.random.uniform(-np.pi, np.pi, num_params)

        # SPSA parameters
        a = self.qaoa_params.spsa_a
        c = self.qaoa_params.spsa_c

        best_theta = theta.copy()
        best_cost = float("-inf")

        print(f"\nOptimizing QAOA with depth={depth}, {num_params} parameters")
        print(
            f"SPSA: {self.qaoa_params.max_iterations} iterations, {self.qaoa_params.num_shots} shots"
        )

        for iteration in range(self.qaoa_params.max_iterations):
            # SPSA step sizes (decay over iterations)
            a_k = a / (iteration + 1) ** 0.602
            c_k = c / (iteration + 1) ** 0.101

            # Random perturbation direction
            delta = 2 * np.random.randint(0, 2, num_params) - 1

            # Evaluate at perturbed points
            theta_plus = theta + c_k * delta
            theta_minus = theta - c_k * delta

            cost_plus = self.evaluate_cost_function(theta_plus, depth)
            cost_minus = self.evaluate_cost_function(theta_minus, depth)

            # Gradient estimate
            gradient = (cost_plus - cost_minus) / (2 * c_k) * delta

            # Update parameters (gradient ascent for maximization)
            theta = theta + a_k * gradient

            # Evaluate current point
            current_cost = self.evaluate_cost_function(theta, depth)

            # Track best
            if current_cost > best_cost:
                best_cost = current_cost
                best_theta = theta.copy()

            self.cost_history.append(current_cost)

            # Print progress
            if (iteration + 1) % 10 == 0:
                print(
                    f"  Iteration {iteration + 1}/{self.qaoa_params.max_iterations}: "
                    f"cost={current_cost:.4f}, best={best_cost:.4f}"
                )

        print(f"Optimization complete. Best cost: {best_cost:.4f}")

        return best_theta, best_cost

    def optimize(self) -> RoutingConfiguration:
        """
        Run full QAOA optimization with adaptive depth selection.

        Strategy:
        1. Start with shallow circuit (depth=1)
        2. Optimize parameters with SPSA
        3. If improvement plateaus and noise budget allows, increase depth
        4. Continue until max depth or convergence

        Returns:
            Best routing configuration found
        """
        start_time = time.time()

        # Determine initial and max depth based on noise regime
        depth = 1
        max_depth = self._determine_max_depth()

        print(f"\n{'=' * 70}")
        print(f"QAOA OPTIMIZATION FOR MULTI-FLOW ENTANGLEMENT ROUTING")
        print(f"{'=' * 70}")
        print(f"Problem size: {self.num_demands} demands, {self.total_qubits} qubits")
        print(f"Noise regime: gate_error_2q={self.noise_params.gate_error_2q:.4f}")
        print(f"Max depth: {max_depth}")

        best_overall_cost = float("-inf")
        best_overall_params = None
        best_depth = 1

        # Adaptive depth loop
        while depth <= max_depth:
            print(f"\n{'=' * 70}")
            print(f"DEPTH {depth}")
            print(f"{'=' * 70}")

            # Optimize at this depth
            params, cost = self.optimize_spsa(depth)

            if cost > best_overall_cost:
                best_overall_cost = cost
                best_overall_params = params
                best_depth = depth

            # Check if we should increase depth
            if not self.qaoa_params.adaptive_depth:
                break

            # Simple heuristic: increase depth if we're still improving
            improvement = cost - best_overall_cost
            if improvement < 0.01 * abs(best_overall_cost):
                print(f"\nConvergence plateau reached. Stopping at depth {depth}")
                break

            depth += 1

        # Extract best solution
        print(f"\n{'=' * 70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Best depth: {best_depth}")
        print(f"Best cost: {best_overall_cost:.4f}")

        # Run final evaluation to get best configuration
        qc = self.build_qaoa_circuit(best_depth)
        param_dict = {}
        for i in range(best_depth):
            param_dict[f"β_{i}"] = best_overall_params[2 * i]
            param_dict[f"γ_{i}"] = best_overall_params[2 * i + 1]

        bound_circuit = qc.assign_parameters(param_dict)
        transpiled_circuit = transpile(bound_circuit, self.backend)
        job = self.backend.run(transpiled_circuit, shots=self.qaoa_params.num_shots)
        result = job.result()
        counts = result.get_counts()

        # Find best configuration from measurements
        best_config = None
        best_objective = float("-inf")

        for bitstring, count in counts.items():
            config = self.decode_bitstring(bitstring)

            if config.objective_value > best_objective:
                best_objective = config.objective_value
                best_config = config

        elapsed_time = time.time() - start_time
        print(f"\nTotal optimization time: {elapsed_time:.2f} seconds")

        self.best_solution = best_config
        self.best_cost = best_overall_cost

        return best_config

    def _determine_max_depth(self) -> int:
        """
        Determine maximum QAOA depth based on noise regime.

        High noise (gate_error > 0.01): depth 1-2
        Moderate noise (0.001 < gate_error < 0.01): depth 3-5
        Low noise (gate_error < 0.001): depth up to 8
        """
        gate_error = self.noise_params.gate_error_2q

        if gate_error > 0.01:
            return min(2, self.qaoa_params.max_depth)
        elif gate_error > 0.001:
            return min(5, self.qaoa_params.max_depth)
        else:
            return self.qaoa_params.max_depth


def main():
    """Demonstration of QAOA optimization"""
    from network_generation import QuantumRepeaterNetwork, CandidatePathGenerator

    # Create small test network
    print("Setting up test problem...")
    network = QuantumRepeaterNetwork(seed=42)
    G = network.generate_barbell_network(cluster_size=3, bridge_width=1)

    # Generate demands with contention
    demands = network.generate_communication_demands(num_demands=3)

    # Compute candidate paths
    path_gen = CandidatePathGenerator(network)
    candidate_paths = path_gen.compute_candidate_paths(demands, k=2)

    # Setup QAOA
    noise_params = NoiseParameters(
        gate_error_1q=0.0005, gate_error_2q=0.005, readout_error=0.01
    )

    qaoa_params = QAOAParameters(
        depth=2,
        num_shots=512,  # Reduced for demo
        max_iterations=20,  # Reduced for demo
        spsa_a=0.1,
        spsa_c=0.01,
        adaptive_depth=False,
    )

    # Run optimization
    optimizer = QAOARoutingOptimizer(
        network, demands, candidate_paths, noise_params, qaoa_params
    )

    solution = optimizer.optimize()

    # Display results
    print(f"\n{'=' * 70}")
    print("SOLUTION")
    print(f"{'=' * 70}")
    print(f"Valid: {solution.is_valid}")
    print(f"Total utility: {solution.total_utility:.4f}")
    print(f"Total penalty: {solution.total_penalty:.4f}")
    print(f"Objective: {solution.objective_value:.4f}")
    print("\nPath selections:")
    for demand_id, path_id in solution.path_selections.items():
        path = candidate_paths[demand_id][path_id]
        print(f"  Demand {demand_id}: Path {path_id} - {path.nodes}")


if __name__ == "__main__":
    main()

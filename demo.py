"""
Quick Demo: Noise-Robust QAOA for Entanglement Routing

This script demonstrates the complete workflow on a small test problem.
Use this to verify your installation and understand the system components.

Expected runtime: 2-5 minutes
"""

import numpy as np
from network_generation import QuantumRepeaterNetwork, CandidatePathGenerator
from noise_models import EntanglementQualitySimulator, NoiseParameters
from qaoa_optimizer import QAOARoutingOptimizer, QAOAParameters
from classical_baselines import SequentialGreedyRouter, IndependentShortestPathRouter


def main():
    print("\n" + "=" * 70)
    print("QUICK DEMO: Noise-Robust QAOA for Entanglement Routing")
    print("=" * 70)

    # Step 1: Generate network
    print("\n[1/6] Generating quantum repeater network...")
    network = QuantumRepeaterNetwork(seed=42)
    G = network.generate_barbell_network(cluster_size=3, bridge_width=1)

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(
        f"  Average node capacity: {np.mean([p.num_memory_qubits for p in network.node_props.values()]):.1f} qubits"
    )

    # Step 2: Generate demands
    print("\n[2/6] Generating communication demands...")
    demands = network.generate_communication_demands(num_demands=3)

    print(f"  Demands: {len(demands)}")
    for d in demands:
        print(f"    Demand {d.demand_id}: {d.source} → {d.destination}")

    # Step 3: Compute candidate paths
    print("\n[3/6] Computing candidate paths...")
    path_gen = CandidatePathGenerator(network)
    candidate_paths = path_gen.compute_candidate_paths(demands, k=2)

    total_paths = sum(len(paths) for paths in candidate_paths.values())
    print(f"  Total candidate paths: {total_paths}")
    for demand_id, paths in candidate_paths.items():
        print(f"    Demand {demand_id}: {len(paths)} paths")

    # Step 4: Setup noise parameters
    print("\n[4/6] Configuring noise models...")
    noise_params = NoiseParameters(
        gate_error_1q=0.0005, gate_error_2q=0.005, readout_error=0.01
    )

    print(f"  Gate error (2Q): {noise_params.gate_error_2q}")
    print(
        f"  Average T1: {np.mean([p.t1_relaxation for p in network.node_props.values()]):.1f} μs"
    )
    print(
        f"  Average T2: {np.mean([p.t2_coherence for p in network.node_props.values()]):.1f} μs"
    )

    fidelity_sim = EntanglementQualitySimulator(network, noise_params)

    # Step 5: Run classical baselines
    print("\n[5/6] Running classical baselines...")

    # Sequential Greedy
    greedy = SequentialGreedyRouter(network, demands, candidate_paths, fidelity_sim)
    greedy_solution = greedy.solve()

    # Independent (upper bound)
    independent = IndependentShortestPathRouter(
        network, demands, candidate_paths, fidelity_sim
    )
    independent_solution = independent.solve()

    # Step 6: Run QAOA
    print("\n[6/6] Running QAOA optimization...")

    qaoa_params = QAOAParameters(
        depth=2,
        num_shots=512,  # Reduced for demo speed
        max_iterations=20,  # Reduced for demo speed
        spsa_a=0.1,
        spsa_c=0.01,
        adaptive_depth=False,  # Fixed depth for consistent demo
    )

    qaoa_optimizer = QAOARoutingOptimizer(
        network, demands, candidate_paths, noise_params, qaoa_params
    )

    qaoa_solution = qaoa_optimizer.optimize()

    # Results comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Algorithm':<30} {'Utility':>12} {'Valid':>8} {'Objective':>12}")
    print("-" * 70)

    print(
        f"{'Independent (upper bound)':<30} "
        f"{independent_solution.total_utility:>12.4f} "
        f"{str(independent_solution.is_valid):>8} "
        f"{independent_solution.objective_value:>12.4f}"
    )

    print(
        f"{'QAOA':<30} "
        f"{qaoa_solution.total_utility:>12.4f} "
        f"{str(qaoa_solution.is_valid):>8} "
        f"{qaoa_solution.objective_value:>12.4f}"
    )

    print(
        f"{'Sequential Greedy':<30} "
        f"{greedy_solution.total_utility:>12.4f} "
        f"{str(greedy_solution.is_valid):>8} "
        f"{greedy_solution.objective_value:>12.4f}"
    )

    # Compute metrics
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    if independent_solution.total_utility > 0:
        qaoa_ratio = qaoa_solution.total_utility / independent_solution.total_utility
        greedy_ratio = (
            greedy_solution.total_utility / independent_solution.total_utility
        )

        print(f"\nApproximation Ratios (higher is better):")
        print(f"  QAOA:   {qaoa_ratio:.4f} ({qaoa_ratio * 100:.2f}% of optimal)")
        print(f"  Greedy: {greedy_ratio:.4f} ({greedy_ratio * 100:.2f}% of optimal)")

    if greedy_solution.total_utility > 0:
        advantage = (
            qaoa_solution.total_utility - greedy_solution.total_utility
        ) / greedy_solution.total_utility
        print(f"\nQAOA Advantage over Greedy:")
        print(
            f"  {advantage * 100:+.2f}% {'(QAOA better)' if advantage > 0 else '(Greedy better)'}"
        )

    # Path selections
    print("\n" + "=" * 70)
    print("SELECTED PATHS")
    print("=" * 70)

    print("\nQAOA selections:")
    for demand_id, path_id in qaoa_solution.path_selections.items():
        path = candidate_paths[demand_id][path_id]
        utility = fidelity_sim.compute_path_utility(path)
        print(
            f"  Demand {demand_id}: Path {path_id} - {path.nodes} (utility={utility:.4f})"
        )

    print("\nGreedy selections:")
    for demand_id, path_id in greedy_solution.path_selections.items():
        path = candidate_paths[demand_id][path_id]
        utility = fidelity_sim.compute_path_utility(path)
        print(
            f"  Demand {demand_id}: Path {path_id} - {path.nodes} (utility={utility:.4f})"
        )

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run full experiments: python run_experiments.py --num-trials 10")
    print(
        "  2. Generate visualizations: python visualize_results.py results/experiment_results.csv"
    )
    print("  3. Read README.md for detailed usage and scientific background")
    print("\n")


if __name__ == "__main__":
    main()

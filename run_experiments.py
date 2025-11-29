"""
Experimental Framework and Statistical Analysis

This module implements rigorous experimental design and statistical validation
for comparing QAOA against classical baselines. Publication-quality results
require proper statistical testing, not just point comparisons.

Key components:
- Experimental parameter sweeps (network size, contention level, noise rates)
- Multiple independent trials with different random seeds
- Statistical significance testing (paired t-tests, Wilcoxon)
- Confidence intervals and effect sizes
- CSV output for reproducibility
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import time
import argparse
from pathlib import Path
from scipy import stats

from network_generation import (
    QuantumRepeaterNetwork,
    CandidatePathGenerator,
    CommunicationDemand,
)
from noise_models import EntanglementQualitySimulator, NoiseParameters
from qaoa_optimizer import QAOARoutingOptimizer, QAOAParameters, RoutingConfiguration
from classical_baselines import (
    SequentialGreedyRouter,
    IndependentShortestPathRouter,
    RandomPathRouter,
)
from adversarial_topologies import (
    generate_adversarial_network,
)


@dataclass
class ExperimentParameters:
    """Parameters defining an experimental configuration"""

    network_type: str  # 'barbell', 'grid', 'random'
    num_nodes: int
    num_demands: int
    num_candidate_paths: int
    contention_level: float  # Demands / available resources ratio
    gate_error_2q: float
    t1_avg: float  # Average T1 time (microseconds)
    t2_avg: float  # Average T2 time (microseconds)
    seed: int


@dataclass
class ExperimentResult:
    """Results from a single experimental trial"""

    # Configuration
    experiment_id: int
    network_type: str
    num_nodes: int
    num_demands: int
    contention_level: float
    gate_error_2q: float
    seed: int

    # QAOA results
    qaoa_objective: float
    qaoa_utility: float
    qaoa_penalty: float
    qaoa_valid: bool
    qaoa_time: float

    # Greedy results
    greedy_objective: float
    greedy_utility: float
    greedy_penalty: float
    greedy_valid: bool
    greedy_time: float

    # Upper bound
    independent_utility: float

    # Metrics
    qaoa_approximation_ratio: float  # QAOA utility / independent utility
    greedy_approximation_ratio: float
    qaoa_advantage: float  # (QAOA - Greedy) / Greedy
    greedy_gap: float  # Gap between greedy and optimal (verification metric)
    has_bottleneck: bool  # Topology has bottleneck structure
    instance_valid: bool  # Instance passes adversarial verification


class ExperimentRunner:
    """
    Run comprehensive experiments comparing QAOA vs classical baselines.

    Experimental design:
    1. Generate problem instances with controlled parameters
    2. Run each algorithm with multiple random seeds
    3. Collect performance metrics
    4. Perform statistical analysis
    5. Save results to CSV
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []

    def run_single_trial(self, params: ExperimentParameters) -> ExperimentResult:
        """
        Run single experimental trial comparing all algorithms.

        Uses adversarial topology generation to create problem instances
        designed to expose weaknesses in greedy sequential routing.

        Args:
            params: Experimental configuration

        Returns:
            Experimental result with performance metrics and verification
        """
        # Generate adversarial network topology
        # Map network types to adversarial topology generators
        topology_map = {
            "hourglass": "hourglass",
            "diamond": "diamond",
            "grid": "grid_cut_vertex",
        }

        topology_type = topology_map.get(params.network_type, "hourglass")

        G, node_props, link_props, demands = generate_adversarial_network(
            topology_type=topology_type,
            num_nodes=params.num_nodes,
            num_demands=params.num_demands,
            seed=params.seed,
        )

        # Create network object and populate with generated topology
        network = QuantumRepeaterNetwork(seed=params.seed)
        network.graph = G
        network.node_props = node_props
        network.link_props = link_props

        # Compute candidate paths
        path_gen = CandidatePathGenerator(network)
        candidate_paths = path_gen.compute_candidate_paths(
            demands, k=params.num_candidate_paths
        )

        # Setup noise parameters
        noise_params = NoiseParameters(
            gate_error_1q=params.gate_error_2q / 10,
            gate_error_2q=params.gate_error_2q,
            readout_error=0.02,
        )

        fidelity_sim = EntanglementQualitySimulator(network, noise_params)

        # Run Independent (upper bound)
        independent_router = IndependentShortestPathRouter(
            network, demands, candidate_paths, fidelity_sim
        )
        independent_solution = independent_router.solve()
        independent_utility = independent_solution.total_utility

        # Run Sequential Greedy
        greedy_start = time.time()
        greedy_router = SequentialGreedyRouter(
            network, demands, candidate_paths, fidelity_sim
        )
        greedy_solution = greedy_router.solve()
        greedy_time = time.time() - greedy_start

        # Run QAOA with BOOSTED parameters for better optimization
        qaoa_params = QAOAParameters(
            depth=2,  # Depth 2 for expressivity
            num_shots=2048,  # INCREASED shots for better statistics
            max_iterations=50,  # INCREASED iterations for convergence
            spsa_a=0.2,  # Larger step size for faster exploration
            spsa_c=0.02,
            penalty_weight=2.0,  # LOWERED penalty so utility differences matter more
            adaptive_depth=False,  # Fixed depth for consistency
            max_depth=3,  # Lower max depth to avoid noise accumulation
        )

        qaoa_start = time.time()
        qaoa_optimizer = QAOARoutingOptimizer(
            network, demands, candidate_paths, noise_params, qaoa_params
        )
        qaoa_solution = qaoa_optimizer.optimize()
        qaoa_time = time.time() - qaoa_start

        # Compute metrics
        qaoa_approx_ratio = (
            qaoa_solution.total_utility / independent_utility
            if independent_utility > 0
            else 0
        )
        greedy_approx_ratio = (
            greedy_solution.total_utility / independent_utility
            if independent_utility > 0
            else 0
        )

        qaoa_advantage = (
            (qaoa_solution.total_utility - greedy_solution.total_utility)
            / greedy_solution.total_utility
            if greedy_solution.total_utility > 0
            else 0
        )

        # Compute greedy gap (percentage below optimal)
        greedy_gap = (
            (independent_utility - greedy_solution.total_utility) / independent_utility
            if independent_utility > 0
            else 0
        )

        # Verify adversarial properties
        from adversarial_topologies import AdversarialTopologyGenerator

        verifier = AdversarialTopologyGenerator(seed=params.seed)
        verification = verifier.verify_adversarial_properties(
            G,
            node_props,
            link_props,
            demands,
            greedy_solution=greedy_solution,
            independent_solution=independent_solution,
        )

        return ExperimentResult(
            experiment_id=len(self.results),
            network_type=params.network_type,
            num_nodes=params.num_nodes,
            num_demands=params.num_demands,
            contention_level=params.contention_level,
            gate_error_2q=params.gate_error_2q,
            seed=params.seed,
            qaoa_objective=qaoa_solution.objective_value,
            qaoa_utility=qaoa_solution.total_utility,
            qaoa_penalty=qaoa_solution.total_penalty,
            qaoa_valid=qaoa_solution.is_valid,
            qaoa_time=qaoa_time,
            greedy_objective=greedy_solution.objective_value,
            greedy_utility=greedy_solution.total_utility,
            greedy_penalty=greedy_solution.total_penalty,
            greedy_valid=greedy_solution.is_valid,
            greedy_time=greedy_time,
            independent_utility=independent_utility,
            qaoa_approximation_ratio=qaoa_approx_ratio,
            greedy_approximation_ratio=greedy_approx_ratio,
            qaoa_advantage=qaoa_advantage,
            greedy_gap=greedy_gap,
            has_bottleneck=verification.has_bottleneck,
            instance_valid=verification.is_valid_instance,
        )

    def run_experiment_suite(
        self,
        network_types: List[str] = ["hourglass", "diamond", "grid"],
        num_nodes_range: List[int] = [8, 10],
        num_demands_range: List[int] = [3, 4],
        gate_error_range: List[float] = [0.001, 0.01],
        num_trials: int = 10,
    ):
        """
        Run comprehensive suite of experiments.

        Sweeps over network types, sizes, demand counts, and noise levels.
        Each configuration is repeated with multiple random seeds for
        statistical validity.

        Args:
            network_types: Network topology types to test
            num_nodes_range: Network sizes to test
            num_demands_range: Number of demands to test
            gate_error_range: Gate error rates to test
            num_trials: Number of independent trials per configuration
        """
        print("=" * 70)
        print("COMPREHENSIVE EXPERIMENTAL SUITE")
        print("=" * 70)

        total_experiments = (
            len(network_types)
            * len(num_nodes_range)
            * len(num_demands_range)
            * len(gate_error_range)
            * num_trials
        )

        print(f"Total experiments to run: {total_experiments}")
        print(f"Configurations: {total_experiments // num_trials}")
        print(f"Trials per configuration: {num_trials}\n")

        experiment_count = 0

        for network_type in network_types:
            for num_nodes in num_nodes_range:
                for num_demands in num_demands_range:
                    for gate_error in gate_error_range:
                        # Compute contention level
                        # Rough estimate: average node capacity is 4 qubits
                        # Each demand uses ~2 qubits on average
                        avg_capacity = num_nodes * 4
                        contention = (num_demands * 2) / avg_capacity

                        print(f"\n{'=' * 70}")
                        print(f"Configuration {experiment_count // num_trials + 1}:")
                        print(
                            f"  Network: {network_type}, Nodes: {num_nodes}, "
                            f"Demands: {num_demands}"
                        )
                        print(
                            f"  Gate error: {gate_error:.4f}, "
                            f"Contention: {contention:.2f}"
                        )
                        print(f"{'=' * 70}")

                        # Run trials with different seeds
                        for trial in range(num_trials):
                            seed = 1000 * experiment_count + trial

                            params = ExperimentParameters(
                                network_type=network_type,
                                num_nodes=num_nodes,
                                num_demands=num_demands,
                                num_candidate_paths=3,
                                contention_level=contention,
                                gate_error_2q=gate_error,
                                t1_avg=300.0,
                                t2_avg=150.0,
                                seed=seed,
                            )

                            print(
                                f"\n  Trial {trial + 1}/{num_trials} (seed={seed})..."
                            )

                            try:
                                result = self.run_single_trial(params)
                                self.results.append(result)

                                print(
                                    f"    QAOA:   utility={result.qaoa_utility:.4f}, "
                                    f"time={result.qaoa_time:.1f}s"
                                )
                                print(
                                    f"    Greedy: utility={result.greedy_utility:.4f}, "
                                    f"time={result.greedy_time:.1f}s"
                                )
                                print(
                                    f"    Advantage: {result.qaoa_advantage * 100:.2f}%"
                                )

                            except Exception as e:
                                print(f"    ERROR: {e}")
                                continue

                            experiment_count += 1

        print(f"\n{'=' * 70}")
        print(f"EXPERIMENTS COMPLETE: {len(self.results)} successful trials")
        print(f"{'=' * 70}")

    def save_results(self, filename: str = "experiment_results.csv"):
        """Save results to CSV file"""
        filepath = self.output_dir / filename
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(filepath, index=False)
        print(f"\nResults saved to {filepath}")
        return df

    def analyze_results(self) -> pd.DataFrame:
        """
        Perform statistical analysis on experimental results.

        Computes:
        - Mean and std of metrics per configuration
        - Statistical significance tests (paired t-test)
        - Confidence intervals
        - Effect sizes (Cohen's d)

        Returns:
            Summary statistics DataFrame
        """
        if not self.results:
            print("No results to analyze")
            return pd.DataFrame()

        df = pd.DataFrame([asdict(r) for r in self.results])

        print("\n" + "=" * 70)
        print("STATISTICAL ANALYSIS")
        print("=" * 70)

        # Group by configuration
        grouping_cols = ["network_type", "num_nodes", "num_demands", "gate_error_2q"]

        # Aggregate statistics
        summary = (
            df.groupby(grouping_cols)
            .agg(
                {
                    "qaoa_utility": ["mean", "std", "count"],
                    "greedy_utility": ["mean", "std"],
                    "qaoa_advantage": ["mean", "std"],
                    "qaoa_approximation_ratio": ["mean", "std"],
                    "greedy_approximation_ratio": ["mean", "std"],
                    "qaoa_time": ["mean"],
                    "greedy_time": ["mean"],
                }
            )
            .round(4)
        )

        print("\nPerformance Summary (mean ± std):")
        print(summary)

        # Statistical significance testing
        print("\n" + "=" * 70)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 70)

        significance_results = []

        for config, group in df.groupby(grouping_cols):
            if len(group) < 2:
                continue

            qaoa_utils = group["qaoa_utility"].values
            greedy_utils = group["greedy_utility"].values

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(qaoa_utils, greedy_utils)

            # Wilcoxon signed-rank test (non-parametric alternative)
            w_stat, w_p_value = stats.wilcoxon(qaoa_utils, greedy_utils)

            # Effect size (Cohen's d)
            diff = qaoa_utils - greedy_utils
            cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)

            # Confidence interval (95%)
            ci_95 = stats.t.interval(
                0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
            )

            significance_results.append(
                {
                    "network_type": config[0],
                    "num_nodes": config[1],
                    "num_demands": config[2],
                    "gate_error_2q": config[3],
                    "mean_advantage": np.mean(qaoa_utils - greedy_utils),
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "wilcoxon_p": w_p_value,
                    "cohens_d": cohens_d,
                    "ci_95_lower": ci_95[0],
                    "ci_95_upper": ci_95[1],
                    "n_trials": len(group),
                }
            )

            print(f"\nConfiguration: {config}")
            print(
                f"  QAOA utility:   {np.mean(qaoa_utils):.4f} ± {np.std(qaoa_utils):.4f}"
            )
            print(
                f"  Greedy utility: {np.mean(greedy_utils):.4f} ± {np.std(greedy_utils):.4f}"
            )
            print(f"  Mean advantage: {np.mean(qaoa_utils - greedy_utils):.4f}")
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
            print(f"  Wilcoxon test: p={w_p_value:.4f}")
            print(f"  Cohen's d: {cohens_d:.3f}")
            print(f"  95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

            if p_value < 0.05:
                print("  ✓ Statistically significant (p < 0.05)")
            else:
                print("  ✗ Not significant (p >= 0.05)")

        # Save significance results
        sig_df = pd.DataFrame(significance_results)
        sig_filepath = self.output_dir / "significance_tests.csv"
        sig_df.to_csv(sig_filepath, index=False)
        print(f"\nSignificance test results saved to {sig_filepath}")

        return summary


def main():
    """Run experiments from command line"""
    parser = argparse.ArgumentParser(
        description="Run experiments comparing QAOA vs classical routing"
    )
    parser.add_argument(
        "--network-types",
        nargs="+",
        default=["hourglass", "diamond", "grid"],
        help="Adversarial network topology types (hourglass, diamond, grid)",
    )
    parser.add_argument(
        "--num-nodes",
        nargs="+",
        type=int,
        default=[8, 10],
        help="Number of nodes to test",
    )
    parser.add_argument(
        "--num-demands",
        nargs="+",
        type=int,
        default=[3, 4],
        help="Number of demands to test",
    )
    parser.add_argument(
        "--gate-errors",
        nargs="+",
        type=float,
        default=[0.001, 0.01],
        help="Gate error rates to test",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials per configuration (10 recommended for statistical power)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )

    args = parser.parse_args()

    # Run experiments
    runner = ExperimentRunner(output_dir=args.output_dir)

    runner.run_experiment_suite(
        network_types=args.network_types,
        num_nodes_range=args.num_nodes,
        num_demands_range=args.num_demands,
        gate_error_range=args.gate_errors,
        num_trials=args.num_trials,
    )

    # Save and analyze
    runner.save_results()
    runner.analyze_results()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total successful trials: {len(runner.results)}")
    print(f"Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()

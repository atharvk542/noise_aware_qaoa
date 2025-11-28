"""
Visualization and Analysis Tools

Publication-quality plots for comparing QAOA vs classical baselines.

Generates:
- Performance comparison plots (utility, approximation ratio)
- Statistical significance visualizations
- Noise threshold analysis
- Contention level sweeps
- Convergence curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
import argparse

# Set publication-quality style
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 150


class ResultsVisualizer:
    """Generate publication-quality visualizations from experimental results"""

    def __init__(self, results_file: str, output_dir: str = "figures"):
        self.df = pd.read_csv(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print(f"Loaded {len(self.df)} experimental results")
        print(
            f"Configurations: {self.df.groupby(['network_type', 'num_nodes', 'num_demands', 'gate_error_2q']).ngroups}"
        )

    def plot_performance_comparison(self, metric: str = "utility"):
        """
        Compare QAOA vs Greedy performance across configurations.

        Args:
            metric: 'utility', 'approximation_ratio', or 'objective'
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if metric == "utility":
            qaoa_col = "qaoa_utility"
            greedy_col = "greedy_utility"
            ylabel = "Total Utility"
            title_suffix = "Utility"
        elif metric == "approximation_ratio":
            qaoa_col = "qaoa_approximation_ratio"
            greedy_col = "greedy_approximation_ratio"
            ylabel = "Approximation Ratio"
            title_suffix = "Approximation Ratio"
        else:
            qaoa_col = "qaoa_objective"
            greedy_col = "greedy_objective"
            ylabel = "Objective Value"
            title_suffix = "Objective"

        # Plot by network size
        ax = axes[0]
        grouped = self.df.groupby("num_nodes")

        x = []
        qaoa_means = []
        qaoa_stds = []
        greedy_means = []
        greedy_stds = []

        for nodes, group in grouped:
            x.append(nodes)
            qaoa_means.append(group[qaoa_col].mean())
            qaoa_stds.append(group[qaoa_col].std())
            greedy_means.append(group[greedy_col].mean())
            greedy_stds.append(group[greedy_col].std())

        x = np.array(x)
        qaoa_means = np.array(qaoa_means)
        qaoa_stds = np.array(qaoa_stds)
        greedy_means = np.array(greedy_means)
        greedy_stds = np.array(greedy_stds)

        ax.errorbar(
            x,
            qaoa_means,
            yerr=qaoa_stds,
            marker="o",
            label="QAOA",
            linewidth=2,
            capsize=5,
        )
        ax.errorbar(
            x,
            greedy_means,
            yerr=greedy_stds,
            marker="s",
            label="Sequential Greedy",
            linewidth=2,
            capsize=5,
        )

        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_suffix} vs Network Size")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot by number of demands
        ax = axes[1]
        grouped = self.df.groupby("num_demands")

        x = []
        qaoa_means = []
        qaoa_stds = []
        greedy_means = []
        greedy_stds = []

        for demands, group in grouped:
            x.append(demands)
            qaoa_means.append(group[qaoa_col].mean())
            qaoa_stds.append(group[qaoa_col].std())
            greedy_means.append(group[greedy_col].mean())
            greedy_stds.append(group[greedy_col].std())

        x = np.array(x)
        qaoa_means = np.array(qaoa_means)
        qaoa_stds = np.array(qaoa_stds)
        greedy_means = np.array(greedy_means)
        greedy_stds = np.array(greedy_stds)

        ax.errorbar(
            x,
            qaoa_means,
            yerr=qaoa_stds,
            marker="o",
            label="QAOA",
            linewidth=2,
            capsize=5,
        )
        ax.errorbar(
            x,
            greedy_means,
            yerr=greedy_stds,
            marker="s",
            label="Sequential Greedy",
            linewidth=2,
            capsize=5,
        )

        ax.set_xlabel("Number of Demands")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_suffix} vs Demand Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"performance_comparison_{metric}.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"Saved {filename}")
        plt.close()

    def plot_advantage_vs_contention(self):
        """Plot QAOA advantage vs resource contention level"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by contention level
        grouped = self.df.groupby("contention_level")

        x = []
        advantage_means = []
        advantage_stds = []

        for contention, group in grouped:
            x.append(contention)
            advantages = (group["qaoa_utility"] - group["greedy_utility"]) / group[
                "greedy_utility"
            ]
            advantage_means.append(advantages.mean())
            advantage_stds.append(advantages.std())

        x = np.array(x)
        advantage_means = np.array(advantage_means) * 100  # Convert to percentage
        advantage_stds = np.array(advantage_stds) * 100

        ax.errorbar(
            x,
            advantage_means,
            yerr=advantage_stds,
            marker="o",
            linewidth=2,
            capsize=5,
            color="C2",
        )
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        ax.set_xlabel("Resource Contention Level")
        ax.set_ylabel("QAOA Advantage over Greedy (%)")
        ax.set_title("Quantum Advantage vs Resource Contention")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = "advantage_vs_contention.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"Saved {filename}")
        plt.close()

    def plot_noise_threshold(self):
        """Analyze noise threshold where quantum advantage vanishes"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by gate error rate
        grouped = self.df.groupby("gate_error_2q")

        x = []
        qaoa_means = []
        qaoa_stds = []
        greedy_means = []
        greedy_stds = []

        for error_rate, group in grouped:
            x.append(error_rate)
            qaoa_means.append(group["qaoa_utility"].mean())
            qaoa_stds.append(group["qaoa_utility"].std())
            greedy_means.append(group["greedy_utility"].mean())
            greedy_stds.append(group["greedy_utility"].std())

        x = np.array(x)
        qaoa_means = np.array(qaoa_means)
        qaoa_stds = np.array(qaoa_stds)
        greedy_means = np.array(greedy_means)
        greedy_stds = np.array(greedy_stds)

        ax.errorbar(
            x,
            qaoa_means,
            yerr=qaoa_stds,
            marker="o",
            label="QAOA",
            linewidth=2,
            capsize=5,
        )
        ax.errorbar(
            x,
            greedy_means,
            yerr=greedy_stds,
            marker="s",
            label="Sequential Greedy",
            linewidth=2,
            capsize=5,
        )

        # Find crossover point (approximate)
        if len(x) > 1:
            diff = qaoa_means - greedy_means
            if np.any(diff > 0) and np.any(diff < 0):
                # Find approximate crossover
                for i in range(len(diff) - 1):
                    if diff[i] * diff[i + 1] < 0:
                        # Linear interpolation
                        crossover = x[i] + (x[i + 1] - x[i]) * abs(diff[i]) / (
                            abs(diff[i]) + abs(diff[i + 1])
                        )
                        ax.axvline(
                            x=crossover,
                            color="red",
                            linestyle="--",
                            label=f"Crossover ≈ {crossover:.4f}",
                        )
                        break

        ax.set_xlabel("Two-Qubit Gate Error Rate")
        ax.set_ylabel("Total Utility")
        ax.set_title("Noise Threshold Analysis")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = "noise_threshold.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"Saved {filename}")
        plt.close()

    def plot_statistical_significance(self, significance_file: str = None):
        """
        Visualize statistical significance of QAOA vs Greedy comparison.

        Args:
            significance_file: Path to significance test results CSV
        """
        if significance_file is None:
            significance_file = str(
                Path(self.df.attrs.get("results_dir", "results"))
                / "significance_tests.csv"
            )

        try:
            sig_df = pd.read_csv(significance_file)
        except FileNotFoundError:
            print(f"Significance file not found: {significance_file}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # P-value plot
        ax = axes[0, 0]
        configs = [
            f"N{row['num_nodes']}D{row['num_demands']}" for _, row in sig_df.iterrows()
        ]
        p_values = sig_df["p_value"].values
        colors = ["green" if p < 0.05 else "red" for p in p_values]

        ax.bar(range(len(configs)), p_values, color=colors, alpha=0.7)
        ax.axhline(y=0.05, color="black", linestyle="--", label="α = 0.05")
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha="right")
        ax.set_ylabel("P-value")
        ax.set_title("Statistical Significance (Paired t-test)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Effect size (Cohen's d)
        ax = axes[0, 1]
        cohens_d = sig_df["cohens_d"].values
        colors = [
            "green" if d > 0.5 else "orange" if d > 0.2 else "red" for d in cohens_d
        ]

        ax.bar(range(len(configs)), cohens_d, color=colors, alpha=0.7)
        ax.axhline(y=0.2, color="gray", linestyle=":", label="Small")
        ax.axhline(y=0.5, color="gray", linestyle="--", label="Medium")
        ax.axhline(y=0.8, color="gray", linestyle="-", label="Large")
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha="right")
        ax.set_ylabel("Cohen's d")
        ax.set_title("Effect Size")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Mean advantage with confidence intervals
        ax = axes[1, 0]
        mean_adv = sig_df["mean_advantage"].values
        ci_lower = sig_df["ci_95_lower"].values
        ci_upper = sig_df["ci_95_upper"].values

        ax.errorbar(
            range(len(configs)),
            mean_adv,
            yerr=[mean_adv - ci_lower, ci_upper - mean_adv],
            fmt="o",
            capsize=5,
            linewidth=2,
        )
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha="right")
        ax.set_ylabel("Mean Utility Difference (QAOA - Greedy)")
        ax.set_title("Mean Advantage with 95% CI")
        ax.grid(True, alpha=0.3)

        # Summary table
        ax = axes[1, 1]
        ax.axis("off")

        # Count significant results
        num_significant = (sig_df["p_value"] < 0.05).sum()
        num_total = len(sig_df)
        pct_significant = 100 * num_significant / num_total if num_total > 0 else 0

        summary_text = f"""
        Statistical Summary
        ═══════════════════════════════════
        
        Total configurations: {num_total}
        Statistically significant: {num_significant} ({pct_significant:.1f}%)
        
        P-value threshold: 0.05
        
        Effect sizes (Cohen's d):
          Large (d > 0.8): {(sig_df["cohens_d"] > 0.8).sum()}
          Medium (0.5 < d ≤ 0.8): {((sig_df["cohens_d"] > 0.5) & (sig_df["cohens_d"] <= 0.8)).sum()}
          Small (0.2 < d ≤ 0.5): {((sig_df["cohens_d"] > 0.2) & (sig_df["cohens_d"] <= 0.5)).sum()}
          Negligible (d ≤ 0.2): {(sig_df["cohens_d"] <= 0.2).sum()}
        
        Mean advantage: {sig_df["mean_advantage"].mean():.4f}
        """

        ax.text(
            0.1,
            0.5,
            summary_text,
            fontsize=11,
            family="monospace",
            verticalalignment="center",
        )

        plt.tight_layout()
        filename = "statistical_significance.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"Saved {filename}")
        plt.close()

    def plot_execution_time_comparison(self):
        """Compare computational overhead of QAOA vs Greedy"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Time vs problem size
        ax = axes[0]
        grouped = self.df.groupby("num_demands")

        x = []
        qaoa_times = []
        qaoa_stds = []
        greedy_times = []
        greedy_stds = []

        for demands, group in grouped:
            x.append(demands)
            qaoa_times.append(group["qaoa_time"].mean())
            qaoa_stds.append(group["qaoa_time"].std())
            greedy_times.append(group["greedy_time"].mean())
            greedy_stds.append(group["greedy_time"].std())

        x = np.array(x)

        ax.errorbar(
            x,
            qaoa_times,
            yerr=qaoa_stds,
            marker="o",
            label="QAOA",
            linewidth=2,
            capsize=5,
        )
        ax.errorbar(
            x,
            greedy_times,
            yerr=greedy_stds,
            marker="s",
            label="Sequential Greedy",
            linewidth=2,
            capsize=5,
        )

        ax.set_xlabel("Number of Demands")
        ax.set_ylabel("Execution Time (seconds)")
        ax.set_title("Computational Overhead")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Time vs utility scatter
        ax = axes[1]
        ax.scatter(
            self.df["qaoa_time"], self.df["qaoa_utility"], alpha=0.6, label="QAOA", s=50
        )
        ax.scatter(
            self.df["greedy_time"],
            self.df["greedy_utility"],
            alpha=0.6,
            label="Greedy",
            s=50,
        )

        ax.set_xlabel("Execution Time (seconds)")
        ax.set_ylabel("Total Utility")
        ax.set_title("Performance vs Computational Cost")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = "execution_time.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"Saved {filename}")
        plt.close()

    def generate_all_plots(self, significance_file: Optional[str] = None):
        """Generate complete set of publication-quality figures"""
        print("\n" + "=" * 70)
        print("GENERATING PUBLICATION-QUALITY FIGURES")
        print("=" * 70)

        print("\n1. Performance comparison plots...")
        self.plot_performance_comparison("utility")
        self.plot_performance_comparison("approximation_ratio")

        print("\n2. Advantage vs contention...")
        self.plot_advantage_vs_contention()

        print("\n3. Noise threshold analysis...")
        self.plot_noise_threshold()

        print("\n4. Statistical significance...")
        if significance_file:
            self.plot_statistical_significance(significance_file)

        print("\n5. Execution time comparison...")
        self.plot_execution_time_comparison()

        print("\n" + "=" * 70)
        print(f"All figures saved to {self.output_dir}/")
        print("=" * 70)


def main():
    """Generate visualizations from command line"""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality visualizations"
    )
    parser.add_argument("results_file", type=str, help="Path to experiment results CSV")
    parser.add_argument(
        "--significance-file",
        type=str,
        default=None,
        help="Path to significance tests CSV",
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures", help="Output directory for figures"
    )

    args = parser.parse_args()

    visualizer = ResultsVisualizer(args.results_file, args.output_dir)
    visualizer.generate_all_plots(args.significance_file)


if __name__ == "__main__":
    main()

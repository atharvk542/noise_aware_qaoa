# Quick Reference Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python demo.py
```

## Quick Test (5 minutes)

```bash
# Run demo with all components
python demo.py
```

Expected output:

- Network generation: 7 nodes, barbell topology
- 3 communication demands
- 2 candidate paths per demand
- QAOA optimization (20 iterations)
- Comparison with greedy baseline
- Performance metrics and advantage calculation

## Small Experiment (30 minutes)

```bash
# Test configuration with few trials
python run_experiments.py \
    --network-types barbell \
    --num-nodes 8 \
    --num-demands 3 4 \
    --gate-errors 0.005 0.01 \
    --num-trials 5 \
    --output-dir results_test
```

Results:

- `results_test/experiment_results.csv` - Raw data
- `results_test/significance_tests.csv` - Statistics

Visualize:

```bash
python visualize_results.py \
    results_test/experiment_results.csv \
    --significance-file results_test/significance_tests.csv \
    --output-dir figures_test
```

## Full Publication Suite (1-2 days)

```bash
# Comprehensive experiments with 30 trials each
python run_experiments.py \
    --network-types barbell grid random \
    --num-nodes 8 10 12 \
    --num-demands 3 4 5 \
    --gate-errors 0.0005 0.001 0.005 0.01 \
    --num-trials 30 \
    --output-dir results_full

# Generate all figures
python visualize_results.py \
    results_full/experiment_results.csv \
    --significance-file results_full/significance_tests.csv \
    --output-dir figures_full
```

## Individual Component Testing

### Network Generation

```bash
python network_generation.py
```

Tests:

- Barbell topology generation
- Node/link property assignment
- Demand creation
- K-shortest paths computation

### Noise Models

```bash
python noise_models.py
```

Tests:

- Fidelity calculation with noise
- Amplitude/phase damping
- Entanglement swapping
- Resource conflict detection

### QAOA Optimizer

```bash
python qaoa_optimizer.py
```

Tests:

- Circuit construction
- Parameter optimization
- Bitstring decoding
- Adaptive depth selection

### Classical Baselines

```bash
python classical_baselines.py
```

Tests:

- Sequential greedy routing
- Independent shortest path
- Random path selection
- Performance comparison

## Customization Examples

### Test Specific Network Type

```bash
python run_experiments.py \
    --network-types grid \
    --num-nodes 9 12 \
    --num-demands 3 \
    --gate-errors 0.005 \
    --num-trials 10
```

### Focus on Noise Threshold

```bash
python run_experiments.py \
    --network-types barbell \
    --num-nodes 10 \
    --num-demands 4 \
    --gate-errors 0.0001 0.0005 0.001 0.005 0.01 0.05 \
    --num-trials 20
```

### High Contention Study

```bash
python run_experiments.py \
    --network-types barbell \
    --num-nodes 8 \
    --num-demands 4 5 6 \
    --gate-errors 0.005 \
    --num-trials 15
```

## Output Files

### experiment_results.csv

Columns:

- `experiment_id`: Unique trial identifier
- `network_type`: barbell/grid/random
- `num_nodes`, `num_demands`: Problem size
- `contention_level`: Resource usage ratio
- `gate_error_2q`: Two-qubit gate error rate
- `seed`: Random seed for reproducibility
- `qaoa_*`: QAOA results (utility, penalty, time)
- `greedy_*`: Greedy results
- `independent_utility`: Upper bound
- `*_approximation_ratio`: Performance vs optimal
- `qaoa_advantage`: (QAOA - Greedy) / Greedy

### significance_tests.csv

Columns:

- `network_type`, `num_nodes`, `num_demands`, `gate_error_2q`: Configuration
- `mean_advantage`: Average utility difference
- `t_statistic`, `p_value`: Paired t-test results
- `wilcoxon_p`: Non-parametric test
- `cohens_d`: Effect size
- `ci_95_lower`, `ci_95_upper`: Confidence interval
- `n_trials`: Sample size

## Figures Generated

1. **performance_comparison_utility.png**

   - Left: Utility vs number of nodes
   - Right: Utility vs number of demands
   - Error bars show standard deviation

2. **performance_comparison_approximation_ratio.png**

   - Same layout, normalized to optimal performance

3. **advantage_vs_contention.png**

   - QAOA advantage percentage vs resource contention
   - Shows where quantum advantage is strongest

4. **noise_threshold.png**

   - Utility vs gate error rate (log scale)
   - Identifies crossover point

5. **statistical_significance.png**

   - 4-panel figure:
     - P-values per configuration
     - Effect sizes (Cohen's d)
     - Mean advantage with 95% CI
     - Summary statistics table

6. **execution_time.png**
   - Left: Time vs problem size
   - Right: Time-utility scatter

## Interpreting Results

### Statistical Significance

- **p < 0.05**: QAOA significantly better than Greedy
- **p ≥ 0.05**: No significant difference
- **Cohen's d > 0.8**: Large practical effect
- **Cohen's d 0.5-0.8**: Medium effect
- **Cohen's d 0.2-0.5**: Small effect

### Approximation Ratio

- **>0.95**: Excellent (near-optimal)
- **0.85-0.95**: Good
- **0.70-0.85**: Moderate
- **<0.70**: Poor (investigate why)

### QAOA Advantage

- **>20%**: Strong quantum advantage
- **10-20%**: Moderate advantage
- **5-10%**: Weak advantage
- **<5%**: Marginal/no advantage

## Troubleshooting

### "No module named 'qiskit'"

```bash
pip install qiskit qiskit-aer
```

### Experiments too slow

Reduce:

- `--num-trials` (from 30 to 10)
- Number of nodes (from 12 to 8)
- Gate error sweep points

In code:

- `num_shots` (default 1024 → 512)
- `max_iterations` (default 100 → 50)

### Memory errors

- Reduce number of demands
- Reduce candidate paths per demand (k parameter)
- Run fewer configurations in parallel

### Import errors

Ensure all files in same directory:

- network_generation.py
- noise_models.py
- qaoa_optimizer.py
- classical_baselines.py
- run_experiments.py
- visualize_results.py

## Performance Optimization

### For Faster Experiments

Edit `run_experiments.py`, modify `QAOAParameters`:

```python
qaoa_params = QAOAParameters(
    depth=1,                # Reduce from 2
    num_shots=256,         # Reduce from 1024
    max_iterations=30,     # Reduce from 50
    adaptive_depth=False   # Disable for consistency
)
```

### For Better Results

```python
qaoa_params = QAOAParameters(
    depth=3,               # Increase from 2
    num_shots=2048,        # Increase from 1024
    max_iterations=100,    # Increase from 50
    adaptive_depth=True,   # Enable
    max_depth=5           # Allow deeper circuits
)
```

## Common Workflows

### Quick Sanity Check

1. `python demo.py` - Verify installation
2. Check QAOA finds valid solution
3. Confirm QAOA ≥ Random

### Preliminary Results

1. Small experiment (5 trials)
2. Generate figures
3. Check if QAOA > Greedy in high contention
4. Decide if worth full study

### Publication Preparation

1. Full experiments (30 trials)
2. Generate all figures
3. Export CSV to analysis tool (Excel/Python)
4. Write paper sections
5. Prepare supplementary materials

## Data Analysis Tips

### Load Results in Python

```python
import pandas as pd

# Load experiment results
df = pd.read_csv('results_full/experiment_results.csv')

# Filter high contention scenarios
high_contention = df[df['contention_level'] > 0.3]

# Compute average advantage
avg_advantage = high_contention['qaoa_advantage'].mean()

# Count significant wins
sig_df = pd.read_csv('results_full/significance_tests.csv')
num_significant = (sig_df['p_value'] < 0.05).sum()
```

### Statistical Tests in Python

```python
from scipy import stats

# Paired t-test
qaoa_utils = df['qaoa_utility'].values
greedy_utils = df['greedy_utility'].values
t_stat, p_value = stats.ttest_rel(qaoa_utils, greedy_utils)

# Effect size
diff = qaoa_utils - greedy_utils
cohens_d = diff.mean() / diff.std()
```

## Contact & Support

For issues or questions:

1. Check README.md for detailed documentation
2. Review IMPLEMENTATION_SUMMARY.md for technical details
3. Inspect code comments for theory explanations
4. Open GitHub issue (if using version control)

## Citation

When publishing results:

```bibtex
@article{noise_robust_qaoa_routing_2025,
  title={Noise-Robust QAOA for Multi-Flow Entanglement Routing},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2025}
}
```

---

**Quick Start**: `python demo.py`  
**Full Experiments**: `python run_experiments.py --num-trials 30`  
**Visualize**: `python visualize_results.py results_*/experiment_results.csv`

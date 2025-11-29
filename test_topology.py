"""
Quick test to verify the new adversarial topology generates correctly
and creates conditions where greedy should fail.
"""

from adversarial_topologies import generate_adversarial_network
from network_generation import QuantumRepeaterNetwork, CandidatePathGenerator
from noise_models import EntanglementQualitySimulator, NoiseParameters
from classical_baselines import SequentialGreedyRouter, IndependentShortestPathRouter
import networkx as nx

# Generate adversarial hourglass network
G, node_props, link_props, demands = generate_adversarial_network(
    topology_type="hourglass",
    num_nodes=12,
    num_demands=5,
    seed=42,
)

print("=" * 70)
print("ADVERSARIAL HOURGLASS TOPOLOGY TEST")
print("=" * 70)

print(f"\nNetwork: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Demands: {len(demands)}")

print("\nNode Capacities:")
bottleneck_nodes = []
bypass_nodes = []
for node_id in sorted(node_props.keys()):
    capacity = node_props[node_id].num_memory_qubits
    print(f"  Node {node_id}: {capacity} qubits", end="")
    if capacity <= 2:
        print(" <- BOTTLENECK")
        bottleneck_nodes.append(node_id)
    elif capacity >= 4 and node_id not in range(4) and node_id not in range(7, 11):
        print(" <- BYPASS")
        bypass_nodes.append(node_id)
    else:
        print()

print(f"\nBottleneck nodes: {bottleneck_nodes}")
print(f"Bypass nodes: {bypass_nodes}")

# Check articulation points
articulation_points = list(nx.articulation_points(G))
print(f"Articulation points (critical nodes): {articulation_points}")

print("\nDemands (all cross from Cluster A to Cluster B):")
for d in demands:
    print(f"  Demand {d.demand_id}: {d.source} → {d.destination} (priority: {d.priority:.1f})")

print("\nLink Quality Summary:")
bottleneck_links = []
bypass_links = []
for edge in G.edges():
    source, target = edge
    if source in bottleneck_nodes or target in bottleneck_nodes:
        dist = link_props[edge].distance
        fid = link_props[edge].initial_fidelity
        bottleneck_links.append((dist, fid))
    elif source in bypass_nodes or target in bypass_nodes:
        dist = link_props[edge].distance
        fid = link_props[edge].initial_fidelity
        bypass_links.append((dist, fid))

if bottleneck_links:
    avg_bn_dist = sum(d for d, f in bottleneck_links) / len(bottleneck_links)
    avg_bn_fid = sum(f for d, f in bottleneck_links) / len(bottleneck_links)
    print(f"  Bottleneck path: avg {avg_bn_dist:.1f} km, avg {avg_bn_fid:.3f} fidelity")

if bypass_links:
    avg_bp_dist = sum(d for d, f in bypass_links) / len(bypass_links)
    avg_bp_fid = sum(f for d, f in bypass_links) / len(bypass_links)
    print(f"  Bypass path: avg {avg_bp_dist:.1f} km, avg {avg_bp_fid:.3f} fidelity")

# Now test actual routing
print("\n" + "=" * 70)
print("ROUTING TEST")
print("=" * 70)

# Create network object
network = QuantumRepeaterNetwork(seed=42)
network.graph = G
network.node_props = node_props
network.link_props = link_props

# Generate candidate paths
path_gen = CandidatePathGenerator(network)
candidate_paths = path_gen.compute_candidate_paths(demands, k=3)

print(f"\nCandidate paths found: {sum(len(paths) for paths in candidate_paths.values())}")
for demand_id, paths in candidate_paths.items():
    print(f"  Demand {demand_id}: {len(paths)} paths")
    for i, path in enumerate(paths):
        uses_bottleneck = any(node in bottleneck_nodes for node in path.nodes)
        uses_bypass = any(node in bypass_nodes for node in path.nodes)
        path_type = "BOTTLENECK" if uses_bottleneck else ("BYPASS" if uses_bypass else "OTHER")
        print(f"    Path {i}: {path.nodes} [{path_type}]")

# Setup noise model
noise_params = NoiseParameters(
    gate_error_1q=0.0005,
    gate_error_2q=0.005,
    readout_error=0.01
)
fidelity_sim = EntanglementQualitySimulator(network, noise_params)

# Compute path utilities
print("\nPath Utilities:")
for demand_id, paths in candidate_paths.items():
    print(f"  Demand {demand_id}:")
    for i, path in enumerate(paths):
        utility = fidelity_sim.compute_path_utility(path)
        uses_bottleneck = any(node in bottleneck_nodes for node in path.nodes)
        path_type = "BOTTLENECK" if uses_bottleneck else "BYPASS"
        print(f"    Path {i} [{path_type}]: utility={utility:.4f}")

# Run greedy
print("\n" + "=" * 70)
print("GREEDY ROUTING")
print("=" * 70)
greedy = SequentialGreedyRouter(network, demands, candidate_paths, fidelity_sim)
greedy_solution = greedy.solve()

print(f"Objective: {greedy_solution.objective_value:.4f}")
print(f"Total Utility: {greedy_solution.total_utility:.4f}")
print(f"Total Penalty: {greedy_solution.total_penalty:.4f}")
print(f"Valid: {greedy_solution.is_valid}")

print("\nGreedy path selections:")
for demand_id, path_id in greedy_solution.path_selections.items():
    path = candidate_paths[demand_id][path_id]
    utility = fidelity_sim.compute_path_utility(path)
    uses_bottleneck = any(node in bottleneck_nodes for node in path.nodes)
    path_type = "BOTTLENECK" if uses_bottleneck else "BYPASS"
    print(f"  Demand {demand_id}: Path {path_id} - {path.nodes} [{path_type}] (utility={utility:.4f})")

# Run independent (upper bound)
print("\n" + "=" * 70)
print("INDEPENDENT ROUTING (UPPER BOUND)")
print("=" * 70)
independent = IndependentShortestPathRouter(network, demands, candidate_paths, fidelity_sim)
independent_solution = independent.solve()

print(f"Total Utility: {independent_solution.total_utility:.4f}")
print(f"Total Penalty: {independent_solution.total_penalty:.4f}")
print(f"Valid: {independent_solution.is_valid}")

print("\nIndependent path selections:")
for demand_id, path_id in independent_solution.path_selections.items():
    path = candidate_paths[demand_id][path_id]
    utility = fidelity_sim.compute_path_utility(path)
    uses_bottleneck = any(node in bottleneck_nodes for node in path.nodes)
    path_type = "BOTTLENECK" if uses_bottleneck else "BYPASS"
    print(f"  Demand {demand_id}: Path {path_id} - {path.nodes} [{path_type}] (utility={utility:.4f})")

# Analysis
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

if independent_solution.total_utility > 0:
    greedy_gap = (independent_solution.total_utility - greedy_solution.total_utility) / independent_solution.total_utility
    print(f"Greedy gap: {greedy_gap * 100:.2f}% below optimal")
    
if greedy_gap > 0.05:  # More than 5% gap
    print("✓ ADVERSARIAL INSTANCE: Greedy has significant gap from optimal")
    print("✓ QAOA should be able to find better solution than greedy")
else:
    print("✗ WARNING: Greedy too close to optimal, QAOA may not show advantage")

if greedy_solution.is_valid and not independent_solution.is_valid:
    print("Note: Greedy found valid solution, but independent violates constraints")
    print("      This means optimal requires careful resource allocation")
elif not greedy_solution.is_valid and independent_solution.is_valid:
    print("✓ PERFECT: Greedy violates constraints, but optimal solution exists")
    print("✓ This is the ideal adversarial scenario for QAOA")

print("\n" + "=" * 70)
print("EXPECTED QAOA BEHAVIOR:")
print("=" * 70)
print("- QAOA should explore mixed strategies:")
print("  * Route 1-2 demands through bottleneck (high quality)")
print("  * Route 2-3 demands through bypass (lower quality but valid)")
print("- Expected QAOA objective: between greedy and independent")
print(f"  Greedy: {greedy_solution.objective_value:.4f}")
print(f"  Target: {independent_solution.objective_value:.4f}")
print("=" * 70)

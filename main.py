import numpy as np

from graph_utils import (
    bristol_to_node_adjacency_matrix,
    add_fanin_connections,
    make_undirected_from_directed,
    find_xor_and_cases,
    find_andXor_and_cases,
    forbidden_nodes,
)
from mis_solver import GPUMaxCut, MIS_via_MaxCut


# ─── Configuration ─────────────────────────────────────────────────────────────

# CIRCUIT_PATH = "circuits/EPFL_circuits/sample.txt"

CIRCUIT_PATH = input("Enter the path for input netlist: ")

SOLVER_CONFIG = dict(
    num_replicas = 128,
    sa_steps     = 100_000,
    initial_temp = 1.0,
    cooling_rate = 0.99995,
    block_size   = 1024,
)


# ─── Netlist Loading ───────────────────────────────────────────────────────────

def load_netlist(filepath: str) -> list:
    with open(filepath) as f:
        lines = [ln.strip() for ln in f.readlines()]

    netlist = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if 'INV' in parts:
            continue
        converted = [int(x) for x in parts[:-1]] + [parts[-1]]
        netlist.append(converted)

    return netlist


# ─── Pre-processing ────────────────────────────────────────────────────────────

def build_adjacency(netlist):
    adj_list, idx_to_node = bristol_to_node_adjacency_matrix(netlist)
    adj = np.array(adj_list, dtype=np.int32)
    adj = add_fanin_connections(adj)
    adj = make_undirected_from_directed(adj)
    return adj, idx_to_node


def collect_special_cases(netlist):
    xor_and     = find_xor_and_cases(netlist)
    andXor_and  = find_andXor_and_cases(netlist)

    print(f"  XOR→AND cases     : {len(xor_and)}")
    print(f"  AND-XOR→AND cases : {len(andXor_and)}")

    return list(set(xor_and) | set(andXor_and))


# ─── Validation ────────────────────────────────────────────────────────────────

def validate_independent_set(is_set, adj_matrix) -> bool:
    nodes = list(is_set)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if adj_matrix[u][v] == 1:
                print(f"  Error: nodes {u} and {v} are adjacent but both in the independent set!")
                return False
    return True


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nLoading circuit: {CIRCUIT_PATH}")
    netlist            = load_netlist(CIRCUIT_PATH)
    initial_gate_count = len(netlist)
    print(f"  Initial gate count: {initial_gate_count}")

    print("\nBuilding adjacency matrix...")
    adj_matrix, idx_to_node = build_adjacency(netlist)
    print(f"  Matrix shape: {adj_matrix.shape}")

    print("\nIdentifying special gate cases...")
    xorAnd_cases = collect_special_cases(netlist)
    print(f"  Total special cases: {len(xorAnd_cases)}")

    print("\nInitialising GPU solver...")
    solver = GPUMaxCut(**SOLVER_CONFIG)

    print("\nBuilding partition tree and solving MIS...")
    is_set = MIS_via_MaxCut(adj_matrix, xorAnd_cases, solver)
    print(f"  Raw independent set size: {len(is_set)}")

    forbidden = set(forbidden_nodes(netlist))
    is_set    = is_set - forbidden
    bootstrapping_reduction = len(is_set)

    print(f"\nBootstrapping reduction : {bootstrapping_reduction}")
    print(f"Final gate count        : {initial_gate_count - bootstrapping_reduction}")

    print("\nValidating independent set...")

    val_adj_list, _ = bristol_to_node_adjacency_matrix(netlist)
    val_adj         = np.array(val_adj_list, dtype=np.int32)
    val_adj         = add_fanin_connections(val_adj)
    val_adj         = make_undirected_from_directed(val_adj)

    if validate_independent_set(is_set, val_adj):
        print("  Independent set is VALID.")
    else:
        print("  Independent set is INVALID.")

    return is_set


if __name__ == "__main__":
    result = main()
    print(f"\nFinal independent set size: {len(result)}")

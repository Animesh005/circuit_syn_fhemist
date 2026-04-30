import numpy as np
import scipy.sparse as sp
import networkx as nx
import cupy as cp


def random_graph_csr(n: int, edge_prob: float = 0.01, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(n):
        k = max(1, int(n * edge_prob * 3))
        candidates = rng.choice(n, size=min(k, n - 1), replace=False)
        keep = candidates[rng.random(len(candidates)) < edge_prob * n / k]
        keep = keep[keep != i]
        rows.append(np.full(len(keep), i, dtype=np.int32))
        cols.append(keep.astype(np.int32))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.int32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int32)
    adj = (adj + adj.T)
    adj.data[:] = 1
    adj.eliminate_zeros()
    return adj


def subgraph_csr_sparse(adj: np.ndarray, nodes) -> sp.csr_matrix:
    nodes = np.array(nodes, dtype=np.int32)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    rows, cols = np.nonzero(adj)
    new_rows, new_cols = [], []
    for r, c in zip(rows, cols):
        if r in node_to_idx and c in node_to_idx:
            new_rows.append(node_to_idx[r])
            new_cols.append(node_to_idx[c])
    data = np.ones(len(new_rows), dtype=np.int32)
    csr = sp.csr_matrix((data, (new_rows, new_cols)), shape=(len(nodes), len(nodes)))
    csr = csr + csr.T
    csr.data[:] = 1
    csr.eliminate_zeros()
    return csr


def subgraph_csr_preserve_labels(adj: np.ndarray, nodes) -> sp.csr_matrix:
    nodes = set(nodes)
    rows, cols = np.nonzero(adj)
    keep_rows, keep_cols = [], []
    for r, c in zip(rows, cols):
        if r in nodes and c in nodes:
            keep_rows.append(r)
            keep_cols.append(c)
    data = np.ones(len(keep_rows), dtype=np.int32)
    csr = sp.csr_matrix((data, (keep_rows, keep_cols)), shape=adj.shape, dtype=np.int32)
    csr.eliminate_zeros()
    return csr


def load_adj_txt_csr(filepath: str) -> sp.csr_matrix:
    adj_dense = np.loadtxt(filepath, dtype=np.int32)
    assert adj_dense.ndim == 2 and adj_dense.shape[0] == adj_dense.shape[1]
    return sp.csr_matrix(adj_dense)


def load_adj_txt(filepath: str) -> np.ndarray:
    adj = np.loadtxt(filepath, dtype=np.int32)
    assert adj.ndim == 2 and adj.shape[0] == adj.shape[1]
    return adj


def count_edges(adj: np.ndarray) -> int:
    return int(np.sum(np.triu(adj, k=1)))


def ndarray_to_csr_sparse(A: np.ndarray) -> sp.csr_matrix:
    rows, cols = np.nonzero(A)
    data = np.ones(len(rows), dtype=np.int32)
    csr = sp.csr_matrix((data, (rows, cols)), shape=A.shape)
    csr = csr + csr.T
    csr.data[:] = 1
    csr.eliminate_zeros()
    return csr


def adjacency_matrix_to_edge_list(adj_matrix):
    edges = []
    n = len(adj_matrix)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))
    return edges


def make_undirected_from_directed(adj_matrix):
    n = len(adj_matrix)
    new_adj = np.copy(adj_matrix)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1 or adj_matrix[j][i] == 1:
                new_adj[i][j] = 1
                new_adj[j][i] = 1
    return new_adj


def graph_to_csr(n, edge_list):
    adj = [[] for _ in range(n)]
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
    indptr, indices = [0], []
    for neighbors in adj:
        indices.extend(neighbors)
        indptr.append(len(indices))
    return cp.array(indices, dtype=cp.int32), cp.array(indptr, dtype=cp.int32)


def build_graph_from_adjacency(adj_matrix, node_list):
    G = nx.Graph()
    G.add_nodes_from(node_list)
    n = len(node_list)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                G.add_edge(node_list[i], node_list[j])
    return G


def add_fanin_connections(adj_matrix):
    n = len(adj_matrix)
    new_adj = np.copy(adj_matrix)
    for j in range(n):
        fanin_nodes = [i for i in range(n) if adj_matrix[i][j] == 1]
        for idx1 in range(len(fanin_nodes)):
            for idx2 in range(idx1 + 1, len(fanin_nodes)):
                u, v = fanin_nodes[idx1], fanin_nodes[idx2]
                new_adj[u][v] = 1
                new_adj[v][u] = 1
    return new_adj


def bristol_to_node_adjacency_matrix(gate_lines):
    G = nx.DiGraph()
    wire_to_node = {}
    node_set = {}

    for parts in gate_lines:
        num_inputs   = int(parts[0])
        num_outputs  = int(parts[1])
        input_wires  = tuple(sorted(map(int, parts[2:2 + num_inputs])))
        output_wires = list(map(int, parts[2 + num_inputs:2 + num_inputs + num_outputs]))
        gate_type    = parts[-1]
        node_id      = (gate_type, input_wires)
        if node_id not in node_set:
            node_set[node_id] = len(node_set)
        for wire in output_wires:
            wire_to_node[wire] = node_id

    for parts in gate_lines:
        num_inputs  = int(parts[0])
        input_wires = list(map(int, parts[2:2 + num_inputs]))
        gate_type   = parts[-1]
        input_node  = (gate_type, tuple(sorted(input_wires)))
        for wire in input_wires:
            if wire in wire_to_node:
                producer_node = wire_to_node[wire]
                if producer_node != input_node:
                    G.add_edge(node_set[producer_node], node_set[input_node])

    n = len(node_set)
    adj_matrix = np.zeros((n, n), dtype=int)
    for u, v in G.edges():
        adj_matrix[u][v] = 1

    idx_to_node = {idx: node for node, idx in node_set.items()}
    return adj_matrix.tolist(), idx_to_node


def find_xor_and_cases(netlist):
    xor_and = []
    for i, gate in enumerate(netlist):
        if gate[-1] == 'XOR':
            out_wire = gate[-2]
            for other in netlist:
                if other[5] == 'AND' and out_wire in (other[2], other[3]):
                    xor_and.append(i)
                    break
    return xor_and


def find_andXor_and_cases(netlist):
    andXor_and = []
    for i, gate in enumerate(netlist):
        if gate[-1] == 'AND-XOR':
            out_wire = gate[-2]
            for other in netlist:
                if other[-1] == 'AND' and out_wire in (other[2], other[3]):
                    andXor_and.append(i)
                    break
    return andXor_and


def forbidden_nodes(netlist):
    multi_input = []
    for i, gate in enumerate(netlist):
        if gate[-1] in ('AND3', 'XOR3', 'AND-XOR'):
            in1, in2, in3 = gate[2], gate[3], gate[4]
            multi_input.append(i)
            for j, other in enumerate(netlist):
                if other[-1] in ('AND', 'XOR'):
                    if in1 == other[-2] or in2 == other[-2] or in3 == other[-2]:
                        multi_input.append(j)
    return multi_input

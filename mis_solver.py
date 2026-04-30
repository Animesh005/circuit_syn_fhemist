import ctypes
import numpy as np
import scipy.sparse as sp
import cupy as cp
from cupy import RawKernel
from numba import njit, prange
from collections import defaultdict

from cuda_utils import (
    managed_alloc, managed_free, prefetch_to_gpu,
    ptr_to_cupy, ptr_to_numpy,
    cudaMemPrefetchAsync, cudaDeviceSynchronize,
)
from graph_utils import subgraph_csr_sparse


# ─── CUDA Kernel Source ────────────────────────────────────────────────────────

MAXCUT_KERNEL = r"""
#include <curand_kernel.h>
#include <math.h>

extern "C" {

__global__ void init_rng(
    curandState* __restrict__ states,
    unsigned long long seed,
    int n_replicas
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_replicas) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void sa_maxcut_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int*        partitions,
    int*        best_cuts,
    curandState* __restrict__ rng_states,
    int N,
    int R,
    int steps,
    float T0,
    float cooling
) {
    int rep = blockIdx.x * blockDim.x + threadIdx.x;
    if (rep >= R) return;

    curandState local_rng = rng_states[rep];
    int* part = &partitions[(long long)rep * N];

    for (int i = 0; i < N; ++i)
        part[i] = (int)(curand(&local_rng) & 1);

    int cur_cut = 0;
    for (int i = 0; i < N; ++i)
        for (int e = row_ptr[i]; e < row_ptr[i + 1]; ++e) {
            int j = col_idx[e];
            if (j > i && part[j] != part[i]) cur_cut++;
        }

    int best_cut = cur_cut;
    float T = T0;

    for (int step = 0; step < steps; ++step) {
        int v  = (int)(curand(&local_rng) % (unsigned int)N);
        int pv = part[v];
        int delta = 0;
        for (int e = row_ptr[v]; e < row_ptr[v + 1]; ++e) {
            int u = col_idx[e];
            delta += (part[u] == pv) ? 1 : -1;
        }
        bool accept = (delta > 0) ||
                      (curand_uniform(&local_rng) < __expf((float)delta / T));
        if (accept) {
            part[v] ^= 1;
            cur_cut += delta;
            if (cur_cut > best_cut) best_cut = cur_cut;
        }
        T *= cooling;
    }
    best_cuts[rep]  = best_cut;
    rng_states[rep] = local_rng;
}

} // extern "C"
"""


# ─── GPU Solver ────────────────────────────────────────────────────────────────

class GPUMaxCut:
    _CURAND_STATE_BYTES = 64

    def __init__(
        self,
        num_replicas: int   = 128,
        sa_steps:     int   = 100000,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.99995,
        seed:         int   = 42,
        block_size:   int   = 256,
    ):
        self.num_replicas = num_replicas
        self.sa_steps     = sa_steps
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.seed         = seed
        self.block_size   = block_size
        self._compile_kernels()

    def _compile_kernels(self):
        opts = ("-O3", "--use_fast_math")
        self._rng_kernel = RawKernel(MAXCUT_KERNEL, "init_rng",
                                     backend="nvcc", options=opts)
        self._sa_kernel  = RawKernel(MAXCUT_KERNEL, "sa_maxcut_kernel",
                                     backend="nvcc", options=opts)

    def solve(self, adj_csr: sp.csr_matrix, map_list: list) -> dict:
        N  = adj_csr.shape[0]
        R  = self.num_replicas
        BS = self.block_size
        blocks = (R + BS - 1) // BS
        dev    = cp.cuda.Device().id

        row_ptr_cp = cp.asarray(adj_csr.indptr.astype(np.int32))
        col_idx_cp = cp.asarray(adj_csr.indices.astype(np.int32))

        part_bytes = R * N * 4
        cuts_bytes = R * 4
        rng_bytes  = R * self._CURAND_STATE_BYTES

        part_ptr = managed_alloc(part_bytes)
        cuts_ptr = managed_alloc(cuts_bytes)
        rng_ptr  = managed_alloc(rng_bytes)

        try:
            ptr_to_numpy(part_ptr, (R * N,),    np.int32)[:] = 0
            ptr_to_numpy(cuts_ptr, (R,),         np.int32)[:] = 0
            ptr_to_numpy(rng_ptr,  (rng_bytes,), np.uint8)[:] = 0

            d_partitions = ptr_to_cupy(part_ptr, (R * N,),    np.int32)
            d_best_cuts  = ptr_to_cupy(cuts_ptr, (R,),         np.int32)
            d_rng        = ptr_to_cupy(rng_ptr,  (rng_bytes,), np.uint8)

            prefetch_to_gpu(part_ptr, part_bytes, dev)
            prefetch_to_gpu(cuts_ptr, cuts_bytes, dev)
            prefetch_to_gpu(rng_ptr,  rng_bytes,  dev)
            cp.cuda.Stream.null.synchronize()

            self._rng_kernel(
                (blocks,), (BS,),
                (d_rng, np.uint64(self.seed), np.int32(R)),
            )
            cp.cuda.Stream.null.synchronize()

            t0 = cp.cuda.Event()
            t1 = cp.cuda.Event()
            t0.record()

            self._sa_kernel(
                (blocks,), (BS,),
                (
                    row_ptr_cp, col_idx_cp,
                    d_partitions, d_best_cuts, d_rng,
                    np.int32(N), np.int32(R),
                    np.int32(self.sa_steps),
                    np.float32(self.initial_temp),
                    np.float32(self.cooling_rate),
                ),
            )

            t1.record()
            t1.synchronize()
            print(f"  GPU kernel time: {cp.cuda.get_elapsed_time(t0, t1):.1f} ms")

            cudaMemPrefetchAsync(
                ctypes.c_void_p(part_ptr), ctypes.c_size_t(part_bytes),
                ctypes.c_int(-1), ctypes.c_void_p(None),
            )
            cudaMemPrefetchAsync(
                ctypes.c_void_p(cuts_ptr), ctypes.c_size_t(cuts_bytes),
                ctypes.c_int(-1), ctypes.c_void_p(None),
            )
            cudaDeviceSynchronize()

            h_best_cuts  = ptr_to_numpy(cuts_ptr, (R,),   np.int32).copy()
            h_partitions = ptr_to_numpy(part_ptr, (R, N), np.int32).copy()

        finally:
            managed_free(part_ptr)
            managed_free(cuts_ptr)
            managed_free(rng_ptr)

        best_rep  = int(np.argmax(h_best_cuts))
        partition = h_partitions[best_rep]

        return {
            "cut_value": int(h_best_cuts[best_rep]),
            "partition": partition,
            "set_a":     [map_list[i] for i in np.where(partition == 0)[0]],
            "set_b":     [map_list[i] for i in np.where(partition == 1)[0]],
        }


# ─── Partition Tree ────────────────────────────────────────────────────────────

class TreeNode:
    def __init__(self, nodes):
        self.nodes = nodes
        self.left  = None
        self.right = None


def is_independent(adj, nodes):
    nodes = list(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if adj[nodes[i]][nodes[j]] != 0:
                return False
    return True


def count_leaves(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def build_partition_tree(adj, nodes, solver: GPUMaxCut):
    nodes = set(nodes)
    node  = TreeNode(nodes)

    if len(nodes) <= 1 or is_independent(adj, nodes):
        return node

    result = solver.solve(subgraph_csr_sparse(adj, list(nodes)), list(nodes))
    S = set(result['set_a'])
    T = set(result['set_b'])

    if not S or not T:
        return node

    node.left  = build_partition_tree(adj, S, solver)
    node.right = build_partition_tree(adj, T, solver)
    return node


# ─── Candidate Classification ──────────────────────────────────────────────────

def precompute_degrees(adj: np.ndarray) -> np.ndarray:
    return adj.sum(axis=1).astype(np.int32)


def classify_pairs_vectorized(left_ids, right_ids, bad_mask, adj, degrees) -> dict:
    L = left_ids[:, None]
    R = right_ids[None, :]

    vi_bad      = bad_mask[L]
    vj_bad      = bad_mask[R]
    only_vi_bad = vi_bad & ~vj_bad
    only_vj_bad = vj_bad & ~vi_bad
    neither_bad = ~vi_bad & ~vj_bad

    pair_adj    = adj[L, R]
    independent = neither_bad & (pair_adj == 0)
    deg_rule    = neither_bad & (pair_adj != 0)
    pick_vi     = deg_rule & (degrees[L] <= degrees[R])
    pick_vj     = deg_rule & (degrees[L] >  degrees[R])

    candidates = defaultdict(set)

    rows, cols = np.where(only_vi_bad)
    candidates['from_right'].update(right_ids[cols].tolist())

    rows, cols = np.where(only_vj_bad)
    candidates['from_left'].update(left_ids[rows].tolist())

    rows, cols = np.where(independent)
    candidates['from_left'].update(left_ids[rows].tolist())
    candidates['from_right'].update(right_ids[cols].tolist())

    rows, cols = np.where(pick_vi)
    candidates['from_left'].update(left_ids[rows].tolist())

    rows, cols = np.where(pick_vj)
    candidates['from_right'].update(right_ids[cols].tolist())

    return candidates


@njit(parallel=True, cache=True)
def classify_pairs_numba(
    left_ids, right_ids, bad_mask, adj, degrees,
    out_left_candidates, out_right_candidates,
):
    nl = len(left_ids)
    nr = len(right_ids)
    for i in prange(nl):
        vi     = left_ids[i]
        vi_bad = bad_mask[vi]
        for j in range(nr):
            vj     = right_ids[j]
            vj_bad = bad_mask[vj]
            if vi_bad and vj_bad:
                continue
            if vi_bad:
                out_right_candidates[vj] = True
                continue
            if vj_bad:
                out_left_candidates[vi] = True
                continue
            if adj[vi, vj] == 0:
                out_left_candidates[vi] = True
                out_right_candidates[vj] = True
                continue
            if degrees[vi] <= degrees[vj]:
                out_left_candidates[vi] = True
            else:
                out_right_candidates[vj] = True


def resolve_candidates(candidate_ids, adj, degrees, existing_combined_mask):
    accepted = existing_combined_mask.copy()
    feasible = candidate_ids[adj[candidate_ids, :] @ accepted == 0]
    if len(feasible) == 0:
        return accepted
    feasible = feasible[np.argsort(degrees[feasible])]
    for node in feasible:
        if adj[node, :] @ accepted == 0:
            accepted[node] = True
    return accepted


def can_add(adj, v, combined, index):
    iv = index[v]
    for u in combined:
        if adj[index[u]][iv] == 1:
            return False
    return True


def accelerated_combine(left_set, right_set, xor_and_cases, adj, index, combined):
    n         = adj.shape[0]
    degrees   = precompute_degrees(adj)
    left_ids  = np.array([index[v] for v in left_set],  dtype=np.int32)
    right_ids = np.array([index[v] for v in right_set], dtype=np.int32)

    bad_mask = np.zeros(n, dtype=bool)
    for v in xor_and_cases:
        if v in index:
            bad_mask[index[v]] = True

    existing_mask = np.zeros(n, dtype=bool)
    for v in combined:
        if v in index:
            existing_mask[index[v]] = True

    pair_count    = len(left_ids) * len(right_ids)
    MEM_THRESHOLD = 50_000_000  # 50M pairs → switch to Numba

    if pair_count < MEM_THRESHOLD:
        candidates    = classify_pairs_vectorized(left_ids, right_ids, bad_mask, adj, degrees)
        all_candidates = np.array(
            list(candidates['from_left'] | candidates['from_right']),
            dtype=np.int32
        )
    else:
        out_left  = np.zeros(n, dtype=bool)
        out_right = np.zeros(n, dtype=bool)
        classify_pairs_numba(left_ids, right_ids, bad_mask, adj, degrees, out_left, out_right)
        all_candidates = np.where(out_left | out_right)[0].astype(np.int32)

    final_mask = resolve_candidates(all_candidates, adj, degrees, existing_mask)

    rev_index  = {v: k for k, v in index.items()}
    new_nodes  = {rev_index[i] for i in np.where(final_mask)[0]}
    combined.clear()
    combined.update(new_nodes)
    return combined


def combine_sets(adj, node, node_list, xor_and_cases):
    if node is None:
        return set()
    if node.left is None and node.right is None:
        return set(node.nodes)

    index     = {v: i for i, v in enumerate(node_list)}
    left_set  = combine_sets(adj, node.left,  node_list, xor_and_cases)
    right_set = combine_sets(adj, node.right, node_list, xor_and_cases)

    combined  = set()
    combined  = accelerated_combine(left_set, right_set, xor_and_cases, adj, index, combined)

    if not combined:
        for v in left_set.union(right_set):
            if can_add(adj, v, combined, index):
                combined.add(v)

    return combined


def MIS_via_MaxCut(adj_matrix, xorAnd_cases, solver: GPUMaxCut) -> set:
    N    = len(adj_matrix)
    root = build_partition_tree(adj_matrix, range(N), solver)
    S    = combine_sets(adj_matrix, root, range(N), xorAnd_cases)
    return S


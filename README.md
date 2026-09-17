# Circuit Synthesis for FHEMIST

GPU-accelerated synthesis of Boolean netlists that **reduces the number of bootstrapping
operations** required to evaluate a circuit homomorphically.

The artifact implements a two-stage optimization pipeline:

| Stage | Script | What it does |
|-------|--------|--------------|
| **CSO** — *(expand acronym)* | `cso_synthesis.py` | Structural rewriting: fuses chains of 2-input gates into multi-input gates (`AND3`, `XOR3`, `AND-XOR`), shrinking the gate count before MIS is run. |
| **CMIS** — *(expand acronym)* | `main.py` | Selects a maximum independent set of gates in a conflict graph derived from the netlist. Gates in the set can safely skip bootstrapping, so the size of the set is the bootstrapping reduction. |

The MIS problem is solved on the GPU by recursively bipartitioning the conflict graph with a
**simulated-annealing Max-Cut solver** (a custom CUDA kernel running 128 independent replicas
in parallel over CUDA Managed Memory), then merging the partitions bottom-up into a single
independent set.

The two stages compose: running CSO first and then CMIS over the CSO-optimized netlist gives
the best reduction.

---

## Repository layout

```
circuit_syn_fhemist/
├── main.py                       # CMIS driver — MIS-based bootstrapping minimization
├── cso_synthesis.py              # CSO driver  — multi-input gate fusion
├── mis_solver.py                 # CUDA SA Max-Cut kernel, partition tree, set-merging
├── graph_utils.py                # netlist → conflict graph, special-case detection
├── cuda_utils.py                 # cudaMallocManaged / prefetch wrappers (HMM)
├── requirements.txt
└── circuits/
    ├── EPFL_circuits/            # 21 EPFL Combinational Benchmark Suite netlists
    ├── SCALE_MAMBA_circuits/     # 20 SCALE-MAMBA / Bristol Fashion netlists
    └── CSO_circuits/             # output directory for CSO-optimized netlists
```

---

## Requirements

### Hardware

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA GPU with **Heterogeneous Memory Management (HMM)** support *(fill in the exact model and VRAM you evaluated on)* |
| Driver | NVIDIA **open** kernel modules, r535 or newer (HMM is not available on the proprietary modules) |
| Host RAM | Scales with circuit size — the conflict graph is stored as a dense `n × n` matrix *(fill in what you used)* |

> **Why HMM?** `mis_solver.py` allocates the replica/partition buffers with
> `cudaMallocManaged` and migrates them on demand between host and device
> (`cuda_utils.py`). On large circuits these buffers exceed device memory, and HMM is what
> lets the kernel oversubscribe GPU memory transparently.

### Software

| Component | Version |
|-----------|---------|
| OS | *(fill in, e.g. Ubuntu 22.04)* |
| CUDA Toolkit | 12.x — **`nvcc` must be on `PATH`**, the Max-Cut kernel is JIT-compiled with the `nvcc` backend |
| Python | *(fill in, e.g. 3.10)* |
| Python packages | see `requirements.txt` |

---

## Installation

### 1. Enable HMM

Follow NVIDIA's guide to enable and verify HMM on the machine:
<https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/>

Verify it is active:

```bash
cat /sys/module/nvidia/parameters/uvm_disable_hmm   # expect: N
nvidia-smi                                          # confirm driver + GPU
```

### 2. Point the environment at the CUDA toolkit

```bash
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nvcc --version   # must succeed
```

### 3. Install the Python dependencies

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`cupy-cuda12x` is pinned for CUDA 12. If you are on CUDA 11, install `cupy-cuda11x` instead.

---

## Input format

Netlists use **Bristol-style gate lines**, one gate per line, with the file header removed:

```
<#inputs> <#outputs> <input wires...> <output wire> <GATE>
```

Examples:

```
2 1 128 0 257 XOR          # 2-input XOR: wires 128, 0  →  wire 257
2 1 0 128 258 AND          # 2-input AND: wires 0, 128  →  wire 258
3 1 4 8 5 12 AND3          # 3-input AND produced by the CSO stage
```

Recognized gate types: `AND`, `XOR`, `INV`, and — after CSO — `AND3`, `XOR3`, `AND-XOR`.
`main.py` drops `INV` lines when loading, since inverters are free in TFHE-style evaluation
and never need bootstrapping.

> **Note:** the netlists in `circuits/` have already had their Bristol headers stripped
> (the two header lines giving wire/input/output counts). If you bring your own Bristol
> Fashion circuit, remove those header lines first.

---

## Usage

All three entry points read their paths from **stdin prompts**, so they can be driven
interactively or by piping input.

### A. CMIS only (baseline)

```bash
python3 main.py
```
```
Enter the path for input netlist: circuits/EPFL_circuits/adder.txt
```

Or non-interactively:

```bash
echo "circuits/EPFL_circuits/adder.txt" | python3 main.py
```

Output:

```
Loading circuit: circuits/EPFL_circuits/adder.txt
  Initial gate count: 890

Building adjacency matrix...
  Matrix shape: (N, N)

Identifying special gate cases...
  XOR→AND cases     : ...
  AND-XOR→AND cases : ...

Initialising GPU solver...

Building partition tree and solving MIS...
  GPU kernel time: ... ms
  Raw independent set size: ...

Bootstrapping reduction : R      <-- gates that do NOT need bootstrapping
Final gate count        : 890-R  <-- bootstrapping operations remaining

Validating independent set...
  Independent set is VALID.
```

The last check re-derives the conflict graph from scratch and verifies that no two selected
gates are adjacent — i.e. that the reported reduction is sound.

### B. Hybrid CSO + CMIS (full pipeline)

**Step 1 — run CSO to produce a fused netlist:**

```bash
python3 cso_synthesis.py
```
```
Enter the path for input netlist:  circuits/EPFL_circuits/adder.txt
Enter the path for output netlist: circuits/CSO_circuits/adder.txt
```

```
Netlist size before:  890
Netlist size after AND3 gates:     ...
Netlist size after XOR3 gates:     ...
Netlist size after AND-XOR gates:  ...

Final netlist size:  572
```

**Step 2 — run CMIS over the CSO output:**

```bash
python3 main.py
```
```
Enter the path for input netlist: circuits/CSO_circuits/adder.txt
```

Scripted end to end:

```bash
printf 'circuits/EPFL_circuits/adder.txt\ncircuits/CSO_circuits/adder.txt\n' | python3 cso_synthesis.py
echo   'circuits/CSO_circuits/adder.txt' | python3 main.py
```

`circuits/CSO_circuits/adder.txt` is shipped as a reference output — re-running step 1 should
reproduce it byte for byte (572 gates: 191 `AND3`, 64 `XOR3`, 63 `AND-XOR`, 127 `AND`, 127 `XOR`).

---

## How it works

### 1. CSO — multi-input gate fusion (`cso_synthesis.py`)

Three rewrite passes, applied in order. Each fuses a producer gate into its consumer when the
intermediate wire has **exactly one** consumer (so the producer's output is not needed
elsewhere):

| Pattern | Rewritten as |
|---------|--------------|
| `AND → AND` | `AND3` |
| `XOR → XOR` | `XOR3` |
| `AND → XOR` | `AND-XOR` |

The fused gate is evaluated with a single (multi-value) bootstrap, so every fusion removes one
gate — and one bootstrap — from the circuit.

### 2. Conflict graph construction (`graph_utils.py`)

* `bristol_to_node_adjacency_matrix` — each distinct `(gate_type, sorted_input_wires)` pair
  becomes one node, so structurally identical gates are deduplicated. A directed edge runs
  from a producer node to each node that consumes its output wire.
* `add_fanin_connections` — gates that feed the *same* consumer are made mutually adjacent,
  since they cannot both skip bootstrapping.
* `make_undirected_from_directed` — symmetrizes the result into the final conflict graph.

### 3. Forbidden and special-case gates

* `find_xor_and_cases` / `find_andXor_and_cases` — gates whose output feeds an `AND`. These are
  passed to the solver as `bad_mask` and are de-prioritized during set merging.
* `forbidden_nodes` — multi-input gates (`AND3`, `XOR3`, `AND-XOR`) and their immediate
  predecessors. These are subtracted from the final set in `main.py`, because a fused gate's
  inputs must arrive at a refreshed noise level.

### 4. MIS via recursive Max-Cut (`mis_solver.py`)

```
build_partition_tree(G):
    if G is independent or |G| <= 1: return leaf
    (S, T) <- GPU simulated-annealing Max-Cut of G
    return node(build_partition_tree(S), build_partition_tree(T))

combine_sets(tree):
    bottom-up; at each internal node merge the left and right independent sets,
    resolving every conflicting pair by the bad_mask rule, then by lowest degree
```

The CUDA kernel (`sa_maxcut_kernel`) runs `num_replicas` independent annealing chains, one per
thread, over a CSR copy of the subgraph; the best cut across replicas defines the split.
Pair classification during merging is vectorized with NumPy, and automatically switches to a
parallel Numba kernel above 50 M candidate pairs.

---

## Configuration

Solver parameters live in `SOLVER_CONFIG` at the top of `main.py`:

```python
SOLVER_CONFIG = dict(
    num_replicas = 128,        # parallel annealing chains (one CUDA thread each)
    sa_steps     = 100_000,    # annealing steps per chain
    initial_temp = 1.0,        # T0
    cooling_rate = 0.99995,    # geometric cooling, T *= rate each step
    block_size   = 1024,       # CUDA threads per block
)
```

Raising `num_replicas` and `sa_steps` improves cut quality at the cost of runtime and managed
memory (`num_replicas × N × 4` bytes per partition buffer). The solver seed is fixed
(`seed=42` in `GPUMaxCut`), so a given configuration is deterministic for a given GPU.

---

## Included benchmarks

Gate counts below are as shipped, before any optimization. `INV` gates are listed separately
because `main.py` skips them.

<details>
<summary><b>EPFL Combinational Benchmark Suite</b> (21 circuits)</summary>

| Circuit | Gates | AND | XOR |
|---------|------:|----:|----:|
| toy | 12 | 5 | 7 |
| ctrl | 124 | 121 | 3 |
| router | 218 | 180 | 38 |
| int2float | 237 | 233 | 4 |
| dec | 304 | 304 | 0 |
| cavlc | 685 | 670 | 15 |
| adder | 890 | 572 | 318 |
| priority | 974 | 974 | 0 |
| i2c | 1,301 | 1,292 | 9 |
| max | 2,865 | 2,865 | 0 |
| bar | 3,141 | 3,141 | 0 |
| sin | 4,461 | 3,811 | 650 |
| voter | 8,282 | 6,192 | 2,090 |
| arbiter | 11,839 | 11,839 | 0 |
| square | 14,584 | 11,545 | 3,039 |
| sqrt | 19,203 | 17,242 | 1,961 |
| multiplier | 19,416 | 14,895 | 4,521 |
| log2 | 24,670 | 20,523 | 4,147 |
| div | 38,010 | 30,085 | 7,925 |
| mem_ctrl | 46,442 | 46,059 | 383 |
| hyp | 166,948 | 122,755 | 44,193 |

</details>

<details>
<summary><b>SCALE-MAMBA / Bristol Fashion circuits</b> (20 circuits)</summary>

| Circuit | Gates | AND | XOR | INV |
|---------|------:|----:|----:|----:|
| neg64 | 190 | 62 | 63 | 64 |
| 64-bit_add | 376 | 63 | 313 | 0 |
| sub64 | 439 | 63 | 313 | 63 |
| FP-eq | 1,217 | 315 | 65 | 837 |
| FP-lt | 1,536 | 381 | 257 | 898 |
| FP-floor | 1,613 | 651 | 595 | 367 |
| FP-ceil | 1,618 | 650 | 597 | 371 |
| FP-f2i | 3,932 | 1,467 | 1,625 | 840 |
| FP-i2f | 7,136 | 2,416 | 3,605 | 1,115 |
| mult64 | 13,675 | 4,033 | 9,642 | 0 |
| FP-add | 15,637 | 5,385 | 8,190 | 2,062 |
| divide64 | 29,926 | 4,664 | 24,817 | 445 |
| aes_128 | 36,663 | 6,400 | 28,176 | 2,087 |
| aes_192 | 41,565 | 7,168 | 32,080 | 2,317 |
| FP-mul | 44,899 | 19,626 | 21,947 | 3,326 |
| aes_256 | 50,666 | 8,832 | 39,008 | 2,826 |
| sha256 | 135,073 | 22,573 | 110,644 | 1,856 |
| FP-div | 184,007 | 82,269 | 84,151 | 17,587 |
| FP-sqrt | 211,524 | 91,504 | 100,120 | 19,900 |
| sha512 | 349,617 | 57,947 | 286,724 | 4,946 |

`neg64.txt` additionally contains 1 `EQW` gate.

</details>

---

## Suggested evaluation order

1. **Smoke test** — `echo circuits/EPFL_circuits/toy.txt | python3 main.py`
   (12 gates; confirms CUDA, `nvcc` JIT and HMM are all working, finishes in seconds).
2. **Small circuit, both modes** — run CMIS alone on `EPFL_circuits/adder.txt`, then the
   hybrid pipeline, and compare the two `Bootstrapping reduction` numbers.
3. **Reference output check** — regenerate `circuits/CSO_circuits/adder.txt` and `diff` it
   against the shipped copy.
4. **Scale up** — `cavlc`, `i2c`, `sin`, then the larger EPFL / SCALE-MAMBA circuits as time
   and memory allow.

*(Add your measured runtimes and reduction figures here so reviewers know what to expect.)*

---

## Known limitations

* **Dense adjacency matrix.** The conflict graph is materialized as a dense `n × n` NumPy
  array, so host memory grows quadratically with the number of distinct gates. The largest
  circuits (`hyp`, `sha512`, `FP-sqrt`) need a machine with substantial RAM.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `cudaMallocManaged failed with code 2` | Out of managed memory. Lower `num_replicas`, or confirm HMM is enabled — without it, managed allocations cannot exceed VRAM. |
| `nvrtc`/`nvcc` compilation error on startup | `nvcc` is not on `PATH`. Re-run the `export CUDA_HOME` / `export PATH` block. |
| `OSError: libcudart.so: cannot open shared object file` | Add `$CUDA_HOME/lib64` to `LD_LIBRARY_PATH`. |
| `ImportError: cupy` / wrong CUDA version | Install the CuPy wheel matching your toolkit: `cupy-cuda12x` or `cupy-cuda11x`. |
| Host RAM exhausted on a large circuit | Expected — see *Known limitations*. Start with smaller benchmarks. |

---

## Acknowledgements

Benchmark circuits are taken from the
[EPFL Combinational Benchmark Suite](https://github.com/lsils/benchmarks) and the
[SCALE-MAMBA](https://github.com/KULeuven-COSIC/SCALE-MAMBA) / Bristol Fashion circuit
collections.

# CUDA Optimization Studies

Profiling-driven GPU kernel optimization case studies on an NVIDIA RTX 2000 Ada (Ada
Lovelace, CC 8.9), using **Nsight Compute** to identify each bottleneck empirically before
fixing it — and validated against NVIDIA's production libraries (CUB, Thrust, cuBLAS).

The organizing idea across both studies: **classify the bottleneck first — memory-bound or
compute-bound — then optimize the binding constraint.** The two studies are deliberate
opposites, covering both halves of the roofline.

## Studies

### [Parallel Reduction](reduction/) — a memory-bound problem
Sum-reduction over 32M floats, optimized from naive atomics to **97% of peak DRAM bandwidth
(124×)**, matching `cub::DeviceReduce` and `thrust::reduce`.

The bottleneck path: **atomic contention → shared-memory/sync overhead → DRAM bandwidth →
ceiling reached.** The goal is *saturating memory bandwidth*; the study ends by proving the
97% figure is the true hardware ceiling (a further optimization yields nothing) and
validating against the vendor libraries, which land at the same 93–97% wall.

### [Matrix Multiplication](matmul/) — a compute-bound problem
Single-precision GEMM at N=4096, optimized from naive to a register-tiled kernel reaching
**55% of cuBLAS throughput** (within 1.8× of NVIDIA's production GEMM), from scratch.

The bottleneck path: **redundant global reads → shared-memory throughput → balanced
compute/memory**, by reusing data harder at each level of the hierarchy (global → shared →
registers). Includes the counterintuitive occupancy/arithmetic-intensity tradeoff and an
instructive 8×8 regression showing that optimization parameters must be tuned *jointly*.

## What these demonstrate

- **Profiler-driven methodology** — every optimization is chosen from Nsight Compute
  evidence (Speed-Of-Light, occupancy, memory workload), not guessed.
- **Both halves of the roofline** — a memory-bound kernel driven to the bandwidth ceiling,
  and a compute-bound kernel driven toward the compute ceiling.
- **Honest engineering** — reported null results (float4 on an already-bandwidth-bound
  reduction) and regressions (8×8 register tiling), with profiler-backed explanations of
  *why*, and validation against production libraries rather than inflated claims.

## Hardware & tools

NVIDIA RTX 2000 Ada Laptop GPU (CC 8.9), CUDA 13.x, Nsight Compute. Build commands and
per-stage details are in each study's README.
# CUDA Optimization Studies

Profiling-driven GPU kernel optimization case studies on an NVIDIA RTX 2000 Ada (CC 8.9),
using Nsight Compute to identify and fix bottlenecks empirically.

## Studies

### [Parallel Reduction](reduction/)
Sum-reduction optimized from naive atomics to **97% of peak DRAM bandwidth (124×)**,
matching NVIDIA's CUB library. A memory-bound problem: the goal is saturating bandwidth.
Bottleneck path: atomic contention → shared-memory/sync overhead → DRAM bandwidth.

### [Matrix Multiplication](matmul/)
*(in progress)* — a compute-bound problem, the mirror image of the reduction: the goal is
raising compute utilization through shared-memory tiling and data reuse.
# CUDA Parallel Reduction — A Profiling-Driven Optimization Study

Optimizing a sum-reduction over **32M floats (128 MB)** on an NVIDIA RTX 2000 Ada
(Ada Lovelace, CC 8.9), using **Nsight Compute** to identify the binding bottleneck at
each stage before fixing it. Every optimization is chosen from profiler evidence, not
guessed — including a deliberate null result that demonstrates when to stop.

## Results

Benchmarked on 32M floats, profiled with Nsight Compute (`--set full`).

| Stage | Kernel | Duration | DRAM % | L1 % | Compute % | Speedup |
|-------|--------|----------|--------|------|-----------|---------|
| 1 | Naive global atomics | 65.7 ms | 0.8% | 2.9% | 1.4% | 1× |
| 2 | Shared-memory tree | 1.74 ms | 30% | 90% | 90% | 38× |
| 3 | Grid-stride + warp shuffle | 0.529 ms | **97%** | 22% | 11% | **124×** |
| 4 | Vectorized float4 | 0.529 ms | 97% | 22% | 3% | 124× |

The bottleneck moves at every stage: **atomic contention → shared-memory/sync overhead →
DRAM bandwidth → (ceiling reached).**

## The optimization journey

### Stage 1 — Naive global atomics (65.7 ms)

Every thread does one `atomicAdd` to a single global address. An atomic is an indivisible
read-modify-write, so the hardware serializes all ~32M of them on that one location — a
queue millions deep. The "parallelism" is illusory.

The profiler shows the damage starkly: **DRAM 0.8%, compute 1.4%** — the GPU does almost
nothing but wait in the atomic queue. This is contention in its purest form, and it gets
*relatively worse* as the array grows, which is why the eventual speedup is so large.

### Stage 2 — Shared-memory tree (1.74 ms, 38×)

Each block reduces its own elements privately in `__shared__` memory (on-chip, ~L1 speed)
using a tree — 256 → 128 → 64 → … → 1 in log₂(256) = 8 steps — then performs **one** atomic
per block. Contention drops from ~32M atomics to a few thousand. A `__syncthreads()` after
each tree step ensures all writes from one step are visible before the next step reads.

**The key diagnostic moment:** intuition says "memory-bound, so DRAM is the limit." The
profiler disagrees — **L1 at 90%, DRAM at only 30%.** The kernel reads the data once
(cheap) and then spends all its time churning it through the shared-memory tree with a
barrier between every step. The bottleneck is **shared-memory bandwidth and
`__syncthreads` overhead, not DRAM.** You optimize what the data says, not what intuition
says.

### Stage 3 — Grid-stride loading + warp shuffle (0.529 ms, 124×)

Two optimizations targeting the L1/sync bottleneck the profiler exposed:

**Grid-stride loading.** Instead of one element per thread, launch a fixed small grid (256
blocks) where each thread sums *many* elements from DRAM into a register before the tree:

```cpp
for (int idx = i; idx < n; idx += gridSize)
    mySum += in[idx];
```

Consecutive threads read consecutive addresses (**coalesced** — the warp's 32 loads merge
into wide transactions), so this streams real DRAM bandwidth. It also shrinks the tree
dramatically, cutting shared-memory churn and barriers.

**Warp shuffle for the last 32.** The 32 threads of a warp execute in lockstep, so once the
tree narrows to one warp they need neither shared memory nor `__syncthreads()`. Register-to-
register shuffles finish the reduction:

```cpp
for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
```

This replaces the last 5 barrier-laden tree steps with 5 barrier-free register exchanges,
directly killing the sync overhead.

**Result: 97% of peak DRAM bandwidth.** The bottleneck has moved all the way to memory —
the correct terminal state for a reduction, since reading every element once is irreducible.

### Stage 4 — Vectorized float4 loads (0.529 ms) — a deliberate null result

Each thread loads 128 bits (four floats) per instruction via the `float4` type:

```cpp
const float4* in4 = reinterpret_cast<const float4*>(in);
float4 v = in4[idx];
mySum += v.x + v.y + v.z + v.w;
```

**This produced no measurable improvement** — DRAM 97.31% vs 97.22%, identical duration.
And that is the point. Grid-stride coalesced loading had *already* saturated DRAM bandwidth,
so wider per-thread loads have no headroom: you cannot push more through a bus that is
already full. float4 helps when a kernel is *instruction-issue*-bound; this kernel was
*bandwidth*-bound first, so the optimization was redundant.

Reporting a null result with the profiler-backed reason is the point of measurement
discipline — it confirms stage 3 as the true endpoint.

## Benchmarking note (why 32M, not 1M)

An earlier run at 1M elements showed stage 3 topping out at ~85% DRAM. At 32M it reaches
**97%.** The difference is fixed overhead — kernel launch, the final atomics, ramp-up —
which is a large fraction of a microsecond-scale kernel but negligible when the kernel runs
for ~0.5 ms. **Benchmark at a size where the kernel runs long enough that fixed costs
disappear**, or bandwidth numbers read pessimistically.

## Build & profile

```bash
nvcc -arch=sm_89 03_gridstride_shuffle.cu -o reduction
ncu --set full -o report ./reduction     # profile with Nsight Compute
```

Files, in optimization order:
- `01_naive_atomic.cu` — one atomic per element (contention-bound)
- `02_shared_tree.cu` — per-block shared-memory tree (L1/sync-bound)
- `03_gridstride_shuffle.cu` — grid-stride + warp shuffle (bandwidth-bound, 97% peak)
- `04_float4_vectorized.cu` — float4 loads (null result — already at ceiling)

## Key takeaway

The bottleneck moved at every stage — atomic contention → shared-memory/sync → DRAM
bandwidth — and each fix targeted wherever the profiler said the constraint currently was,
verified by re-measurement. The final stage reaches **97% of peak DRAM bandwidth (124×
over naive)** and then *proves* the ceiling by showing a further optimization yields nothing.
Knowing when a kernel is done — and demonstrating it with data — is as much the skill as the
speedups.
```
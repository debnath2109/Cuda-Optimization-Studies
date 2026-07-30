#include <cstdio>

__global__ void float4Optimized(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    // Opt 1: grid-stride accumulate into a register, but load 4 floats at a time
    const float4* in4 = reinterpret_cast<const float4*>(in);
    int n4 = n / 4;                                  // number of float4 elements

    float mySum = 0.0f;
    for (int idx = i; idx < n4; idx += gridSize) {
        float4 v = in4[idx];                         // one 128-bit load, 4 floats
        mySum += v.x + v.y + v.z + v.w;              // sum the four lanes
    }

    sdata[tid] = mySum;
    __syncthreads();

    // Shared-memory tree, but stop at 32 (one warp)
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Opt 2: last 32 via warp shuffle — no shared mem, no __syncthreads
    if (tid < 32) {
        float val = sdata[tid] + sdata[tid + 32];   // fold in the 33-64 range first
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0)
            atomicAdd(out, val);
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int n = 32000000;              // ~32 million elements
    size_t bytes = n * sizeof(float);

    // 1. Allocate host (CPU) memory and fill it
    float *h_A = (float*)malloc(bytes);
    float h_out = 0.0f;
    for (int i = 0; i < n; i++) { h_A[i] = 1.0f;}

    // 2. Allocate device (GPU) memory
    float *d_A;
    float *d_out;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));

    // 3. Copy inputs from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    // 4. Launch the kernel: choose threads-per-block and number-of-blocks
    int threadsPerBlock = 256;
    int blocks = 120;
    float4Optimized<<<blocks, threadsPerBlock>>>(d_A, d_out, n);
    cudaDeviceSynchronize();

    
    cudaMemset(d_out, 0, sizeof(float));
    cudaEventRecord(start);
    float4Optimized<<<blocks, threadsPerBlock>>>(d_A, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);
    
    // 5. Copy result back from device to host
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Verify (every element should be 1.0)
    bool ok = true;
    if (h_out != 32000000.0f) { ok = false; }
    printf("Result: %f\n", h_out);
    printf("%s\n", ok ? "PASS: all elements correct" : "FAIL");

    // 7. Free memory
    cudaFree(d_A); cudaFree(d_out);
    free(h_A);
    return 0;
}
#include <cstdio>

// The kernel: runs on the GPU. Each thread handles one element.
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    if (i < n)               // guard: don't run off the end
        C[i] = A[i] + B[i];
}

int main() {
    int n = 1 << 20;              // ~1 million elements
    size_t bytes = n * sizeof(float);

    // 1. Allocate host (CPU) memory and fill it
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // 2. Allocate device (GPU) memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. Copy inputs from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. Launch the kernel: choose threads-per-block and number-of-blocks
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;  // ceil(n/256)
    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // 5. Copy result back from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 6. Verify (every element should be 3.0)
    bool ok = true;
    for (int i = 0; i < n; i++)
        if (h_C[i] != 3.0f) { ok = false; break; }
    printf("%s\n", ok ? "PASS: all elements correct" : "FAIL");

    // 7. Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}

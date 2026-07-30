#include <cstdio>

__global__ void reduceNaive(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(out, in[i]);   // every thread hammers the same address
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = 1000000;              // ~1 million elements
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
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;  // ceil(n/256)
    reduceNaive<<<blocks, threadsPerBlock>>>(d_A, d_out, n);
    cudaDeviceSynchronize();
    
    cudaMemset(d_out, 0, sizeof(float));
    cudaEventRecord(start);
    reduceNaive<<<blocks, threadsPerBlock>>>(d_A, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);


    // 5. Copy result back from device to host
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Verify (every element should be 3.0)
    bool ok = true;
    if (h_out != 1000000.0f) { ok = false; }
    printf("Result: %f\n", h_out);
    printf("%s\n", ok ? "PASS: all elements correct" : "FAIL");

    // 7. Free memory
    cudaFree(d_A); cudaFree(d_out);
    free(h_A);
    return 0;
}
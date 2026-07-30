#include <cub/cub.cuh>
#include <cstdio>

int main() {
    int n = 32'000'000;
    size_t bytes = n * sizeof(float);

    // Host data
    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) h_data[i] = 1.0f;

    // Device input and output
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));       // single float result
    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);

    // --- CUB's two-call pattern ---
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    // Call 1: pass nullptr for temp storage -> CUB fills temp_bytes with what it needs
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);

    // Allocate exactly that much scratch space
    cudaMalloc(&d_temp, temp_bytes);

    // Call 2: same call, now with real temp storage -> actually runs the reduction
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);

    // Copy result back
    float sum = 0.0f;
    cudaMemcpy(&sum, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("CUB sum: %.1f\n", sum);          // expect 32000000.0

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_temp);
    free(h_data);
    return 0;
}
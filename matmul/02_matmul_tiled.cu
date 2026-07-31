#include <iostream>
__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    
    constexpr int TILE = 16; // Assuming a tile size of 16 for this example
    __shared__ float As[TILE][TILE];    // <-- you have none of this
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    for (int t = 0; t < N / TILE; t++) {
        
        As[ty][tx] = A[row * N + (t * TILE + tx)];   // load tile into shared mem
        Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        __syncthreads();                              // <-- you have no barriers

        for (int k = 0; k < TILE; k++)
            sum += As[ty][k] * Bs[k][tx];             // read from SHARED, not global
        __syncthreads();
    }
    C[row * N + col] = sum;
    
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N = 4096;
    int bytes = N * N * sizeof(float);    

    // 1. Allocate host (CPU) memory and fill it
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) { h_A[i] = h_B[i] = 1.0f; }

    // 2. Allocate device (GPU) memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. Copy inputs from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. Launch the kernel: choose threads-per-block and number-of-blocks
    dim3 threadsPerBlock(16, 16);                              // 256 threads/block
    dim3 blocks((N + 15) / 16, (N + 15) / 16);                // grid covers N×N
    matmulTiled<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    matmulTiled<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);


    // 5. Copy result back from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 6. Verify (every element should be 4096.0)
    bool ok = true;
    for (int i = 0; i < N * N; i++) {
        if (h_C[i] != 4096.0f) { ok = false; break; }
    }
    std::cout << (ok ? "PASS: all elements correct" : "FAIL") << std::endl;


    // 7. Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
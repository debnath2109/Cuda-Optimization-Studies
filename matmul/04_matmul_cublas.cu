#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

int main() {
    int N = 4096;
    size_t bytes = (size_t)N * N * sizeof(float);

    // Host matrices, all ones
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for (long i = 0; i < (long)N * N; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;   // C = alpha*A*B + beta*C

    // Warm-up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,     // swapped: B first
                d_A, N,     // then A  -> handles row-major via the transpose trick
                &beta,
                d_C, N);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);

    // Verify
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (long i = 0; i < (long)N * N; i++)
        if (h_C[i] != (float)N) { ok = false; break; }
    printf("cuBLAS: %s, time = %.3f ms\n", ok ? "PASS" : "FAIL", ms);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
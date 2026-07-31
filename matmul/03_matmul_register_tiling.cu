#include <iostream>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
// #define TM 8
// #define TN 8

__global__ void matmulRegisterTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BM][BK];   // 64 x 16
    __shared__ float Bs[BK][BN];   // 16 x 64

    // This thread's position in the 16x16 thread block
    int tx = threadIdx.x;          // 0..15
    int ty = threadIdx.y;          // 0..15

    // The top-left corner of this block's 64x64 output tile
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // This thread computes a 4x4 micro-tile. Its top-left element in C:
    int threadRow = ty * TM;       // 0,4,8,...,60
    int threadCol = tx * TN;       // 0,4,8,...,60

    // 16 accumulators for the 4x4 micro-tile, in registers
    float acc[TM][TN] = {0.0f};

    // Loop over k-strips
    for (int t = 0; t < N / BK; t++) {

        // --- COOPERATIVE LOAD (stage-1 simple version) ---
        // 256 threads must fill As (64x16=1024) and Bs (16x64=1024).
        // Simple approach: each thread loads 4 elements of each via a flat loop.
        int tid = ty * 16 + tx;                 // flat thread id 0..255
        for (int i = tid; i < BM * BK; i += 256) {
        // int tid = ty * 8 + tx;             // was ty*16+tx
        // for (int i = tid; i < BM * BK; i += 64) {
            int r = i / BK;                     // row in As (0..63)
            int c = i % BK;                     // col in As (0..15)
            As[r][c] = A[(blockRow + r) * N + (t * BK + c)];
        }
        for (int i = tid; i < BK * BN; i += 256) {
        // for (int i = tid; i < BK * BN; i += 64) {
            int r = i / BN;                     // row in Bs (0..15)
            int c = i % BN;                     // col in Bs (0..63)
            Bs[r][c] = B[(t * BK + r) * N + (blockCol + c)];
        }
        __syncthreads();

        // --- COMPUTE: 4x4 outer product accumulation ---
        //   load 4 A-values (As[threadRow + 0..3][k]) into a small register array,
        //   load 4 B-values (Bs[k][threadCol + 0..3]) into another,
        //   then acc[m][n] += a_reg[m] * b_reg[n] for all m,n in 0..3
        for (int k = 0; k < BK; k++) {
            float a_reg[TM], b_reg[TN];

            for (int m = 0; m < TM; m++)
                a_reg[m] = As[threadRow + m][k];      // 4 values down a column of As

            for (int n = 0; n < TN; n++)
                b_reg[n] = Bs[k][threadCol + n];      // 4 values across a row of Bs

            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    acc[m][n] += a_reg[m] * b_reg[n];  // 4x4 outer product
        }

        __syncthreads();
    }

    // --- 4x4 micro-tile to C ---
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            C[(blockRow + threadRow + m) * N + (blockCol + threadCol + n)] = acc[m][n];
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
    // dim3 threadsPerBlock(8, 8);        // 64 threads now
    dim3 blocks(N / BN, N / BM);               // N/64 × N/64 blocks
    matmulRegisterTiled<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    matmulRegisterTiled<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
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
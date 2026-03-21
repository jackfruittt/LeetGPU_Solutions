#include <cuda_runtime.h>

#define TILE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// constexpr int BM = 64; // block tile rows
// constexpr int BN = 64; // block tile cols
// constexpr int BK = 16; // block tile depth
// constexpr int TM = 4; // thread computes TM Rows
// constexpr int TN = 44; // thread computes TN cols

__global__ void matmul_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int K, int N)
{
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];
    // __shared__ float a_tile[BK][BM];
    // __shared__ float b_tile[BK][BN];

    int tx  = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;

    for (int phase = 0; phase < CEIL_DIV(K, TILE); ++phase) {
        int a_col = phase * TILE + tx;
        int b_row = phase * TILE + ty;

        // size_t cast prevents 32-bit overflow on large matrices
        a_tile[ty][tx] = (row < M  && a_col < K)
                         ? A[(size_t)row  * (size_t)K + (size_t)a_col] : 0.0f;
        b_tile[ty][tx] = (b_row < K && col  < N)
                         ? B[(size_t)b_row * (size_t)N + (size_t)col]  : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            sum += a_tile[ty][i] * b_tile[i][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[(size_t)row * (size_t)N + (size_t)col] = sum;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threads(TILE, TILE);
    dim3 blocks(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
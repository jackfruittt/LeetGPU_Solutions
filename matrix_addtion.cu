// Using B200

// #include <cuda_runtime.h>

// __global__ void matrix_add(const float* __restrict__ A,
//                             const float* __restrict__ B,
//                             float* __restrict__ C, int total) {
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;

//     if (tid * 4 + 3 < total) {
//         float4 a = reinterpret_cast<const float4*>(A)[tid];
//         float4 b = reinterpret_cast<const float4*>(B)[tid];
//         reinterpret_cast<float4*>(C)[tid] =
//             make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
//     } else {
//         for (int i = tid * 4; i < total; i++)
//             C[i] = A[i] + B[i];
//     }
// }

// // A, B, C are device pointers (i.e. pointers to memory on the GPU)
// extern "C" void solve(const float* A, const float* B, float* C, int N) {
//     int total = N * N;
//     int threadsPerBlock = 256;
//     int blocksPerGrid = ((total + 3) / 4 + threadsPerBlock - 1) / threadsPerBlock;

//     matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, total);
//     cudaDeviceSynchronize();
// }

// Using Tesla T4
#include <cuda_runtime.h>

__global__ void matrix_add(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N * N) return;
    C[tid] = A[tid] + B[tid];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    // int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = ( N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
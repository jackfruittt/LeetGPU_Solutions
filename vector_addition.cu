#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int idx = i * 4;
    if(idx + 3 < N) {
        float4 a = *reinterpret_cast<const float4*>(&A[idx]);
        float4 b = *reinterpret_cast<const float4*>(&B[idx]);
        float4 c;
    

    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;

    *reinterpret_cast<float4*>(&C[idx]) = c;
}
    else {
        for(int j = idx; j < N; j++) {
            C[j] = A[j] + B[j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

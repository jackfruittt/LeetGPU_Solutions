#include <cuda_runtime.h>

#define TILE 32
#define BR 2
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void matrix_transpose_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int rows, int cols) {

    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // coalesced load into shared memory
    #pragma unroll
    for (int j = 0; j < TILE; j += BR) {
        if (x < cols && (y + j) < rows)
            tile[threadIdx.y + j][threadIdx.x] = input[(size_t)(y + j) * cols + x];
    }
    __syncthreads();

    // write transposed - swap block indices
    int out_x = blockIdx.y * TILE + threadIdx.x;
    int out_y = blockIdx.x * TILE + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE; j += BR) {
        if (out_x < rows && (out_y + j) < cols)
            output[(size_t)(out_y + j) * rows + out_x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threads(TILE, BR);   // BR threads in y dimension
    dim3 blocks(CEIL_DIV(cols, TILE), CEIL_DIV(rows, TILE));
    matrix_transpose_kernel<<<blocks, threads>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
# LeetGPU Solutions

My solutions to [LeetGPU](https://www.leetgpu.com/) challenges - a platform for practicing GPU programming and optimization.

## Solutions

### 1. Vector Addition (`vector_addtion.cu`)
- Optimized vector addition using CUDA
- Implements vectorized memory access with `float4` for improved throughput
- Handles edge cases for non-aligned array sizes

### 2. Matrix Multiplication (`matrix_multiplication.cu`)
- Tiled matrix multiplication using shared memory
- 32x32 tile size for efficient memory access patterns
- Reduces global memory accesses through shared memory caching

## Environment

All solutions are tested on:
- NVIDIA Tesla T4
- NVIDIA B200

## Building & Running

Solutions follow the LeetGPU submission format with an `extern "C" void solve(...)` entry point that can be called from the platform's test harness.

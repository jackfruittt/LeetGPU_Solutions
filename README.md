# LeetGPU Solutions

My solutions to [LeetGPU](https://www.leetgpu.com/) challenges - a platform for practicing GPU programming and optimization.

## Solutions

### 1. Vector Addition (`vector_addtion.cu`)
- Optimized vector addition using CUDA
- Implements vectorized memory access with `float4` for improved throughput
- Handles edge cases for non-aligned array sizes
- **Performance**: 25.1 percentile (T4), 81.5 percentile (B200)

### 2. Matrix Multiplication (`matrix_multiplication.cu`)
- Tiled matrix multiplication using shared memory
- 32x32 tile size for efficient memory access patterns
- Reduces global memory accesses through shared memory caching
- **Performance**: 76.9 percentile (T4), 38.5 percentile (B200)

### 3. Matrix Transpose (`matrix_transpose.cu`)
- Efficient matrix transpose using shared memory tiling
- 32x32 tile size with bank conflict avoidance (TILE + 1 padding)
- Block row striding (BR=2) to maximize thread utilization
- Coalesced memory access for both reads and writes
- **Performance**: 85.0 percentile (T4), 86.6 percentile (B200)

## Environment

All solutions are tested on:
- NVIDIA Tesla T4
- NVIDIA B200

## Building & Running

Solutions follow the LeetGPU submission format with an `extern "C" void solve(...)` entry point that can be called from the platform's test harness.

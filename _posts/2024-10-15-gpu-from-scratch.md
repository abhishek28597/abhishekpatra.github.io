---
layout: post
title: "Building a GPU from Scratch (Theoretically)"
date: 2024-10-15
categories: [hardware, gpu]
tags: [gpu, cuda, parallel-computing]
excerpt: "What would it take to build a GPU from first principles? A thought experiment in parallel computing architecture."
reading_time: 10
---

Everyone uses GPUs for deep learning, but how many of us really understand what's happening at the hardware level? Let's do a thought experiment: if we wanted to build a GPU from scratch, what would we need?

## The Core Insight: Embarrassing Parallelism

CPUs are built for sequential tasks with complex branching logic. They have sophisticated branch predictors, large caches, and out-of-order execution units. GPUs take the opposite approach: many simple cores doing the same thing at the same time.

The key observation is that many computational problems—graphics rendering, matrix multiplication, convolution—involve doing the same operation on different pieces of data. This is called Single Instruction, Multiple Data (SIMD) parallelism.

## Architecture Basics

A modern GPU consists of:

1. **Streaming Multiprocessors (SMs)**: Groups of cores that execute in lockstep
2. **CUDA Cores**: The actual processing units (really just ALUs)
3. **Memory Hierarchy**: Registers, shared memory, L1/L2 cache, global memory
4. **Warp Schedulers**: Hardware that manages thread execution

Let's design a minimal SM:

```c
// Simplified SM execution model
struct StreamingMultiprocessor {
    int cores[32];           // 32 CUDA cores
    float shared_memory[48 * 1024];  // 48KB shared memory
    WarpScheduler scheduler;
    
    void execute_warp(Instruction inst, int warp_id) {
        // All 32 threads in a warp execute the same instruction
        for (int thread = 0; thread < 32; thread++) {
            cores[thread].execute(inst, warp_id * 32 + thread);
        }
    }
};
```

## The Memory Problem

The dirty secret of GPU programming is that computation is rarely the bottleneck—memory is. A modern GPU can perform thousands of operations in the time it takes to fetch one value from global memory.

Consider matrix multiplication:

```python
# Naive implementation - memory bound
def matmul_naive(A, B, C):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
```

Each multiplication requires two memory reads. With global memory latency around 400 cycles and an FMA (fused multiply-add) taking 1 cycle, we're wasting 99.75% of our theoretical compute!

## The Solution: Memory Hierarchy

GPUs solve this with an elaborate memory hierarchy:

- **Registers**: 0 cycle latency, ~255 per thread
- **Shared Memory**: ~20 cycles, 48-164KB per SM
- **L1 Cache**: ~28 cycles, merged with shared memory
- **L2 Cache**: ~200 cycles, several MB
- **Global Memory**: ~400 cycles, tens of GB

The art of GPU programming is organizing your computation to maximize reuse at each level:

```cuda
// Tiled matrix multiplication - much better memory efficiency
__global__ void matmul_tiled(float *A, float *B, float *C) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < gridDim.x; t++) {
        // Load tiles into shared memory
        tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // Compute on tiles
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

## Thread Divergence: The Hidden Performance Killer

Remember how all threads in a warp execute the same instruction? What happens with branching code?

```cuda
if (threadIdx.x < 16) {
    // Path A: threads 0-15
    expensive_operation_A();
} else {
    // Path B: threads 16-31
    expensive_operation_B();
}
```

The GPU executes BOTH paths for ALL threads, masking out results for inactive threads. This effectively doubles execution time. This is why GPU kernels should minimize branching.

## Building the Instruction Set

A minimal GPU instruction set needs:

- **Arithmetic**: FMA, ADD, MUL, etc.
- **Memory**: LD, ST, with various addressing modes
- **Synchronization**: BAR (barrier), MEMBAR (memory fence)
- **Control Flow**: BRA (branch), EXIT

Modern GPUs add specialized instructions:

- **Tensor Cores**: Matrix multiply-accumulate operations
- **Ray Tracing**: BVH traversal and intersection testing
- **Atomics**: For parallel reductions

## The Modern Beast

A modern datacenter GPU like the H100 has:

- 16,896 CUDA cores
- 528 Tensor Cores  
- 80GB HBM3 memory at 3.35TB/s bandwidth
- 50MB L2 cache

But it's still fundamentally doing the same thing: executing the same instruction across many threads, with clever tricks to hide memory latency.

## Why This Matters for Deep Learning

Understanding GPU architecture explains many deep learning phenomena:

1. **Why batch size matters**: Larger batches = better GPU utilization
2. **Why convolutions are fast**: Perfect memory access patterns
3. **Why RNNs are slow**: Sequential dependencies prevent parallelization
4. **Why mixed precision works**: Tensor Cores provide 2-4x speedup for FP16

## Could We Actually Build One?

Building a competitive GPU would require:

- **Silicon design**: Years of work and millions in tooling
- **Drivers**: Massive software engineering effort  
- **Ecosystem**: CUDA's moat is its software, not hardware

But understanding the principles helps us write better code. Every time you reshape a tensor to improve memory coalescing or increase batch size to improve utilization, you're thinking like a GPU architect.

The beauty of GPUs is that they're conceptually simple—just many cores doing the same thing—but the engineering to make them fast is endlessly deep. It's a perfect example of how simple ideas, taken to their extreme, can revolutionize computing.

// greedy_reduction_packed_cuda.cu
// CNMS greedy reduction kernel for packed tensor format
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Packed format kernel: processes variable-length batches using offset array
// retain is int32 (1 = retain, 0 = remove) for proper alignment of atomics
__global__ void greedy_reduction_packed_kernel(
    const int* __restrict__ sorted_indices,  // (N,) global sorted indices
    const int* __restrict__ idx,             // (N, K) neighbor indices
    const int* __restrict__ offset,          // (B,) cumulative counts
    int* __restrict__ retain,                // (N,) output: 1 = retain, 0 = remove
    int num_batches,
    int num_neighbors,
    int ignore_idx
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= num_batches) return;
    
    int batch_start = (batch_idx == 0) ? 0 : offset[batch_idx - 1];
    int batch_end = offset[batch_idx];
    int batch_size = batch_end - batch_start;
    
    if (batch_size == 0) return;
    
    extern __shared__ int shared_retain[];
    
    for (int i = tid; i < batch_size; i += blockDim.x) {
        shared_retain[i] = 1;
    }
    __syncthreads();
    
    for (int i = 0; i < batch_size; i++) {
        int global_idx = sorted_indices[batch_start + i];
        int local_idx = global_idx - batch_start;
        if (local_idx < 0 || local_idx >= batch_size) continue;
        
        if (shared_retain[local_idx] == 0) continue;
        
        for (int j = tid; j < num_neighbors; j += blockDim.x) {
            int neighbor_global = idx[global_idx * num_neighbors + j];
            if (neighbor_global != ignore_idx && neighbor_global != global_idx) {
                int neighbor_local = neighbor_global - batch_start;
                if (neighbor_local >= 0 && neighbor_local < batch_size) {
                    shared_retain[neighbor_local] = 0;
                }
            }
        }
        __syncthreads();
    }
    
    for (int i = tid; i < batch_size; i += blockDim.x) {
        retain[batch_start + i] = shared_retain[i];
    }
}


// Large batch version: use int32 retain for aligned atomics
__global__ void greedy_reduction_packed_large_kernel(
    const int* __restrict__ sorted_indices,
    const int* __restrict__ idx,
    const int* __restrict__ offset,
    int* __restrict__ retain,
    int num_batches,
    int num_neighbors,
    int ignore_idx
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= num_batches) return;
    
    int batch_start = (batch_idx == 0) ? 0 : offset[batch_idx - 1];
    int batch_end = offset[batch_idx];
    int batch_size = batch_end - batch_start;
    
    if (batch_size == 0) return;
    
    for (int i = tid; i < batch_size; i += blockDim.x) {
        retain[batch_start + i] = 1;
    }
    __syncthreads();
    
    for (int i = 0; i < batch_size; i++) {
        int global_idx = sorted_indices[batch_start + i];
        int local_idx = global_idx - batch_start;
        if (local_idx < 0 || local_idx >= batch_size) continue;
        
        int should_process = retain[batch_start + local_idx];
        __syncthreads();
        
        if (should_process == 0) continue;
        
        for (int j = tid; j < num_neighbors; j += blockDim.x) {
            int neighbor_global = idx[global_idx * num_neighbors + j];
            if (neighbor_global != ignore_idx && neighbor_global != global_idx) {
                int neighbor_local = neighbor_global - batch_start;
                if (neighbor_local >= 0 && neighbor_local < batch_size) {
                    atomicAnd(&retain[batch_start + neighbor_local], 0);
                }
            }
        }
        __syncthreads();
    }
}


void launch_greedy_reduction_packed_cuda(
    const int* sorted_indices,
    const int* idx,
    const int* offset,
    int* retain,
    int num_batches,
    int total_points,
    int num_neighbors,
    int ignore_idx,
    int max_batch_size
) {
    int threads = 256;
    int blocks = num_batches;
    
    if (max_batch_size <= 4096) {
        int shared_mem_size = max_batch_size * sizeof(int);
        greedy_reduction_packed_kernel<<<blocks, threads, shared_mem_size>>>(
            sorted_indices, idx, offset, retain,
            num_batches, num_neighbors, ignore_idx
        );
    } else {
        greedy_reduction_packed_large_kernel<<<blocks, threads>>>(
            sorted_indices, idx, offset, retain,
            num_batches, num_neighbors, ignore_idx
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Failed: %s\n", cudaGetErrorString(err));
    }
}

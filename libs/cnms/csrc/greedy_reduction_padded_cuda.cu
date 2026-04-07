// greedy_reduction_padded_cuda.cu
// CNMS greedy reduction kernel for padded tensor format (B, P) layout
// Ported from PoLAr-MAE/extensions/cnms with int retain for proper atomic alignment.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Shared-memory kernel for small num_spheres (<=4096).
// One block per batch, shared int[] for retain flags.
__global__ void greedy_reduction_padded_kernel(
    const int* __restrict__ sorted_indices,  // (B, P)
    const int* __restrict__ idx,             // (B, P, K)
    const int* __restrict__ lengths,         // (B,)
    int* __restrict__ retain,                // (B, P)
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
) {
    extern __shared__ int shared_retain[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= num_batches) return;

    int valid_length = lengths[batch_idx];

    // init shared memory
    for (int i = tid; i < num_spheres; i += blockDim.x) {
        shared_retain[i] = (i < valid_length) ? 1 : 0;
    }
    __syncthreads();

    // process spheres in sorted order (descending overlap count)
    for (int i = 0; i < num_spheres; i++) {
        int sphere_idx = sorted_indices[batch_idx * num_spheres + i];

        if (shared_retain[sphere_idx] == 0) continue;

        // threads cooperatively mark neighbors for removal
        for (int j = tid; j < num_neighbors; j += blockDim.x) {
            int neighbor = idx[batch_idx * num_spheres * num_neighbors + sphere_idx * num_neighbors + j];
            if (neighbor != sphere_idx && neighbor != ignore_idx) {
                shared_retain[neighbor] = 0;
            }
        }
        __syncthreads();
    }

    // write back to global memory
    for (int i = tid; i < num_spheres; i += blockDim.x) {
        retain[batch_idx * num_spheres + i] = shared_retain[i];
    }
}


// Large kernel for num_spheres > 4096.
// One block per batch, uses global memory with atomics.
__global__ void greedy_reduction_padded_large_kernel(
    const int* __restrict__ sorted_indices,  // (B, P)
    const int* __restrict__ idx,             // (B, P, K)
    const int* __restrict__ lengths,         // (B,)
    int* __restrict__ retain,                // (B, P)
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= num_batches) return;

    int valid_length = lengths[batch_idx];

    // init retain array in global memory
    for (int i = tid; i < num_spheres; i += blockDim.x) {
        retain[batch_idx * num_spheres + i] = (i < valid_length) ? 1 : 0;
    }
    __syncthreads();

    // process spheres in sorted order
    for (int i = 0; i < num_spheres; i++) {
        int sphere_idx = sorted_indices[batch_idx * num_spheres + i];

        int should_process = retain[batch_idx * num_spheres + sphere_idx];
        __syncthreads();

        if (should_process == 0) continue;

        // threads cooperatively mark neighbors for removal
        for (int j = tid; j < num_neighbors; j += blockDim.x) {
            int neighbor = idx[batch_idx * num_spheres * num_neighbors + sphere_idx * num_neighbors + j];
            if (neighbor != sphere_idx && neighbor != ignore_idx) {
                atomicAnd(&retain[batch_idx * num_spheres + neighbor], 0);
            }
        }
        __syncthreads();
    }
}


void launch_greedy_reduction_padded_cuda(
    const int* sorted_indices,
    const int* idx,
    const int* lengths,
    int* retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
) {
    int threads = 256;
    int blocks = num_batches;

    if (num_spheres <= 4096) {
        int shared_mem_size = num_spheres * sizeof(int);
        greedy_reduction_padded_kernel<<<blocks, threads, shared_mem_size>>>(
            sorted_indices, idx, lengths, retain,
            num_batches, num_spheres, num_neighbors, ignore_idx
        );
    } else {
        greedy_reduction_padded_large_kernel<<<blocks, threads>>>(
            sorted_indices, idx, lengths, retain,
            num_batches, num_spheres, num_neighbors, ignore_idx
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Failed: %s\n", cudaGetErrorString(err));
    }
}

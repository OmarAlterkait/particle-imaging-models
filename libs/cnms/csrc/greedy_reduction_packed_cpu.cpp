// greedy_reduction_packed_cpu.cpp
// CPU implementation of packed CNMS greedy reduction
#include <torch/extension.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

void greedy_reduction_packed_cpu(
    const int* sorted_indices,
    const int* idx,
    const int* offset,
    int* retain,
    int num_batches,
    int total_points,
    int num_neighbors,
    int ignore_idx
) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int batch_start = (batch_idx == 0) ? 0 : offset[batch_idx - 1];
        int batch_end = offset[batch_idx];
        int batch_size = batch_end - batch_start;
        
        if (batch_size == 0) continue;
        
        for (int i = 0; i < batch_size; i++) {
            retain[batch_start + i] = 1;
        }
        
        for (int i = 0; i < batch_size; i++) {
            int global_idx = sorted_indices[batch_start + i];
            int local_idx = global_idx - batch_start;
            if (local_idx < 0 || local_idx >= batch_size) continue;
            
            if (retain[batch_start + local_idx] == 0) continue;
            
            for (int j = 0; j < num_neighbors; j++) {
                int neighbor_global = idx[global_idx * num_neighbors + j];
                if (neighbor_global != ignore_idx && neighbor_global != global_idx) {
                    int neighbor_local = neighbor_global - batch_start;
                    if (neighbor_local >= 0 && neighbor_local < batch_size) {
                        retain[batch_start + neighbor_local] = 0;
                    }
                }
            }
        }
    }
}

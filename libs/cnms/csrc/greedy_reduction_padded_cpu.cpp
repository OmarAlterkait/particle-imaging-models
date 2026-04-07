// greedy_reduction_padded_cpu.cpp
// CPU kernel for padded CNMS greedy reduction (B, P) layout.
// Ported from PoLAr-MAE/extensions/cnms with int retain.
#include <torch/extension.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

void greedy_reduction_padded_cpu(
    const int *sorted_indices,  // (B, P)
    const int *idx,             // (B, P, K)
    const int *lengths,         // (B,)
    int *retain,                // (B, P) output
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx)
{
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        int valid_length = lengths[batch_idx];

        for (int i = 0; i < num_spheres; ++i)
        {
            retain[batch_idx * num_spheres + i] = (i < valid_length) ? 1 : 0;
        }

        // process spheres in sorted order
        for (int i = 0; i < valid_length; ++i)
        {
            int sphere_idx = sorted_indices[batch_idx * num_spheres + i];

            if (retain[batch_idx * num_spheres + sphere_idx] == 0)
            {
                continue; // already removed
            }

            int base_offset = batch_idx * num_spheres * num_neighbors + sphere_idx * num_neighbors;

            for (int j = 0; j < num_neighbors; ++j)
            {
                int neighbor = idx[base_offset + j];
                if (neighbor != sphere_idx && neighbor != ignore_idx && neighbor < valid_length)
                {
                    retain[batch_idx * num_spheres + neighbor] = 0;
                }
            }
        }
    }
}

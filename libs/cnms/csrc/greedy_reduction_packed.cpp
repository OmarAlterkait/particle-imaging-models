// greedy_reduction_packed.cpp
// C++ bindings for packed CNMS greedy reduction
#include <torch/extension.h>
#include <vector>

// forward declarations
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
);

void greedy_reduction_packed_cpu(
    const int* sorted_indices,
    const int* idx,
    const int* offset,
    int* retain,
    int num_batches,
    int total_points,
    int num_neighbors,
    int ignore_idx
);

// Padded format forward declarations
void launch_greedy_reduction_padded_cuda(
    const int* sorted_indices,
    const int* idx,
    const int* lengths,
    int* retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
);

void greedy_reduction_padded_cpu(
    const int* sorted_indices,
    const int* idx,
    const int* lengths,
    int* retain,
    int num_batches,
    int num_spheres,
    int num_neighbors,
    int ignore_idx
);


torch::Tensor greedy_reduction_packed(
    torch::Tensor sorted_indices,  // (N,) sorted point indices
    torch::Tensor idx,             // (N, K) neighbor indices
    torch::Tensor offset,          // (B,) cumulative counts
    int ignore_idx
) {
    TORCH_CHECK(sorted_indices.dim() == 1, "sorted_indices must be 1D (packed format)");
    TORCH_CHECK(idx.dim() == 2, "idx must be 2D: (N, K)");
    TORCH_CHECK(offset.dim() == 1, "offset must be 1D");
    
    int total_points = sorted_indices.size(0);
    int num_neighbors = idx.size(1);
    int num_batches = offset.size(0);
    
    TORCH_CHECK(idx.size(0) == total_points, "idx and sorted_indices must have same N");
    
    bool is_cuda = sorted_indices.is_cuda();
    TORCH_CHECK(idx.device() == sorted_indices.device(), "tensors must be on same device");
    TORCH_CHECK(offset.device() == sorted_indices.device(), "tensors must be on same device");
    
    // ensure int32
    if (sorted_indices.dtype() == torch::kInt64) {
        sorted_indices = sorted_indices.to(torch::kInt32).contiguous();
    }
    if (idx.dtype() == torch::kInt64) {
        idx = idx.to(torch::kInt32).contiguous();
    }
    if (offset.dtype() == torch::kInt64) {
        offset = offset.to(torch::kInt32).contiguous();
    }
    
    sorted_indices = sorted_indices.contiguous();
    idx = idx.contiguous();
    offset = offset.contiguous();
    
    // compute max batch size for shared memory allocation
    auto offset_cpu = offset.cpu();
    int max_batch_size = 0;
    int prev = 0;
    for (int i = 0; i < num_batches; i++) {
        int curr = offset_cpu[i].item<int>();
        int batch_size = curr - prev;
        if (batch_size > max_batch_size) max_batch_size = batch_size;
        prev = curr;
    }
    
    // use int32 for retain (CUDA kernel needs 4-byte alignment for atomics)
    auto retain_int = torch::zeros({total_points}, torch::dtype(torch::kInt32).device(sorted_indices.device()));
    
    if (is_cuda) {
        launch_greedy_reduction_packed_cuda(
            sorted_indices.data_ptr<int>(),
            idx.data_ptr<int>(),
            offset.data_ptr<int>(),
            retain_int.data_ptr<int>(),
            num_batches,
            total_points,
            num_neighbors,
            ignore_idx,
            max_batch_size
        );
    } else {
        greedy_reduction_packed_cpu(
            sorted_indices.data_ptr<int>(),
            idx.data_ptr<int>(),
            offset.data_ptr<int>(),
            retain_int.data_ptr<int>(),
            num_batches,
            total_points,
            num_neighbors,
            ignore_idx
        );
    }
    
    return retain_int.to(torch::kBool);
}


torch::Tensor greedy_reduction_padded(
    torch::Tensor sorted_indices,  // (B, P)
    torch::Tensor idx,             // (B, P, K)
    torch::Tensor lengths,         // (B,)
    int ignore_idx
) {
    TORCH_CHECK(sorted_indices.dim() == 2, "sorted_indices must be 2D (B, P)");
    TORCH_CHECK(idx.dim() == 3, "idx must be 3D: (B, P, K)");
    TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D");

    int num_batches = sorted_indices.size(0);
    int num_spheres = sorted_indices.size(1);
    int num_neighbors = idx.size(2);

    TORCH_CHECK(idx.size(0) == num_batches, "batch dimension mismatch");
    TORCH_CHECK(idx.size(1) == num_spheres, "sphere dimension mismatch");
    TORCH_CHECK(lengths.size(0) == num_batches, "lengths batch dimension mismatch");

    bool is_cuda = sorted_indices.is_cuda();
    TORCH_CHECK(idx.device() == sorted_indices.device(), "tensors must be on same device");
    TORCH_CHECK(lengths.device() == sorted_indices.device(), "tensors must be on same device");

    // ensure int32
    if (sorted_indices.dtype() == torch::kInt64)
        sorted_indices = sorted_indices.to(torch::kInt32).contiguous();
    if (idx.dtype() == torch::kInt64)
        idx = idx.to(torch::kInt32).contiguous();
    if (lengths.dtype() == torch::kInt64)
        lengths = lengths.to(torch::kInt32).contiguous();

    sorted_indices = sorted_indices.contiguous();
    idx = idx.contiguous();
    lengths = lengths.contiguous();

    // int32 retain for proper atomic alignment, convert to bool at end
    auto retain_int = torch::zeros({num_batches, num_spheres},
        torch::dtype(torch::kInt32).device(sorted_indices.device()));

    if (is_cuda) {
        launch_greedy_reduction_padded_cuda(
            sorted_indices.data_ptr<int>(),
            idx.data_ptr<int>(),
            lengths.data_ptr<int>(),
            retain_int.data_ptr<int>(),
            num_batches, num_spheres, num_neighbors, ignore_idx
        );
    } else {
        greedy_reduction_padded_cpu(
            sorted_indices.data_ptr<int>(),
            idx.data_ptr<int>(),
            lengths.data_ptr<int>(),
            retain_int.data_ptr<int>(),
            num_batches, num_spheres, num_neighbors, ignore_idx
        );
    }

    return retain_int.to(torch::kBool);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("greedy_reduction_packed", &greedy_reduction_packed,
          "Greedy reduction for packed tensor format (CPU and CUDA)");
    m.def("greedy_reduction_padded", &greedy_reduction_padded,
          "Greedy reduction for padded tensor format (CPU and CUDA)");
}

#include "flux.cuh"
#include <vector>
#include <cassert>

void __global__ testFluxKernel(float3 r0, float3 rf, double* grid, uint gridSize, double voxelSize) {
    // We only need one thread to test a single ray's logic
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        flux(r0, rf, grid, gridSize, voxelSize);
    }
}

// This handles all memory allocation, zeroing, kernel execution, and cleanup.
std::vector<double> runFluxOnGPU(float3 r0, float3 rf, uint gridSize, double voxelSize) {
    size_t num_voxels = gridSize * gridSize * gridSize;
    size_t byte_size = num_voxels * sizeof(double);
    
    double* d_grid;
    cudaMalloc(&d_grid, byte_size);
    // CRITICAL: Zero out the memory before atomicAdd!
    cudaMemset(d_grid, 0, byte_size); 

    // Launch kernel with 1 block, 1 thread
    testFluxKernel<<<1, 1>>>(r0, rf, d_grid, gridSize, voxelSize);
    cudaDeviceSynchronize();

    // Fetch results
    std::vector<double> h_grid(num_voxels);
    cudaMemcpy(h_grid.data(), d_grid, byte_size, cudaMemcpyDeviceToHost);

    cudaFree(d_grid);
    return h_grid;
}

void testSingleVoxelContainment() {
    float3 r0 = make_float3(0.2f, 0.5f, 0.5f);
    float3 rf = make_float3(0.5f, 0.5f, 0.5f);
    
    std::vector<double> grid = runFluxOnGPU(r0, rf, 1, 1.0);

    assert(grid[0] - length(rf - r0) < 1e-6);
}

int main() {
    testSingleVoxelContainment();
    return 0;
}
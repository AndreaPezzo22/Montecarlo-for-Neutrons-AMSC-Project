#ifndef KERNELS_H
#define KERNELS_H

#include "types.cuh"
#include <cuda_runtime.h>

void loadMaterialsToGPU(Material* h_mats, int count);
void loadRegionsToGPU(Region* h_rregions, int count);

__global__ void traverse(float* posX, float* posY, float* posZ, float* dirX, float* dirY, float* dirZ, int* outMatID, int numParticles);

#endif
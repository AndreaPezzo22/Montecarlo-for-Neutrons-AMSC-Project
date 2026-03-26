// Il motore Monte Carlo: i kernel __global__ (initParticles, traverse)

# include "types.cuh"

// Memoria costante per la GPU
__constant__ Material d_materials[10];
__constant__ Region d_regions[20]; // 
__constant__ int d_num_regions; // Quante scataole stiamo definendo


# include "geometry.cuh"
# include <curand_kernel.h>

// Funzione "ponte" per caricare i dati dalla CPU alla memoria costante della GPU
void loadMaterialsToGPU(Material* h_mats, int count) {
    cudaMemcpyToSymbol(d_materials, h_mats, count * sizeof(Material));
}

void loadRegionsToGPU(Region* h_region, int count) {
	cudaMemcpyToSymbol(d_regions, h_region, count * sizeof(Region));
	cudaMemcpyToSymbol(d_num_regions, &count, sizeof(int));
}

__global__ void traverse (float* d_posX, float* d_posY, float* d_posZ,
			  float* d_dirX, float* d_dirY, float* d_dirZ,
			  int* d_outMatID,
			  int numParticles){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numParticles) return;

	// Leggiamo la posizione della particella
	float x = d_posX[id];
	float y = d_posY[id];
	float z = d_posZ[id];
	
	// Calcolo ID materiale
	int matID = getMaterialID(x, y, z);

	// Lettura sezioni d'urto corrispondenti a ID materiale
	float sigma_t = d_materials[matID].sigma_t;

	d_outMatID[id] = matID;
}

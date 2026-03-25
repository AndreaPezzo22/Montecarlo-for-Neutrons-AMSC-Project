// Il motore Monte Carlo: i kernel __global__ (initParticles, traverse)

# include "types.cuh"
# include "geometry.cuh"
# include <curand_kernel.h>


// Dichiariamo QUI la memoria costante per la GPU
__constant__ Material d_materials[10];

// Funzione "ponte" per caricare i dati dalla CPU alla memoria costante della GPU
void loadMaterialsToGPU(Material* h_mats, int count) {
    cudaMemcpyToSymbol(d_materials, h_mats, count * sizeof(Material));
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

#include <iostream>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>
#include "../include/types.cuh"

// Dichiariamo le funzioni presenti in kernel.cu
void loadMaterialsToGPU(Material* h_mats, int count);
__global__ void traverse(float* posX, float* posY, float* posZ, float* dirX, float* dirY, float* dirZ, int* outMatID, int numParticles);

int main() {
    int numParticles = 1;

    // 1. Prepariamo e carichiamo i materiali sulla GPU tramite la funzione ponte
    Material h_materials[3];
    h_materials[0] = {0.1f, 0.05f, 0.15f}; // Acqua (ID 0)
    h_materials[1] = {0.2f, 0.8f, 1.00f};  // Uranio (ID 1)
    h_materials[2] = {0.0f, 0.0f, 0.00f};  // Vuoto (ID 2)
    loadMaterialsToGPU(h_materials, 3);

    // 2. Creiamo i dati sulla CPU (Host)
    float h_x, h_y, h_z;
    float h_dx = 0, h_dy = 0, h_dz = 0; // Direzioni vuote per ora
    int h_outMatID = -1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

    h_x = dist(gen);
    h_y = dist(gen);
    h_z = dist(gen);

    std::cout << "Coordinate generate: X=" << h_x << ", Y=" << h_y << ", Z=" << h_z << "\n";

    // 3. Allochiamo memoria fisica REALE sulla GPU
    float *d_x, *d_y, *d_z, *d_dx, *d_dy, *d_dz;
    int *d_outMatID;
    cudaMalloc(&d_x, sizeof(float));
    cudaMalloc(&d_y, sizeof(float));
    cudaMalloc(&d_z, sizeof(float));
    cudaMalloc(&d_dx, sizeof(float));
    cudaMalloc(&d_dy, sizeof(float));
    cudaMalloc(&d_dz, sizeof(float));
    cudaMalloc(&d_outMatID, sizeof(int));

    // 4. Copiamo la posizione dalla CPU alla GPU
    cudaMemcpy(d_x, &h_x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &h_y, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, &h_z, sizeof(float), cudaMemcpyHostToDevice);

    // 5. Lanciamo il kernel
    traverse<<<1, 1>>>(d_x, d_y, d_z, d_dx, d_dy, d_dz, d_outMatID, numParticles);
    cudaDeviceSynchronize();

    // 6. Riportiamo il risultato dalla GPU alla CPU
    cudaMemcpy(&h_outMatID, d_outMatID, sizeof(int), cudaMemcpyDeviceToHost);

    // 7. Stampiamo il risultato
    std::cout << "Materiale rilevato dalla GPU (ID): " << h_outMatID << "\n";
    if (h_outMatID == 0) std::cout << "-> Siamo nell'Acqua!\n";
    else if (h_outMatID == 1) std::cout << "-> Siamo nell'Uranio!\n";
    else std::cout << "-> Siamo nel Vuoto!\n";

    // Pulizia
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_dz); cudaFree(d_outMatID);

    return 0;
}

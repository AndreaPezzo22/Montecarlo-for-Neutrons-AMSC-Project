// Structs (Material, Particle) e memoria __constant__

#pragma once

struct Material {
	float sigma_s;
	float sigma_a;
	float sigma_t;
};

// Aggiunto Struct Region per definire le posizioni dei materiali 
struct Region {
	int min_ix, max_ix;
	int min_iy, max_iy;
	int min_iz, max_iz;
	int mat_id;
};

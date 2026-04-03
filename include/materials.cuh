#ifndef MATERIALS_H
#define MATERIALS_H

extern __constant__ Region c_regions[]; 
extern __constant__ int c_num_regions; 

__device__ inline int getMaterialID(float x, float y, float z) {
	// Calcolo indici del cubo ??
	// TODO: togliere gli indici, usare solo coordinate
	int ix = floor(x / 10.0f);
	int iy = floor(y / 10.0f);
	int iz = floor(z / 10.0f);

	int final_mat_id = 2;
	
	for ( int i=0; i<c_num_regions; i++) {
		Region r = c_regions[i];
		int inside = (ix >= r.min_ix) * (ix <= r.max_ix) *
			     (iy >= r.min_iy) * (iy <= r.max_iy) *
			     (iz >= r.min_iz) * (iz <= r.max_iz);
		final_mat_id = (inside * r.mat_id) + ((1 - inside) * final_mat_id);
	}
	return final_mat_id;
}
	
#endif
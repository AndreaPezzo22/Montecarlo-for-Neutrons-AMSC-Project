// Mathematical Functions for Constructive Solid Geometry -> getMaterialID, distanceToBoundary

__device__ inline int getMaterialID(float x, float y, float z) {

	// Calcolo indici del cubo
	int ix = floor(x / 10.0f);
	int iy = floor(y / 10.0f);
	int iz = floor(z / 10.0f);
	
	// La particella è definita dalle coordinate ix, iy, ix.
	// I materiali in questo caso sono acqua, uranio e vuoto.
	// L uranio è un cubo definito da coordinate (-1 1, -1 1, -1 1)
	// L acqua invece circonda l uranio e cofina con il vuoto
	// Dove il vuoto ha coordinanate che partono da (-5 5, -5 5, -5 5) in poi

	// Per verificare se la particella si trova nel materiale utiliziamo le operazioni booleane.

	// Verifichiamo se si trova dentro uranio. Alla fine dell operazione se la condizione si avvera, in_core ha valore 1
	int in_core = (ix >= -1) * (ix <=1) * (iy >= -1) * (iy <=1) * (iz >= -1) * (iz <= 1); 
	// Facciamo lo stesso per gli matriali
	int in_reactor = (ix >= -5) * (ix <= 5) * (iy >= -5) * (iy <= 5) * (iz >= -5) * (iz <= 5); 
	int in_water = in_reactor * (1 - in_core);
	int in_void = 1 - in_reactor;
	
	// A questo punto abbiamo calcolato i risultati e vogliamo utilizzarli per fare un return dell indice del materiale corrispondente.
	// Dove Acqua ha indice 0, Uranio 1 e Vuoto 2

	return (in_water * 0) + (in_core * 1) + (in_void * 2);

}

// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


__global__ void d_zero_particle_forces(float* d_f, int ns, int Dim) {
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind < ns) {
		for (int j = 0; j < Dim; j++)
			d_f[ind * Dim + j] = 0.0f;
	}
}




// Calculates dr = r1 - r2
__device__ float d_pbc_mdr2(float* r1, float* r2, float* dr,
	float* bl, float* hbl, int Dim) {

	float mdr2 = 0.0f;
	for (int i = 0; i < Dim; i++) {
		dr[i] = r1[i] - r2[i];

		if (dr[i] > hbl[i]) dr[i] -= bl[i];
		else if (dr[i] < -hbl[i]) dr[i] += bl[i];

		mdr2 += dr[i] * dr[i];
	}
	return mdr2;
}
// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <curand_kernel.h>
#include <curand.h>

// M threads per block
// Block is local in memory?
// NThreads is number of opeerations in the loop each block does
__global__ void cu_random_posits(float *x, float *L, 
	int size, int Dim, curandState *d_states) {

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	curandState l_state; 
	l_state = d_states[ind];

	if (ind < size)
		for (int j = 0; j < Dim; j++)
			x[ind * Dim + j] = curand_uniform(&l_state) * L[j];
	
	d_states[ind] = l_state;
}


__global__ void init_dev_rng(unsigned int seed, curandState* d_states, int ns) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;//check index for >= ns

	if (idx >= ns)//this probably will not compile
		return;

	curand_init(seed, idx, 0, &d_states[idx]);

}
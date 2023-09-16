// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "integrator_EM.h"

EM::~EM(){}

EM::EM(std::istringstream& iss) : Integrator(iss) {
    
}

void EM::Integrate_2(){

    d_EM_integrator<<<group->GRID, group->BLOCK>>>(d_x, d_f,
        delt, noise_mag, d_L, d_Lh, group->d_index.data(),
        group->nsites, Dim, d_states);

}

__global__ void d_EM_integrator(
	float* x, float* f,				// Particle positions, forces
	float delt, float noise_mag,	// Time step size, magnitude of noise
	float *L, float *Lh,			// Box length, half box length
	thrust::device_ptr<int> d_index,					// List of sites in group integrated
	int ns,							// Number of sites in the list
	int Dim,						// Dimensionality of the system
	curandState* d_states ) 
{

	int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (list_ind >= ns)
		return;

	int ind = d_index[list_ind];

	curandState l_state;

	l_state = d_states[ind];

	for (int j = 0; j < Dim; j++) {
		x[ind * Dim + j] = x[ind * Dim + j]
			+ delt * f[ind * Dim + j] +noise_mag * curand_normal(&l_state);

		if (x[ind * Dim + j] > L[j])
			x[ind * Dim + j] -= L[j];
		else if (x[ind * Dim + j] < 0.0)
			x[ind * Dim + j] += L[j];

	}

	d_states[ind] = l_state;
}
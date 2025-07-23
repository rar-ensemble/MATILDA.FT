// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "integrator.h"
#include <curand_kernel.h>
#include <curand.h>
#include <sstream>

__global__ void d_EM_integrator(
	float* x, float* f,				// Particle positions, forces
	float delt, float noise_mag,	// Time step size, magnitude of noise
	float *L, float *Lh,			// Box length, half box length
	thrust::device_ptr<int> d_index,					// List of sites in group integrated
	int ns,							// Number of sites in the list
	int Dim,						// Dimensionality of the system
	curandState* d_states ); 

#ifndef _INTEGRATOR_EM_
#define _INTEGRATOR_EM_

class EM : public Integrator {
public:
    EM(std::istringstream&);
    ~EM();
    void Integrate_2() override;
};

#endif

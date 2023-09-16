// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "integrator.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <sstream>


__global__ void d_GJF_integrator(float* x, float* xo, 
	float* f, float* old_noise,
	float *mass, float *diff, int *typ, float delt, float noise_mag, 
	float* L, float* Lh, 
	thrust::device_ptr<int> d_index,
	int ns, 
	int Dim, 
	curandState* d_states);

#ifndef _INTEGRATOR_GJF_
#define _INTEGRATOR_GJF_

class GJF : public Integrator {
public:
    GJF(std::istringstream&);
    ~GJF();
    void Integrate_2() override;
};

#endif

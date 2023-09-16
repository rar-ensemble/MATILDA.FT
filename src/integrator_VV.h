// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "integrator.h"
#include <sstream>
#include <curand_kernel.h>
#include <curand.h>

__global__ void d_VV_integrator_1(float* x, float* f, float*v, int *typ, float *mass, float delt, 
	float *L, float *Lh, thrust::device_ptr<int> d_index, int ns, int D, float v_max, int step, curandState *d_states, int dist,
    thrust::device_ptr<float> d_TDISP);

__global__ void d_VV_integrator_2(float* x, float* f, float*v, int *typ, float *mass, float delt, 
	float *L, float *Lh, thrust::device_ptr<int> d_index, int ns, int D );




#ifndef _INTEGRATOR_VV_
#define _INTEGRATOR_VV_

class VV : public Integrator {
public:
    float v_max;
    int dist;
    VV(std::istringstream&);
    ~VV();
    void Integrate_1() override;
    void Integrate_2() override;
    thrust::device_vector<float> d_TDISP;
};

#endif


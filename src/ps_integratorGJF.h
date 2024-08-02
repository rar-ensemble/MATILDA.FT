// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_integrator.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <sstream>


__global__ void d_GJF_integrator(float*, float*, float*, const float*, const float*, const float*,
const int*, const float, const float, const float*, const int*, const int, const int, curandState*);


#ifndef _INTEGRATOR_GJF_
#define _INTEGRATOR_GJF_

class PS_Box;

class GJF : public Integrator {
public:
    GJF(std::istringstream&, PS_Box*);
    ~GJF();
    void Integrate_2() override;

	thrust::device_vector<float> d_xOld; 		// [nsites*Dim] dev vector to store prev positions
	float* _d_xOld;								// nsites taken from the group integrated

	thrust::device_vector<float> d_noiseOld;	// [nsites*Dim] stores previous noise
	float* _d_noiseOld;
};

#endif

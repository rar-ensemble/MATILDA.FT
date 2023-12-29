// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "include_libs.h"

__global__ void d_makeDoubleNoise(
    cuDoubleComplex* noiseField,
    curandState* d_states,
    const double noiseMag,
    const int MAX
) {

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= MAX)
		return;

    curandState l_state = d_states[ind];

    noiseField[ind].x = noiseMag * curand_normal(&l_state);
    noiseField[ind].y = 0.0;

}
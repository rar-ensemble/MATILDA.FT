// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Extraforce.h"

__global__ void d_ExtraForce_Langevin(
    float* f,               // [ns*Dim], particle forces
    const float* v,         // [ns*Dim], particle velocities
    const float noise_mag,  // magnitude of the noise, should be sqrt(2.*gamma)
    const float gamma,      // Friction force
    thrust::device_ptr<int> d_index,   // List of sites in the group
    const int ns,           // Number of sites in the list
    const int D,            // Dimensionality of the simulation
    curandState* d_states); // Status of CUDA rng

#ifndef _EXTRAFORCE_LANGEVIN
#define _EXTRAFORCE_LANGEVIN


class Langevin : public ExtraForce {
protected:
    float drag;         // drag coefficient 
    float lang_noise;  // Langevin noise
public:
	~Langevin();
	Langevin(std::istringstream&);
    void AddExtraForce() override;
    void UpdateVirial(void) override;
};

#endif
// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <curand_kernel.h>
#include <curand.h>
#include "Extraforce_langevin.h"
#include "globals.h"

using namespace std;

Langevin::~Langevin(){}
void Langevin::UpdateVirial() {return;};

Langevin::Langevin(istringstream& iss) : ExtraForce(iss) {
    readRequiredParameter(iss, drag);

    // Noise in Langevin equation is sqrtf( 2.0 * gamma * delt);
    // We must divide by delt here b/c the VV algo multiplies f by delt
    lang_noise = sqrtf(drag) * noise_mag / delt;
}

void Langevin::AddExtraForce() {

    d_ExtraForce_Langevin<<<group->GRID, group->BLOCK>>>(d_f, d_v, lang_noise,
        drag, group->d_index.data(), group->nsites,
        Dim, d_states);


}

// Adds Langevin stochastic and friction forces to 
// allow constant T simulations with VV algorithm
__global__ void d_ExtraForce_Langevin(
    float* f,               // [ns*Dim], particle forces
    const float* v,         // [ns*Dim], particle velocities
    const float noise_mag,  // magnitude of the noise, should be sqrt(2.*gamma)
    const float gamma,      // Friction force
    thrust::device_ptr<int> d_index,   // List of sites in the group
    const int ns,           // Number of sites in the list
    const int D,            // Dimensionality of the simulation
    curandState* d_states) {// Status of CUDA rng

    int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns)
        return;

    int ind = d_index[list_ind];

    curandState l_state;

    l_state = d_states[ind];

    for (int j = 0; j < D; j++)
        f[ind * D + j] += -gamma * v[ind * D + j] + noise_mag * curand_normal(&l_state);

    d_states[ind] = l_state;

}

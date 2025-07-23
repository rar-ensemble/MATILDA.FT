// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <curand_kernel.h>
#include <curand.h>
#include "Extraforce_midpush.h"
#include "globals.h"

using namespace std;

Midpush::~Midpush(){return;}

void Midpush::UpdateVirial(void){return;};

Midpush::Midpush(istringstream& iss) : ExtraForce(iss){
    readRequiredParameter(iss, axis);
    readRequiredParameter(iss, force_magnitude);
}

void Midpush::AddExtraForce() {

    d_ExtraForce_midpush<<<group->GRID, group->BLOCK>>>(d_f, d_x, 
        force_magnitude, group->d_index.data(), d_Lh, group->nsites, axis, Dim);

}

// Pushes particles towards the inside of the simulation box along the specified axis
__global__ void d_ExtraForce_midpush(
    float* f,               // [ns*Dim], particle forces
    const float* x,         // [ns*Dim], particle positions
    const float fmag,       // Force magnitude
    thrust::device_ptr<int> d_index,   // List of sites in the group
    const float* Lh,        // [Dim] Half box length
    const int ns,
    const int axis,           // Number of sites in the list
    const int D) {            // Dimensionality of the simulation
    

    int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns)
        return;

    int ind = d_index[list_ind];


    int pi = ind * D + axis;

    if ( x[pi] > Lh[axis] ) 
        f[pi] -= fmag;
    else
        f[pi] += fmag;

}

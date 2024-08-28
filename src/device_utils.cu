// Copyright (c) 2024 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include <curand_kernel.h>
#include <curand.h>

// Assigns a device float array value of 'val'
__global__ void d_assignFloatVal(
    float* f,               // [N] array to be modified
    const float val,        // value to be placed in array
    const int N             // array dimension
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    f[id] = val;
}


// Intended to be called from the group class to fill the group's 
// density field. 
__global__ void d_fillDensityGrid(
    float* rho,             // [M] density field
    const int* sites,       // [ns] indices of particles in the group
    const int* gridInds,    // [ns*gridPerPartic] indices of grids for each partic
    const float* gridW,     // [ns*gridPerPartic] weights for each grid point
    const int gridPerPartic,// Number of grid points per particle
    const int ns            // number of sites in this group
) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    int pind = sites[id];

    for ( int i=0 ; i<gridPerPartic; i++ ) {
        // index \in [0, ns*gridPerPartic)
        int index = pind * gridPerPartic + i;

        // index \in [0,M)
        int gind = gridInds[index];
        
        // Weight of the particle to gind
        float W3 = gridW[index];

        // Accumulate the grid density
        atomicAdd(&rho[gind], W3);
    }// i=0:gridPerPartic

}

__global__ void d_initDeviceRNG(unsigned int seed, curandState* states, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;//check index for >= ns

    if (idx >= N)//this probably will not compile
        return;

    curand_init(seed, idx, 0, &states[idx]);
}

// computes dr = ri - rj corrected for PBCs
__device__ float d_pbc_dr2f(
    float* dr,                          // [Dim] vector from i to j
    const float* ri,                    // [Dim] position for i
    const float* rj,                    // [Dim] position for j
    const float* bx, const float* bxh,  // [Dim] box dimensions
    const int dim                       // system dimensionality
    ) {

    float mdr2 = 0.0;
    for ( int n=0 ; n<dim ; n++ ) {
        dr[n] = ri[n] - rj[n];

        if ( dr[n] > bxh[n] ) dr[n] -= bx[n];
        else if ( dr[n] < -bxh[n] ) dr[n] += bx[n];

        mdr2 += dr[n] * dr[n];
    }

    return mdr2;

}

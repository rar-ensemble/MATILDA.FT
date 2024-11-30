// Copyright (c) 2024 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include <curand_kernel.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>



// Computes f *= val for device float array
__global__ void d_multiplyByFloat(
    float* f,         // [N] array to be multiplied
    const float val,    // value of inp to be multiplied by
    const int N
    ) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    f[id] *= val;
}


// Computes f *= val for device cuComplex array
__global__ void d_multiplyCuComplexByFloat(
    cuComplex* f,         // [N] array to be multiplied
    const float val,    // value of inp to be multiplied by
    const int N
    ) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    f[id].x *= val;
    f[id].y *= val;
}

// computes out += in for floats
__global__ void d_floatPlusEqFloat(
    float* out,
    const float* in,
    const int N
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id] += in[id];
}


// Computes out = Real(in)
__global__ void d_cpxToFloat(
    float* out,             // [N] array to be filled
    const cuComplex* in,    // [N] source array
    const int N             // array dimension
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id] = in[id].x;
}

// Computes real(out) = in, imag(out) = 0.0
__global__ void d_floatToCpx(
    cuComplex *out,         // [N] array to be filled
    const float *in,        // [N] source data
    const int N
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id].x = in[id];
    out[id].y = 0.0f;
}

// assumes fk is a vector function
// extracts the 'dir' component of the vector, 
// performs complex multiplication with rh
__global__ void d_multiplyCpxDirByCpx(
    cuComplex* out,         // [N] array to be filled
    const cuComplex *fk,    // [Dim*N] extracting directional component from f
    const cuComplex *rh,    // [N] array be multiplied by dir comp of fk
    const int dir,          // \in [0, Dim), direction to be extracted
    const int Dim,          // System dimensionality
    const int N             // Array dimension
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id].x = fk[id*Dim + dir].x * rh[id].x - fk[id*Dim + dir].y * rh[id].y;
    out[id].y = fk[id*Dim + dir].y * rh[id].x + fk[id*Dim + dir].x * rh[id].y;
}



// assumes f is a vector function
// extracts the 'dir' component of the vector into 
// field variable out in prep for Fourier transform
__global__ void d_extractCpxDirToCpx(
    cuComplex* out,     // [N] array to be filled
    const cuComplex *f, // [Dim*N] extracting directional component from f
    const int dir,      // \in [0, Dim), direction to be extracted
    const int Dim,      // System dimensionality
    const int N         // Array dimension
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id].x = f[id*Dim + dir].x;
    out[id].y = f[id*Dim + dir].y;
}


// Inverse of above routine. Takes 'in' field, which is assumed to be
// a real-valued component of a vector field, and places it in container
// 'out', which stores the Dim*N vector field
__global__ void d_cpxToFloatVecComponent(
    float* out,             // [Dim*N] storage array
    const cuComplex* in,    // [N] array containing the vector component
    const int dir,          // direction to insert to
    const int Dim,          // dimensionality
    const int N             // Array dimension
){
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id*Dim + dir] = in[id].x;
}



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


// Typically called from group class, accumulates grid forces
// into the particle force array
__global__ void d_mapGridForcesToPartics(
    float *f,               // [Dim*nstot] particle forces
    const float* gridF,     // [Dim*M] grid forces
    const int* sites,       // [ns] indices of particles in the group
    const int* gridInds,    // [ns*gridPerPartic] indices of grids for each partic
    const float* gridW,     // [ns*gridPerPartic] weights for each grid point
    const float gvol,       // Grid volum
    const int gridPerPartic,// Number of grid points per particle
    const int Dim,          // system dimensionality
    const int ns            // number of sites in this group
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    // particle 'id' in this group is 'pind' in total site list
    int pind = sites[id];

    
    for ( int j=0 ; j<Dim ; j++ ) {

        // Accumulate force locally, then atomicAdd
        float floc = 0.0;

        for ( int i=0 ; i<gridPerPartic; i++ ) {
            // index \in [0, ns*gridPerPartic)
            int index = pind * gridPerPartic + i;

            // index \in [0,M)
            int gind = gridInds[index];

            // Weight of this grid point
            float W3 = gridW[index];

            // Accumulate the force
            floc += gridF[gind*Dim + j] * W3 * gvol;
        }// i=0:gridPerPartic

        // Atomic add the force to the particles
        atomicAdd(&f[pind*Dim+j], floc);
    }// j=0:Dim


}


// Initializes the CUDA rng
__global__ void d_initDeviceRNG(unsigned int seed, curandState* states, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;//check index for >= ns

    if (idx >= N)
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




// CUDA kernel for parallel reduction
// This subroutine generated by Claude.ai
__global__ void sumArrayKernel(float *input, float *output, int n) {
    extern __shared__ float sharedData[];

    // Each thread loads one element into shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // Write the result of this block to the output array
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// Copyright (c) 2024 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include <curand_kernel.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>


// Used for ramping/changing interaction potentials during a simulation
__global__ void d_scale_potentials_by_scalar(
    cuComplex* uk,          // [M] pair potential in k-space
    cuComplex* fk,          // [M*Dim] force in k-space
    const float lambda,     // Scaling factor
    const int Dim,          // System dimension
    const int M             // Number of grid points
    ) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= M)
        return;

    // Scale the potential
    uk[id].x *= lambda;
    uk[id].y *= lambda;

    // Scale its gradient
    for (int j=0 ; j<Dim ; j++ ) {
        fk[id*Dim+j].x *= lambda;
        fk[id*Dim+j].y *= lambda;
    }

}


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


// computes out += in for floats
__global__ void d_floatVecPlusEqFloatComp(
    float* out,         // [Dim*N] array to store vector forces
    const float* in,    // [N] array containing component of force
    const int dir,      // direction in out to be accumulated
    const int Dim,      // system dimensionality
    const int N         // size of the grid
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id*Dim+dir] += in[id];
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


// performs complex multiplication with rh
__global__ void d_multiplyCpxByCpx(
    cuComplex* out,         // [N] array to be filled
    const cuComplex *uk,    // [N] extracting directional component from f
    const cuComplex *rh,    // [N] array be multiplied by dir comp of fk
    const int N             // Array dimension
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id].x = uk[id].x * rh[id].x - uk[id].y * rh[id].y;
    out[id].y = uk[id].y * rh[id].x + uk[id].x * rh[id].y;
}


// out = c1 * c2^\ast
__global__ void d_multiplyCpxByCpxConj(
    float* out,         // [N] array to be filled
    const cuComplex *c1,    // [N] extracting directional component from f
    const cuComplex *c2,    // [N] array be multiplied by dir comp of fk
    const int N             // Array dimension
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id] = c1[id].x * c2[id].x  + c1[id].y * c2[id].y;

}



// performs complex multiplication with rh
__global__ void d_multiplyDoubleCpxByCpx(
    cuDoubleComplex* out,         // [N] array to be filled
    const cuDoubleComplex *uk,    // [N] extracting directional component from f
    const cuDoubleComplex *rh,    // [N] array be multiplied by dir comp of fk
    const int N             // Array dimension
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id].x = uk[id].x * rh[id].x - uk[id].y * rh[id].y;
    out[id].y = uk[id].y * rh[id].x + uk[id].x * rh[id].y;
}



// performs complex multiplication with rh and by constant A
__global__ void d_multiplyDoubleCpxByCpxByCpxScalar(
    cuDoubleComplex* out,         // [N] array to be filled
    const cuDoubleComplex *uk,    // [N] extracting directional component from f
    const cuDoubleComplex *rh,    // [N] array be multiplied by dir comp of fk
    const cuDoubleComplex A,      // Scalar
    const int N             // Array dimension
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    double pre = uk[id].x * rh[id].x - uk[id].y * rh[id].y;
    double pim = uk[id].y * rh[id].x + uk[id].x * rh[id].y;

    out[id].x = A.x * pre - A.y * pim;
    out[id].y = A.y * pre + A.x * pim;
}


// assumes fk is a vector function
// performs complex multiplication 
__global__ void d_multiplyFloatByFloat(
    float *out,         // [N] array to be filled
    const float *uk,    // [N] extracting directional component from f
    const float *rh,    // [N] array be multiplied by dir comp of fk
    const int N             // Array dimension
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;

    out[id] = uk[id] * rh[id];
    
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


// Intended to be called from the group class to fill the group's 
// density field. 
__global__ void d_fillChargeDensityGrid(
    float* rho,             // [M] density field
    float* rhoq,            // [M] density field
    const int* sites,       // [ns] indices of particles in the group
    const float* q,         // [nstot] charges of all particles
    const int* gridInds,    // [ns*gridPerPartic] indices of grids for each partic
    const float* gridW,     // [ns*gridPerPartic] weights for each grid point
    const int gridPerPartic,// Number of grid points per particle
    const int ns            // number of sites in this group
) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    int pind = sites[id];

    // get charge for this particle
    float qp = q[pind];

    for ( int i=0 ; i<gridPerPartic; i++ ) {
        // index \in [0, ns*gridPerPartic)
        int index = pind * gridPerPartic + i;

        // index \in [0,M)
        int gind = gridInds[index];
        
        // Weight of the particle to gind
        float W3 = gridW[index];
        float qW3 = W3 * qp;

        // Accumulate the grid density
        atomicAdd(&rho[gind], W3);
        atomicAdd(&rhoq[gind], qW3);
    }// i=0:gridPerPartic

}



// Called from group class, accumulates grid forces
// into the particle force array
__global__ void d_mapGridChargeForcesToPartics(
    float *f,               // [Dim*nstot] particle forces
    const float* charges,   // [nstot] charges on each particle
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

    float qi = charges[pind];
    
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


        // gridF is the electric field pre-convolved with
        // unit Gaussian. Just needs to be x'd by q_i
        f[pind*Dim+j] += floc * qi;

    }// j=0:Dim


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


        // ditched atomic add bc each thread is a unique particle
        // assuming no duplicates in the group
        f[pind*Dim+j] += floc;


        // Atomic add the force to the particles
        // atomicAdd(&f[pind*Dim+j], floc);
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



// CUDA kernel for parallel reduction
// This subroutine generated by Claude.ai
__global__ void sumCpxDoubleArrayKernel(cuDoubleComplex *input, cuDoubleComplex *output, int n) {
    extern __shared__ cuDoubleComplex sharedCDData[];

    // Each thread loads one element into shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedCDData[tid].x = (i < n) ? input[i].x : 0.0;
    sharedCDData[tid].y = (i < n) ? input[i].y : 0.0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedCDData[tid].x += sharedCDData[tid + s].x;
            sharedCDData[tid].y += sharedCDData[tid + s].y;
        }
        __syncthreads();
    }

    // Write the result of this block to the output array
    if (tid == 0) {
        output[blockIdx.x].x = sharedCDData[0].x;
        output[blockIdx.x].y = sharedCDData[0].y;
    }
}
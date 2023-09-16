// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <cmath>
#include <cufft.h>
#include <cufftXt.h>
#include "device_utils.cuh"



// This contribution to th eforce comes from du/dr_i, the derivative of the
// Gaussian potential. This force only acts on the particle ``carrying'' the
// unit vector orientation.
__global__ void d_accumulateMSForce1(
    float* f,                       // [ns*Dim] Particle forces
    const int* upartner,            // [ns] index of th epartner particle
    const float* tensorField,       // [Dim*Dim*M] S tensor field convolved with [dir] componenot of (-mu/rho0*grad u)
    const float* ms_S,              // [Dim*Dim*ns] particle-level S tensors
    const float* grid_W,            // [ns*grid_per_partic] weights assoc'd with each grid
    const int*   grid_inds,         // [ns*grid_per_partic] indicies of the grid
    const float  gvol,              // grid volume
    const int    dir,               // component of this force
    const int    grid_per_partic,   // number of grid points this particle interacts with
    const int    ns,                // number of sites
    const int    Dim                // Dimensionality
){

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    if ( upartner[id] < 0 ) 
        return;

    // Add the force contribution from relevant grid points
    for ( int m=0 ; m<grid_per_partic ; m++ ) {
        
        int gind = id * grid_per_partic + m;

        int Mind = grid_inds[gind];
        float W3 = grid_W[gind];

        int pbase = id * Dim * Dim;
        float dotsum = 0.0f;
        for ( int j=0 ; j<Dim ; j++ ) {
            for ( int k=0 ; k<Dim ; k++ ) {
                // - sign below is because tensorField has (-mu/rho0) as prefactor
                // and since the potential itself is (-mu/rho0), the sign here needs to change
                dotsum += -tensorField[Mind*Dim*Dim + j*Dim + k] * ms_S[pbase + j*Dim + k];
            }
        }

        f[id*Dim + dir] += dotsum * W3 * gvol;

    }// m=0:grid_per_partic
}


// This contribution to the force comes from dS/dri, the derivative of the
// S tensor field w.r.t. the particle position. This acts on both particles
// involved in the definition of the unit vector u
__global__ void d_accumulateMSForce2(
    float* f,                       // [ns*Dim] Particle forces
    const float* x,                 // [ns*Dim] Particle positions
    const int* upartner,            // [ns] index of th epartner particle
    const float* tensorField,       // [Dim*Dim*M] S tensor field convolved with u
    const float* ms_u,              // [Dim*ns] particle-level u vectors
    const float* grid_W,            // [ns*grid_per_partic] weights assoc'd with each grid
    const int*   grid_inds,         // [ns*grid_per_partic] indicies of the grid
    const float  gvol,              // grid volume
    const int    grid_per_partic,   // number of grid points this particle interacts with
    const int    ns,                // number of sites
    const float* L,                 // [Dim] box lengths
    const float* Lh,                // [Dim] half box lengths
    const int    Dim                // Dimensionality
){

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    if ( upartner[id] < 0 ) 
        return;

    // Declare and store some local variables
    int id1 = upartner[id];
    float dr[3];        // vector rij
    float ri[3], rj[3]; // Positions of two relevant partics 
    float fi[3];        // Local force vector
    float T[9];         // 3x3 tensor that gets dotted into S convolved u field
    float I[9];         // Identity tensor
    for ( int j=0 ; j<Dim;  j++ ) {
        ri[j] = x[id*Dim+j];
        rj[j] = x[id1*Dim+j];
        fi[j] = 0.0f;
    }
    // Define the identity 
    for ( int j=0 ; j<Dim ; j++ ) {
        for ( int i=0 ; i<Dim ; i++ ) {
            I[i*Dim+j] = 0.f;
        }
        I[j*Dim+j] = 1.f;
    }

    // Get the vector dr = rj - ri and its magnitude
    float mdr2 = d_pbc_mdr2(rj, ri, dr, L, Lh, Dim);
    float mdr = sqrtf(mdr2);

    for ( int a=0 ; a<Dim; a++ ) {

        float dudra[3];     // Derivative of orientation u w.r.t. ri in the a-direction
        for ( int j=0 ; j<Dim; j++ ) {
            dudra[j] = dr[j] * dr[a] / mdr2;
            if ( j==a ) 
                dudra[j] -= 1.0;

            dudra[j] *= 1.0 / mdr;
        }  // j=0:Dim

        // Zero the tensor
        for ( int j=0 ; j<Dim*Dim; j++ )
            T[j] = 0.f;

        // T = (Iu + uI) \cdot dudra
        for ( int j=0 ; j<Dim ; j++ ) {
            for ( int k=0 ; k<Dim ; k++ ) {
                for ( int m=0 ; m<Dim ; m++ ) {
                    T[j*Dim+k] += ( I[j*Dim+k] * ms_u[id*Dim+m] + ms_u[id*Dim+j] * I[k*Dim+m] ) * dudra[m];
                }
            }
        }

        for ( int m=0 ; m<grid_per_partic; m++ ) {
            int gind = id * grid_per_partic + m;

            int Mind = grid_inds[gind];
            float W3 = grid_W[gind];

            float dotsum = 0.0f;
            for ( int j=0 ; j<Dim ; j++ ) {
                for ( int k=0 ; k<Dim ; k++ ) {
                    dotsum += T[j*Dim+k] * tensorField[Mind*Dim*Dim + j*Dim+k];
                }
            }

            fi[a] += dotsum * W3 * gvol;

        }// m=0:grid_per_partic

    } // a=0:Dim

    // Finally, accumulate the forces
    for ( int a=0 ; a<Dim ; a++ ) {
        atomicAdd(&f[id*Dim + a], fi[a]);
        atomicAdd(&f[id1*Dim+ a], -fi[a]);
    }

}


__global__ void d_storeTensorComponent(
    float* Tfield,                      // [Dim*Dim*M] tensor field to have component populated
    const cufftComplex* source,           // [M] result of convolution theorem to be placed in array
    const int ii, const int jj,         // indices of the components to be extracted
    const int M,
    const int Dim ) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= M)
        return;
    
    int space_ind = id * Dim * Dim ;
    Tfield[space_ind + ii*Dim + jj] = source[id].x;    

}


__global__ void d_extractTensorComponent(
    cufftComplex* dest,            // [M] array to be populated
    const float* Tfield,           // [Dim*Dim*M] tensor field in real-space
    const int ii, const int jj,    // indices of the components to be extracted
    const int M,
    const int Dim ) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= M)
        return;
    
    int space_ind = id * Dim * Dim ;
    dest[id].x = Tfield[space_ind + ii*Dim + jj];
    dest[id].y = 0.f;

}

// Uses the partner list to generat the unit vector orientation for each particle
__global__ void d_calcParticleSTensors(
    float* ms_u,                // [Dim*ns] stores the orientation vector
    float* ms_S,                // [Dim*Dim*ns] stores the particle S-tensor
    const float* x,             // [Dim*ns] particle positions
    const int* upartner,        // [ns] partner particle for defining u-vector
    const float* L,             // [Dim] box dimensions
    const float* Lh,            // [Dim] half box dimensions
    const int Dim,              // Dimensionality
    const int ns ) {            // number of sites

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    if ( upartner[id] < 0 )
        return;

    int id2 = upartner[id];    
    float r1[3], r2[3], dr[3], mdr2, mdr;

    // Pull the particle positions
    for ( int j=0 ; j<Dim ; j++ ) {
        r1[j] = x[id*Dim  + j];
        r2[j] = x[id2*Dim + j];
    }

    // Get the vector and squared distance between the two
    // dr = r2 - r1
    mdr2 = d_pbc_mdr2(r2, r1, dr, L, Lh, Dim);
    mdr = sqrtf(mdr2);

    for ( int j=0 ; j<Dim ; j++ ) {
        ms_u[id*Dim + j] = dr[j] / mdr;
    }

    for ( int j=0 ; j<Dim ; j++ ) {
        for ( int k=0 ; k<Dim ; k++ ) {
            ms_S[id*Dim*Dim + j*Dim + k] = ms_u[id*Dim+j] * ms_u[id*Dim+k];
        }
        ms_S[id*Dim*Dim + j * Dim + j] -= 1.f / float(Dim);
    }

}


// Uses the pre-computed particle-level S tensors and maps them to the mesh
// Assumes the mesh has previously been calculated in this time step so the
// particles 'know' grid_W, grid_inds
__global__ void d_mapFieldSTensors(
    float* field_S,                 // [Dim*Dim*M] S tensor field created in this routine
    const int* upartner,            // [ns] List of partner particles
    const float* ms_S,              // [Dim*Dim*ns] Particles' S tensors
    const float* grid_W,            // [ns*grid_per_partic] particle weight for grid points
    const int* grid_inds,           // [ns*grid_per_partic] index of the relevant grid points
    const int ns,                   
    const int grid_per_partic, 
    const int Dim) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;
    if ( upartner[id] < 0 )
        return;

    for ( int i=0 ; i<grid_per_partic ; i++ ) {
        float W3 = grid_W[id * grid_per_partic + i];
        int space_ind = grid_inds[id * grid_per_partic + i];

        for ( int j=0 ; j<Dim ; j++ ) {
            for ( int k=0 ; k<Dim ; k++ ) {
                int grid_ind = space_ind * Dim * Dim + j*Dim + k;
                int partic_ind = id * Dim * Dim + j * Dim + k;
                atomicAdd(&field_S[grid_ind], W3 * ms_S[partic_ind]);
            }
        }
    }// i=0:grid_per_partic


}

// Uses the partner list to generat the unit vector orientation for each particle
__global__ void d_SumAndAverageSTensors(
    const float* ms_S,                // [Dim*Dim*ns] stores the particle S-tensor
    float* summed_S_Tensor,             // [Dim*ns] particle positions
    const int* upartner,        // [ns] partner particle for defining u-vector
    const int Dim,              // Dimensionality
    const int ns ) {            // number of sites

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    if ( upartner[id] < 0 )
        return;

    int dd = Dim * Dim;

    for (int j = 0; j < dd; j++)
    {
        atomicAdd(&summed_S_Tensor[j], ms_S[id*dd + j]);
    }

}


__global__ void d_doubleDotTensorFields(
    float* out,             // [M] field containing result of double dot operation
    const float* S1,        // [M*Dim*Dim] Tensor field to be double dotted into S2
    const float* S2,        // [M*Dim*Dim]
    const int M,            // Number of grid points
    const int Dim ) {       // Dimensionality

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= M)
        return;

    float ddot = 0.f;
    for ( int i=0 ; i<Dim ; i++ ) {
        for ( int j=0 ; j<Dim ; j++ ) {
            ddot += S1[id*Dim*Dim + i*Dim + j] * S2[id*Dim*Dim + i*Dim + j];
        }
    }
    out[id] = ddot;
}
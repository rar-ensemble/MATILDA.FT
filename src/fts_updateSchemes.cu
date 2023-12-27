// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "include_libs.h"

// this routine does not care about real space or k space
__global__ void d_fts_updateEM(
    cuDoubleComplex* w, 
    const cuDoubleComplex* dHdw, 
    const double delt,
    const int M
    ) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= M)
        return;

    w[ind].x = w[ind].x - delt * dHdw[ind].x;
    w[ind].y = w[ind].y - delt * dHdw[ind].y;

}


// this routine assumes k-space updating 
// and that Ak is purely real in k-space
__global__ void d_fts_update1S(
    cuDoubleComplex* w, 
    const cuDoubleComplex* dHdw, 
    const cuDoubleComplex* Ak,
    const double delt,
    const int M
    ) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= M)
        return;

    double denom = 1.0 + delt * Ak[ind].x;
    if ( denom != 0.0 ) {
        w[ind].x = ( w[ind].x - delt * (dHdw[ind].x - Ak[ind].x * w[ind].x) ) / denom;              
        w[ind].y = ( w[ind].y - delt * (dHdw[ind].y - Ak[ind].x * w[ind].y) ) / denom;
    }

}

// this routine does not care about real space or k space
// Updates using Eq 5.30 in ``FTS in SM and QF''
__global__ void d_fts_updateEMPC(
    cuDoubleComplex* w,             // Field being updated
    const cuDoubleComplex* wo,       // Original field config
    const cuDoubleComplex* dHdw,    // corrected force
    const cuDoubleComplex* dHdwP,   // predicted force
    const double delt,              // Size of time step
    const int M                     // number of grid points
    ) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= M)
        return;

    double halfDelt = 0.5 * delt;
    w[ind].x = wo[ind].x - halfDelt * (dHdw[ind].x + dHdwP[ind].x);
    w[ind].y = wo[ind].y - halfDelt * (dHdw[ind].y + dHdwP[ind].y);

}
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
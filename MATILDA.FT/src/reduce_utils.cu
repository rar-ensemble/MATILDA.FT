// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"


__global__ void d_reduce_float(float tot_sum, const float* dat, 
    const int max) {

    extern __shared__ float block_sum[];

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind > max)
        return;

    const int tid = threadIdx.x;

    if (ind == 0)
        tot_sum = 0.f;

    if (tid == 0)
        block_sum[tid] = 0.f;

    __syncthreads();

    atomicAdd(&block_sum[tid], dat[ind]);

    __syncthreads();

    if ( tid == 0 )
        atomicAdd(&tot_sum, block_sum[tid]);
}

float reduce_device_float(float* d_dat, const int threads, const int m)
{
 
    int loc_threads = (m < threads) ? m : threads;
    int loc_blocks = m / threads;

    float* d_sum, h_sum;
    cudaMalloc(&d_sum, sizeof(float));
    
    d_reduce_float<<<loc_blocks, loc_threads,loc_blocks*sizeof(float)>>>(*d_sum, d_dat, m);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&d_sum);

    return h_sum;
}
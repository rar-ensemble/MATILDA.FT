// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "nlist.h"

__global__ void d_make_nlist_3d_1(
    const float *x, // [ns*Dim], particle positions
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_LOW_DENS_FLAG,
    int step,
    const int nncells,
    const int ad_hoc_density,
    thrust::device_ptr<int> d_index, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D);

__global__ void d_make_nlist_3d_2(
    const float *x, // [ns*Dim], particle positions
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    int step,
    const int nncells,
    const float r_skin,
    const int ad_hoc_density,
    thrust::device_ptr<int> d_index, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D);


__global__ void d_make_nlist_2d_1(
    const float *x, // [ns*Dim], particle positions
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_LOW_DENS_FLAG,
    int step,
    const int nncells,
    const int ad_hoc_density,
    thrust::device_ptr<int> d_index, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D);


__global__ void d_make_nlist_2d_2(
    const float *x, // [ns*Dim], particle positions
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    int step,
    const int nncells,
    const float r_skin,
    const int ad_hoc_density,
    thrust::device_ptr<int> d_index, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D);

#ifndef _NLIST_HALF_DISTANCE
#define _NLIST_HALF_DISTANCE


class NListHalfDistance : public NList {
protected:
                      
public:

	~NListHalfDistance();
	NListHalfDistance(std::istringstream&);

    // override functions

    void MakeNList() override;

};

#endif


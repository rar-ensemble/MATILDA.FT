// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "nlist.h"

__global__ void d_nlist_distance_update_grid(
    const float* x,            // [ns*Dim], particle positions
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    const int ad_hoc_density,
    const int* site_list,   // List of sites in the group
    const int ns,           // Number of sites in the list
    const int D,
    int step,
    thrust::device_ptr<int> d_LOW_DENS_FLAG);


__global__ void d_nlist_distance_update_nlist(
    const float *x, // [ns*Dim], particle positions
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    const int ad_hoc_density,
    const int *site_list, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D,
    const int nncells,
    const float r_skin,
    const int step);

__global__ void d_nlist_distance_update(
    const float *x, // [ns*Dim], particle positions
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_LOW_DENS_FLAG,
    int step,
    const int nncells,
    const float r_skin,
    const int ad_hoc_density,
    const int *site_list, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D);

#ifndef _NLIST_DISTANCE
#define _NLIST_DISTANCE


class NListDistance : public NList {
protected:
                      
public:
	~NListDistance();
	NListDistance(std::istringstream&);
    void MakeNList() override;
    void WriteNList();
    thrust::device_vector<int> d_LOW_DENS_FLAG;

};

#endif


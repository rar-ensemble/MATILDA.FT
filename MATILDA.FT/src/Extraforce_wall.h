// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Extraforce.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Warning - can modify particle positions
// Make sure all the momentum thing is conserved


 __global__ void d_wall_hard(
    float* f,             // [ns*Dim], particle forces
    float* x,            // [ns*Dim], particle positions
    float* v,
    thrust::device_ptr<float> d_w,
    const int w_size,
    thrust::device_ptr<int> d_index,  // List of sites in the group
    const int ns,           // Number of sites in the list
    const int D);

__global__ void d_wall_exp(
    float* f,
    float* x,
    float* v,
    thrust::device_ptr<float> d_w,
    const int w_size,
    thrust::device_ptr<int> d_index,
    const int ns,
    const int D);

__global__ void d_wall_rinv(
    float* f,             // [ns*Dim], particle forces
    float* x,            // [ns*Dim], particle positions
    float* v,
    thrust::device_ptr<float> d_w,
    const int w_size,
    thrust::device_ptr<int> d_index,  // List of sites in the group
    const int ns,           // Number of sites in the list
    const int D);


#ifndef _EXTRAFORCE_WALL
#define _EXTRAFORCE_WALL


class Wall : public ExtraForce {
protected:

    std::string wall_style_str;
    int wall_style, w_size;                        
    thrust::device_vector<float> d_wall;

public:
	~Wall();
	Wall(std::istringstream&);
    void AddExtraForce() override;
    void UpdateVirial(void) override;
};

#endif


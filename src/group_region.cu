// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "group.h"
#include "group_region.h"
#include <stdio.h>

using namespace std;


GroupRegion::~GroupRegion(){}

GroupRegion::GroupRegion(istringstream& iss) : Group(iss) {

    dynamic_group_flag = 1;

    d_all_id.resize(ns);
    thrust::fill(d_all_id.begin(), d_all_id.end(), -1);

    device_mem_use += ns * sizeof(int);

    float tmp;
    int w_size = 0;

    // Read the wall data

    while(iss >> tmp){
        d_wall_data.push_back(tmp); //dim l h
        ++w_size;
    }

    // Check if the input was provided correctly

    if ((w_size%3) != 0){
        die("Incorrect input structure\n <dim> <low> <high> ...");
    }
    n_walls = int(w_size/3);
}

void GroupRegion::CheckGroupMembers(void){

    d_CheckGroupMembers<<<GRID_ALL, BLOCK>>>(
        d_x,
        d_wall_data.data(), n_walls,
        d_all_id.data(),
        ns,
        Dim);

     UpdateGroupMembers();
}

__global__ void d_CheckGroupMembers(
    const float* x,
    thrust::device_ptr<float> d_wall_data,
    const int n_walls,
    thrust::device_ptr<int> d_all_id,
    const int ns,
    const int Dim){

    int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (list_ind >= ns)
        return;
    int ind = list_ind;

    for(int i = 0; i < n_walls; ++i){

        int j = int(d_wall_data[3 * i]);
        float low = d_wall_data[3 * i + 1];
        float high = d_wall_data[3 * i + 2];
        float my_x = x[ind * Dim + j];

        d_all_id[ind] = ind;

        if (my_x >= low && my_x <= high){
            d_all_id[ind] = ind;
        }
        else{
            d_all_id[ind] = -1;
        }
    }
}
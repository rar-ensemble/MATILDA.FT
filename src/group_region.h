// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <string>
#include <sstream>
#include <vector>
#include "group.h"


__global__ void d_CheckGroupMembers(
    const float* x,
    thrust::device_ptr<float> d_wall_data,
    const int n_walls,
    thrust::device_ptr<int> d_all_id,
    const int ns,
    const int Dim);


#ifndef _GROUPS_REGION
#define _GROUPS_REGION


class GroupRegion : public Group {
private:

    thrust::device_vector<float> d_wall_data;
    int n_walls;

public:

    GroupRegion(std::istringstream& iss); 
    ~GroupRegion();
    void CheckGroupMembers(void) override;
    
};


#endif
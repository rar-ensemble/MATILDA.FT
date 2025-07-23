// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <sstream>
#include <string>
#include <curand_kernel.h>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifndef _DYNAMICGROUPS
#define _DYNAMICGROUPS


class GroupDynamic {
protected:

    static int total_num_dynamic_groups;
    std::string command_line;
    std::string style;

public:

    int id;
    int nsites;             // Number of sites in this group
    int GRID, BLOCK;
    
    thrust::device_vector<int> d_index, d_all_id;
    thrust::host_vector<int> index, all_id;

    std::string name;       // Text name of this group

    GroupDynamic(std::istringstream& iss); // Constructor for parsing input file
    ~GroupDynamic();

    void CollectID(void);

    virtual void CheckID(void) = 0;

    struct copy_id_check
    {
        __host__ __device__
        bool operator()(const int x)
        {
            return (x != -1);
        }
    };
};

#endif
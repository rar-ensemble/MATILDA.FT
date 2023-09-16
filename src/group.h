// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <iostream>     // std::cout
#include <sstream>      // std::istringstream
#include <string>       // std::string
#include <curand_kernel.h>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifndef _GROUP
#define _GROUP

/**
 *  Class Name: Group
 *
 *  Description:
 *
 *  Class Memebers:
 *
 *  Functions:
 */


class Group {
protected:

    static int total_num_groups;
    std::string command_line;    // Stores the full command given in input file to create group
    std::string style; // Stores the type of group

public:

    // Basics

    Group();             // Constructor used for the "all" group - called during the initialization
    Group(std::istringstream& iss);
    ~Group();

    // Class Memebers

    int id;                 // ID of the group in Groups vector
    int nsites;             // Number of sites in this group
    int dynamic_group_flag; // Flag for dynamic group: 0 - static, 1 - dynamic
    std::string name;       // name of the group


    thrust::host_vector<int> index;
    thrust::host_vector<int> all_id;

    thrust::device_vector<int> d_index;
    thrust::device_vector<int> d_all_id;
    

    // GPU variables

    int GRID, GRID_ALL;               // block grid size for operations on this group on the GPU
    int BLOCK;              // size of the blocks for operations on this group on GPU

    // Additional Functions

    void UpdateGroupMembers(void);

    std::string printCommand(){return command_line;}

    struct is_group_member
    {
        __host__ __device__
        bool operator()(const int x)
        {
            return (x != -1);
        }
    };

    // Override functions

    virtual void CheckGroupMembers(void){return;};

};


#endif
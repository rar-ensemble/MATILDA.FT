// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "group.h"
#include "group_type.h"
#include "group_region.h"
#include "group_id.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>


using namespace std;

// Initialize static variables

int Group::total_num_groups = 0;

Group::~Group() {}

// Default constructor for the "all" group
Group::Group(){

    id=total_num_groups++;
    
    name = "all";
    command_line = "Default initialization of the 'All' group";

    nsites = ns; // number of group members = all particles in the simulation

    // set GPU variables

    BLOCK = threads;
    GRID = (int)ceil((float)(ns) / threads);

    // Initialize index array

    d_index.resize(ns);
    thrust::sequence(thrust::device, d_index.begin(), d_index.end()); // initialize the index array on the GPU

    index.resize(ns);
    index = d_index;

    device_mem_use += ns * sizeof(int); //device memory use

    dynamic_group_flag = 0;

}

// parametrized constructor for the specific groups
Group::Group(istringstream& iss) {

    id=total_num_groups++;

    command_line = iss.str();

    readRequiredParameter(iss, name); //group name
    readRequiredParameter(iss, style); //group style used to dispatch a specialized constructor

    BLOCK = threads;
    GRID_ALL = (int)ceil((float)(ns) / threads);
    GRID = GRID_ALL;

}


// Specialized constructors

Group* GroupFactory(istringstream &iss){

	std::stringstream::pos_type pos = iss.tellg(); // get the current position - beginning of the string
	string s1;
	iss >> s1 >> s1; // dynamicgroup <name> type
	iss.seekg(pos); // reset the stringf to the initial position and pass to the specific contructor

    // Makes the group containing all the particles
    // Created by default during the initialization

    if (s1 == "all"){
        return new Group();
    }

    // Static groups

	if (s1 == "type"){
		return new GroupType(iss);
	}
    if (s1 == "id"){
		return new GroupID(iss);
	}

    // Dynamic groups

	if (s1 == "region"){
		return new GroupRegion(iss);
	}
	
	die(s1 + " is not a valid Group");
	return 0;
}

// Additional functions

void Group::UpdateGroupMembers(void){

    // Only execute if the group is dynamic
    if (dynamic_group_flag == 1){

        nsites = ns - thrust::count(d_all_id.begin(), d_all_id.end(), -1); // size of the new index array

        index.resize(nsites);
        d_index.resize(nsites);

        GRID = (int)ceil((float)(nsites) / threads);

        thrust::copy_if(thrust::device, d_all_id.begin(), d_all_id.end(), d_index.begin(), is_group_member()); //only copy the id of the group members
        thrust::fill(d_all_id.begin(), d_all_id.end(), -1); //reset the array

        // std::cout << "Group " << name <<" is dynamic: ";
        // for (int i = 0; i < nsites; ++i){
        //     std::cout << d_index[i] << ", "<< std::endl;
        // }
        // std::cout << std::endl;
    }
    // else{
    //     std::cout << "Group " << name <<" is static: ";
    //     for (int i = 0; i < nsites; ++i){
    //         std::cout << d_index[i] << ", "<< std::endl;
    //     }
    //     std::cout << std::endl;
    // }
}

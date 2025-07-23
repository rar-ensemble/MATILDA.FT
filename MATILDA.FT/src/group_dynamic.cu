// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


﻿#include "globals.h"
#include "group_dynamic.h"
#include "group_d_region.h"
#include <curand_kernel.h>
#include <curand.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>


using namespace std;

int GroupDynamic::total_num_dynamic_groups = 0;

GroupDynamic::GroupDynamic(istringstream& iss) {

    id = total_num_dynamic_groups++;

    command_line = iss.str();

    readRequiredParameter(iss, name);
    readRequiredParameter(iss, style);

    d_all_id.resize(ns);
    thrust::fill(d_all_id.begin(), d_all_id.end(), -1);
    device_mem_use += ns * sizeof(int);

    BLOCK = threads;
    GRID = (int)ceil((float)(ns) / threads);

}

GroupDynamic::~GroupDynamic() {
}

GroupDynamic* GroupDynamicFactory(istringstream &iss){

	std::stringstream::pos_type pos = iss.tellg(); // get the current position - beginning of the string
	string s1;
	iss >> s1 >> s1; // dynamicgroup <name> type

	iss.seekg(pos); // reset the stringf to the initial position and pass to the specific contructor

	if (s1 == "region"){
		return new GroupDRegion(iss);
	}
	die(s1 + "is not a correct dynamic group type.");
	return 0;
}


void GroupDynamic::CollectID(void){

    nsites = ns - thrust::count(d_all_id.begin(), d_all_id.end(), -1);
    index.resize(nsites);
    d_index.resize(nsites);
    thrust::copy_if(thrust::device, d_all_id.begin(), d_all_id.end(), d_index.begin(), copy_id_check());

    std::cout << "Step " << step << ": ";

    for (int i = 0; i < nsites; ++i){
        int my_id = d_index[i];
        std::cout << my_id << ", ";
    }
    std::cout << endl;

    thrust::fill(d_all_id.begin(), d_all_id.end(), -1);
}


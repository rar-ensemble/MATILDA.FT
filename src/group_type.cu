// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "group_type.h"
#include <algorithm>


using namespace std;


GroupType::~GroupType(){
}

GroupType::GroupType(istringstream& iss) : Group(iss) {

    dynamic_group_flag = 0; //set the group to static

    int at = -1;

    while (iss >> at){
        atom_types.push_back(at-1);
    }

    for (int i = 0; i < ns; i++) { 
        if (std::count(atom_types.begin(), atom_types.end(), tp[i])){
            index.push_back(i);
            d_index.push_back(i);
        }
    }

    nsites = index.size();
    d_all_id.resize(0);

    // Device Variables

    BLOCK = threads;
    GRID = (int)ceil((float)(nsites) / threads);

}

// Additional functions

void GroupType::CheckGroupMembers(void){
    UpdateGroupMembers();
}









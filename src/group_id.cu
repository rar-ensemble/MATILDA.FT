// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "group_id.h"
#include <algorithm>


using namespace std;

GroupID::~GroupID(){
}

GroupID::GroupID(istringstream& iss) : Group(iss) {

    dynamic_group_flag = 0; //set the group to static

    nsites = 0;
    std::string line;

    readRequiredParameter(iss, input_file);

    // iss >> input_file;

    ifstream in(input_file);
    getline(in, line);
    istringstream str(line);
    
    int i = -1;

    cout << "Group from: " << input_file << endl;

    while (str >> i){
        index.push_back(i-1);
        d_index.push_back(i-1);
    }

    nsites = index.size();
    d_all_id.resize(0);

    // Device Variables

    BLOCK = threads;
    GRID = (int)ceil((float)(nsites) / threads);

}

// Additional functions

void GroupID::CheckGroupMembers(void){
    UpdateGroupMembers();
}
// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "Extraforce.h"
#include "Extraforce_wall.h"
#include "Extraforce_lewis.h"
#include "Extraforce_lewis_full.h"
#include "Extraforce_lewis_serial.h"
#include "Extraforce_dpd.h"
#include "Extraforce_dynamic.h"

using namespace std;


ExtraForce::ExtraForce(istringstream& iss) {

    command_line = iss.str();
	readRequiredParameter(iss, group_name);
    readRequiredParameter(iss, style);
    group_index = get_group_id(group_name);
    group = Groups.at(group_index);

    if (this->group_index == -1) {
        die("Group" + group_name + "not found to apply to "+ style +"!");
    }
}

ExtraForce::~ExtraForce() {
}

ExtraForce* ExtraForceFactory(istringstream &iss){
	std::stringstream::pos_type pos = iss.tellg();
	string s1;
	iss >> s1 >> s1;
	iss.seekg(pos);
	if (s1 == "langevin"){
		return new Langevin(iss);
	}
	if (s1 == "midpush"){
		return new Midpush(iss);
	}
	if (s1 == "wall"){
		return new Wall(iss);
	}

	if (s1 == "dpd"){
		return new DPD(iss);
	}
	if (s1 == "dynamic"){
		return new Dynamic(iss);
	}
	if (s1 == "lewis"){
		return new Lewis(iss);
	}
	// if (s1 == "lewis_full"){
	// 	return new LewisFull(iss);
	// }

	if (s1 == "lewis_serial"){
		return new Lewis_Serial(iss);
	}				
	die(s1 + " is not a valid ExtraForce, you have failed miserably!\n Supported ExtraForces are: midpush, langevin, wall");
	return 0;
}
// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "Measure.h"
#include "Measure_surface_tension.h"


using namespace std;


Measure::Measure(istringstream& iss) {

    // command_line = iss.str();
	// readRequiredParameter(iss, group_name);
    readRequiredParameter(iss, style);
    // group_index = get_group_id(group_name);
    // group = Groups.at(group_index);

    // if (this->group_index == -1) {
    //     die("Group" + group_name + "not found to apply to "+ style +"!");
    // }
}

Measure::~Measure() {
}

Measure* MeasureFactory(istringstream &iss){
	std::stringstream::pos_type pos = iss.tellg();
	string s1;
	iss >> s1 ;
	iss.seekg(pos);


	if (s1 == "surface_tension"){
		return new SurfaceTension(iss);
	}
		
	die(s1 + " is not a valid Measure, you have failed miserably!");
	return 0;
}
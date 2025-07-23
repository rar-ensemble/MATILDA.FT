// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "integrator.h"
#include "integrator_VV.h"
#include "integrator_GJF.h"
// #include "integrator_EM.h"

using namespace std;

void Integrator::Integrate_1(){
}

void Integrator::Integrate_2(){
}

Integrator::Integrator(istringstream& iss) {
    command_line = iss.str();
    readRequiredParameter(iss, group_name);
    readRequiredParameter(iss, name);
    group_index = get_group_id(group_name);
    group = Groups.at(group_index);
}


Integrator::~Integrator() {
}


Integrator* IntegratorFactory(std::istringstream& iss){
	std::stringstream::pos_type pos = iss.tellg();
	string s1;
	iss >> s1 >> s1;
	iss.seekg(pos);

    if (s1 == "VV") {
        return new VV(iss);
    }
    if (s1 == "GJF") {
        return new GJF(iss);
    }


	die(s1 + " is not a valid Integrator");
    return 0;
}

int Integrator::using_GJF = 0;
// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_integrator.h"
#include "PS_Box.h"

// #include "integrator_VV.h"
// #include "integrator_GJF.h"
// #include "integrator_EM.h"


void Integrator::Integrate_1(){
}

void Integrator::Integrate_2(){
}

Integrator::Integrator(std::istringstream& iss, PS_Box* box) : mybox(box) {
    command_line = iss.str();
    
    delt = 0.001;   // set default time step size
    readRequiredParameter(iss, groupName);
    readRequiredParameter(iss, name);
    
    group_index = mybox->findGroupInteger(groupName);
}


Integrator::~Integrator() {
}


Integrator* IntegratorFactory(std::istringstream& iss){
	std::stringstream::pos_type pos = iss.tellg();
	std::string s1;
	iss >> s1 >> s1;
	iss.seekg(pos);

    // if (s1 == "VV") {
    //     return new VV(iss);
    // }
    // if (s1 == "GJF") {
    //     return new GJF(iss);
    // }


	die(s1 + " is not a valid Integrator");
    return 0;
}

int Integrator::using_GJF = 0;
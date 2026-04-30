// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_integrator.h"
#include "ps_integratorGJF.h"
#include "ps_integratorEM.h"
#include "PS_Box.h"


void Integrator::Integrate_1(){}

void Integrator::Integrate_2(){}

void Integrator::finishInitialization() {}

Integrator::Integrator(std::istringstream& iss, PS_Box* box) : mybox(box) {
    command_line = iss.str();
    
    delt = 0.002;   // set default time step size
    readRequiredParameter(iss, groupName);
    readRequiredParameter(iss, name);

    // parse optional arguments
    if ( iss.tellg() != -1 ) {
        std::string word;
        iss >> word;

        if ( word == "delt" ) {
            iss >> delt;
        }

        else { 
            die("Invalid keyword in Integrator command");
        }
    }
    
}

// Checks the groups in the box class for the group with name "groupName"
// findGroupInteger will quit if not found.
void Integrator::findGroup() {
    group_index = mybox->findGroupInteger(groupName);
}

Integrator::~Integrator() {
}


Integrator* IntegratorFactory(std::istringstream& iss, PS_Box* bx){
	std::stringstream::pos_type pos = iss.tellg();
	std::string s1;
	iss >> s1 >> s1;
	iss.seekg(pos);

    if (s1 == "EM") {
        return new EM(iss, bx);
    }
    if (s1 == "GJF") {
        return new GJF(iss, bx);
    }


	die(s1 + " is not a valid Integrator");
    return 0;
}

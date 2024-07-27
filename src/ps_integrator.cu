// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_integrator.h"

using namespace std;

void Integrator::Integrate_1(){
}

void Integrator::Integrate_2(){
}

Integrator::Integrator(istringstream& iss) {
    command_line = iss.str();

}


Integrator::~Integrator() {
}


Integrator* IntegratorFactory(std::istringstream& iss){
	std::stringstream::pos_type pos = iss.tellg();
	string s1;
	iss >> s1 >> s1;
	iss.seekg(pos);


	die(s1 + " is not a valid Integrator");
    return 0;
}

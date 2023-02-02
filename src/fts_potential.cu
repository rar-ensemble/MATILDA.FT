// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_potential.h"
#include "fts_potential_helfand.h"
#include "fts_potential_flory.h"
#include "include_libs.h"
#include <istream>
void die(const char*);

FTS_Potential::FTS_Potential(std::istringstream &iss, FTS_Box* p_box) : mybox(p_box){

    input_command = iss.str();

    mybox = p_box;

}

FTS_Potential::~FTS_Potential() {}

FTS_Potential* FTS_PotentialFactory(std::istringstream &iss, FTS_Box* box) {
//    std::stringstream::pos_type pos = iss.tellg();
//    std::cout << "Found position: " << pos << std::endl;
    std::string s1;
	iss >> s1 ;
//	iss.seekg(pos);
	if (s1 == "Helfand" || s1 == "helfand"){
		return new PotentialHelfand(iss, box);
    }

    else if ( s1 == "Flory" || s1 == "flory" ) {
        return new PotentialFlory(iss, box);
    }
	
	else {
        std::string s2 = s1 + " is not a valid FTS_Potential"; 
        die(s2.c_str());
    }
	return 0;
}

std::string FTS_Potential::printCommand() {
    return input_command;
}

std::string FTS_Potential::printStyle() {
    return potentialStyle;
}
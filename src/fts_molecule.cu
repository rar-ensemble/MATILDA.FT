// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_molecule.h"
#include "fts_molecule_linear.h"
#include "include_libs.h"
#include "FTS_Box.h"

void die(const char*);

FTS_Molec::FTS_Molec(std::istringstream &iss, FTS_Box* bx) {
    input_command = iss.str();

    iss >> molec_type;
    mybox = bx;

    phi = nmolecs = activity = -1.0;
    
    std::string s1;
    iss >> s1;
    
    if ( s1 == "phi" ) {
        iss >> phi;
    }
    else if ( s1 == "nmolecs" ) {
        iss >> nmolecs; 
    }
    else if ( s1 == "activity" ) {
        iss >> activity;
    }

    std::cout << "Molecule genereated with: " << input_command << std::endl;
    std::cout << "  Now has phi, nmolecs, activity: " << phi << " " << nmolecs << " " << activity << std::endl;

    int M = mybox->M;

    // Make the vectors their proper sizes
    density.resize(M);
    d_density.resize(M);
    d_cDensity.resize(M);
}

FTS_Molec::~FTS_Molec(){}



FTS_Molec* FTS_MolecFactory(std::istringstream &iss, FTS_Box* box) {
    std::stringstream::pos_type pos = iss.tellg();
    std::string s1;
	iss >> s1 ;
	iss.seekg(pos);
	if (s1 == "linear"){
		return new LinearMolec(iss, box);
    }
	
	else {
        std::string s2 = s1 + " is not a valid FTS_Molec"; 
        die(s2.c_str());
    }
	return 0;
}
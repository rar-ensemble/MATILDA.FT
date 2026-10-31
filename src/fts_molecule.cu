// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_molecule.h"
#include "fts_molecule_linear.h"
#include "fts_molecule_lc.h"
#include "fts_moleculeHParticle.h"
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
    else {
        std::string LW = s1 + " is not a valid way to define the number of particles/molecules";
        die(LW.c_str());
    }

    // std::cout << "Molecule genereated with: " << input_command << std::endl;
    // std::cout << "  Now has phi, nmolecs, activity: " << phi << " " << nmolecs << " " << activity << std::endl;

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
	
    else if ( s1 == "HParticle" ) {
        return new ParticleMolec(iss, box);
    }

    else if ( s1 == "liquid-crystal") {
        return new LCMolec(iss, box);
    }
    
	else {
        std::string s2 = s1 + " is not a valid FTS_Molec"; 
        die(s2.c_str());
    }
	return 0;
}


// Changes molecule parameters depending on the style passed by 'how2Mod'
// Any other related changes to param that are needed or associated memory 
// allocation need to be taken care of elsewhere. This is just to parse
// the changes to the variable.
void modifyMolecParam(double& param, const std::string how2Mod, const double change) {
    if (how2Mod == "scale" ) {
        param *= change;
    }

    else if ( how2Mod == "delta" ) {
        param += change;
    }

    else if ( how2Mod == "value" ) {
        param = change;
    }
}


// Computes the contribution to the Hamiltonian 
// for this molecule style. 
std::complex<double> FTS_Molec::calcHTerm() {
    std::complex<double> ht = 0.0;

    if ( nmolecs > 0.0 ) {
        ht = (std::complex<double>)( -nmolecs * log( Q ) );
    }
    else if ( activity > 0.0 ) {
        ht = (std::complex<double>)( -activity*mybox->V*Q );
    }
    else {
        die("Invalid conditions to calcHTerm()");
    }

    return ht;
}

void FTS_Molec::modifyMolecule(std::istringstream& iss) {

    std::string property, how2Mod;
    iss >> property;
    
    if ( property == "activity" ) {
        iss >> how2Mod;

        double change;
        iss >> change;

        std::cout << "activity changed from " << activity ;

        modifyMolecParam(activity, how2Mod, change);
        
        // if (how2Mod == "scale" ) {
        //     activity *= change;
        // }

        // else if ( how2Mod == "delta" ) {
        //     activity += change;
        // }

        // else if ( how2Mod == "value" ) {
        //     activity = change;
        // }

        std::cout << " to " << activity << std::endl;
    }

    // Changes here require re-calculating nmolecs
    else if ( property == "phi" ) {
        die("changes in FTS:molec:phi not yet supported");
    }

    // Changes here require re-calculating nmolecs
    else if ( property == "nmolecs" ) {
        die("changes in FTS:molec:nmolecs not yet supported");
    }

    else {
        std::string lastWords = property + " is not a valid property to change in FTS:molecule";
        die(lastWords.c_str());
    }
}


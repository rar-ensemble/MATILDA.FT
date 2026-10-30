// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_molecule_lc.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include "FTS_Box.h"
#include "fts_species.h"
void die(const char*);

LCMolec::~LCMolec(){}

LCMolec::LCMolec(std::istringstream& iss, FTS_Box* p_box) : FTS_Molec(iss, p_box) {
    if ( mybox->Nu_tot == 0 ) {
        die("must define u_grid in input file before initializing liquid crystal molecule");
    }

    // Number of sites on the LC
    iss >> Ntot ;

    // Allocate the site-specific vectors
    text_species.resize(Ntot);
    int_species.resize(Ntot);
    d_int_species.resize(Ntot);

    for ( int i=0 ; i<Ntot ; i++ ) {
        iss >> text_species[i] ;
    }

    // Optional arguments
    if ( iss.tellg() != -1 ) {
        std::string s1;
        iss >> s1;
        if ( s1 == "smear" || s1 == "Smear" ) {
            iss >> s1 ; 
            doSmear = 1;
            if ( s1 == "Gaussian" || s1 == "gaussian" ) {
                iss >> smearLength;
                std::cout << "Smearing with unit Gaussian with width " << smearLength << std::endl;
            }
            else {
                die("invalid smearing function passed to liquid-crystal molecule");
            }
        } // s1 == "smear"

        else if ( s1 == "lc_style" || s1 == "lcStyle" ) {
            iss >> lc_style;
            if ( lc_style != "all" || lc_style != "middle" ) {
                die("Invalid lc_style given for liquid-crystal molecule");
            }
        }

        else if ( s1 == "bond_length" || s1 == "bondLength" ) {
            iss >> bond_length;
        }
    } // end of optional arguments


    // Determine integer block Species
    for ( int j=0 ; j<Ntot; j++ ) {
        int_species[j] = -55;
        for ( int i=0 ; i<mybox->Species.size(); i++ ) {
            if ( text_species[j] == mybox->Species[i].fts_species ) {
                int_species[j] = i;
            }
        }
        if ( int_species[j] < 0 ) {
            std::string lastWords = text_species[j] + " species not found!";
            die(lastWords.c_str());
        }
    }
    // Send integers to device
    d_int_species = int_species;


    // Allocate smearing
    smearFunc.resize(mybox->M);
    d_smearFunc.resize(mybox->M);

    // initializes Gaussian smear in k-space
    mybox->initSmearGaussian(smearFunc, 1.0, smearLength);
    // mybox->writeTComplexGridData("smearGaussian.dat", smearFunc);
    
    // Send smear to the device
    d_smearFunc = smearFunc;

}


void LCMolec::calcDensity() {
    
}


// Recomputes number of molecules for GCE simulations
void LCMolec::recomputeNmolecs() {
    if ( activity > 0.0 ) {
        nmolecs = mybox->C * phi * mybox->Vfree * mybox->Nr / double(Ntot);   
    }
}
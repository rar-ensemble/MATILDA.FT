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

    bond_length = 1.0;

    // Allocate the site-specific vectors
    text_species.resize(Ntot);
    int_species.resize(Ntot);
    d_int_species.resize(Ntot);

    for ( int i=0 ; i<Ntot ; i++ ) {
        iss >> text_species[i] ;
    }

    // Optional arguments
    while ( iss.tellg() != -1 ) {
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
            if ( lc_style != "all" && lc_style != "middle" ) {
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
    


    //////////////////////////
    // begin debug
    double u[2], r0[2], r1[2], r2[2];
    u[0] = u[1] = 0.5 * pow(2.0,0.5);
    
    r1[0] = r1[1] = mybox->Lh[0];
    r0[0] = r1[0]-bond_length * u[0];   r0[1] = r1[1]-bond_length * u[1];
    

    r2[0] = r1[0] + bond_length * u[0];   r2[1] = r1[1] + bond_length * u[1];

    thrust::host_vector<thrust::complex<double>> htot;
    htot.resize(mybox->M);

    std::complex<double> I(0.0, 1.0);

    double dr[2], mdr2, r[2];
    for ( int i=0 ; i<mybox->M ; i++ ) {
        // mybox->get_r(i, r);

        // mdr2 = mybox->pbc_dr2(dr, r, r0);
        // htot[i] = exp(-mdr2/2.0/smearLength/smearLength);
        // std::cout << mdr2 << " " ;
        // mdr2 = mybox->pbc_dr2(dr, r, r1);
        // std::cout << mdr2 << " " ;

        // htot[i] += exp(-mdr2/2.0/smearLength/smearLength);

        // mdr2 = mybox->pbc_dr2(dr, r, r2);
        // std::cout << mdr2 << std::endl;;
        // htot[i] += exp(-mdr2/2.0/smearLength/smearLength);


        double kv[3], k2;
        k2 = mybox->get_kD(i, kv);
        double kdotu = kv[0] * u[0] + kv[1] * u[1];

        thrust::complex<double> arg1 = exp(-I * (kv[0]*r1[0] + kv[1]*r1[1]));
        thrust::complex<double> arg0 = exp( I * kdotu * bond_length);
        thrust::complex<double> arg2 = exp(-I * kdotu * bond_length);

        htot[i] = smearFunc[i] * arg1 * ( 1.0 + arg0 + arg2 );
    }
    // mybox->writeTComplexGridData("htot.dat", htot);

    d_smearFunc = htot;
    
    mybox->cufftWrapperDouble( d_smearFunc, d_smearFunc, -1);
    
    smearFunc = d_smearFunc;

    mybox->writeTComplexGridData("lc-smear.dat", smearFunc);

    die("done");
    // end of debug
    ////////////////////////////



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

void LCMolec::computeLinearTerms() {
}
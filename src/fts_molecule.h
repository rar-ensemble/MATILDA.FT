// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////////////////
// fts_molecule.h               Rob Riggleman   //
//                              9 Aug 2022      //
// Parent class for molecules. Will contain     //
// minimum required variables to define a       //
// molecule.                                    //
//////////////////////////////////////////////////



#ifndef _FTS_MOLEC
#define _FTS_MOLEC

#include "include_libs.h"

class FTS_Box;

class FTS_Molec {
    protected:
        std::string input_command;
    public:
        std::string molec_type;     // what kind of molecule (linear, star, bottle brush)

        // density, d_density is the total density field associated with this type
        // If shape functions are used, they should be the total density, not the
        // center density.
        thrust::host_vector<thrust::complex<double>> density;
        thrust::device_vector<thrust::complex<double>> d_density;
        
        // Molecular partition function
        thrust::complex<double> Q;

        double phi;         // Vol fraction of this molecule
        double nmolecs;     // number of molecules of this molecule

        // cDensity is the particle center density
        thrust::device_vector<thrust::complex<double>> d_cDensity;
        
        // Pointer to the box containing this molecule
        FTS_Box* mybox;

        virtual void computeLinearTerms() = 0;
        
        FTS_Molec();
        FTS_Molec(std::istringstream &iss, FTS_Box*);
        virtual ~FTS_Molec();
        std::string printCommand() {return input_command;}

        virtual void calcDensity()=0;
};

#endif
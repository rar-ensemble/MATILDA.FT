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
#include "random.h"

class FTS_Box;

class FTS_Molec {
    protected:
        std::string input_command;
    public:
        std::string molec_type;     // what kind of molecule (linear, star, bottle brush)
        
        // Smearing parameters
        int doSmear;                // flag for whether smearing is implemented
        std::string smearStyle;     // name of smearing function
        double smearLength;         // length scale for smear function
        double smearWidth;          // interfacial width (if needed)

        // smearing functions
        thrust::host_vector<thrust::complex<double>> smearFunc;     // smearing function
        thrust::device_vector<thrust::complex<double>> d_smearFunc; // smearing function, k-space
        

        // density, d_density is the total density field associated with this type
        // If shape functions are used, they should be the total density, not the
        // center density.
        thrust::host_vector<thrust::complex<double>> density;
        thrust::device_vector<thrust::complex<double>> d_density;
        
        // cDensity is the particle center density
        thrust::device_vector<thrust::complex<double>> d_cDensity;

        // Molecular partition function
        thrust::complex<double> Q;

        double phi;         // Vol fraction of this molecule
        double nmolecs;     // number of molecules of this molecule


        
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
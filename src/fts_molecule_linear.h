// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).



#ifndef _FTS_MOLEC_LINEAR
#define _FTS_MOLEC_LINEAR
#include "fts_molecule.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>

class FTS_Box;

class LinearMolec : public FTS_Molec {
    protected:

    public:
        int numBlocks;
        bool isSymmetric = true;

        thrust::device_vector<int> d_N;                     // [numBlocks] length of each block
        thrust::host_vector<int> N;                         // [numBlocks] length of each block
        int Ntot;                                           // total chain length, sum of N
        
        thrust::host_vector<int> intSpecies;                 // [numBlocks] type for each block   
        thrust::device_vector<int> d_intSpecies;               // [numBlocks] type for each block   
        thrust::host_vector<std::string> blockSpecies;      // [numBlocks] type for each block as a string
        

        thrust::device_vector<thrust::complex<double>> d_q;    // [Ntot*M] Chain propagator
        thrust::device_vector<thrust::complex<double>> d_qdag; // [Ntot*M] inverse propagator

        // [M] Fourier transform of bond transition prob
        thrust::device_vector<thrust::complex<double>> d_bond_fft;    


        ~LinearMolec();
        LinearMolec(std::istringstream& iss, FTS_Box*);

        void calcPropagators();
        void calcDensity() override;
        void computeLinearTerms() override;
};

#endif
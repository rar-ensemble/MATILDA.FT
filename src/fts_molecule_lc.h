// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).



#ifndef _FTS_MOLEC_LC
#define _FTS_MOLEC_LC
#include "fts_molecule.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>

class FTS_Box;

class LCMolec : public FTS_Molec {
    protected:

    public:
        int Ntot;                                           // total mesogen length
        
        thrust::host_vector<int>     intSpecies;            // [Ntot] type for each site   
        thrust::device_vector<int> d_intSpecies;            // [Ntot] type for each site

        // [Dim*(Dim+1)/2*M] orientational order field
        thrust::host_vector<thrust::complex<double>> Stilde;
        thrust::device_vector<thrust::complex<double>> d_Stilde;

        // [Nu_tot*M] orientational order field
        thrust::host_vector<thrust::complex<double>> rhou;
        thrust::device_vector<thrust::complex<double>> d_rhou;

        ~LCMolec();
        LCMolec(std::istringstream&, FTS_Box*);

        void calcDensity() override;
        void recomputeNmolecs(void) override;

};
#endif
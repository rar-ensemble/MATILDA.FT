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
        double bond_length;
        
        std::string lc_style;
        std::vector<std::string>    text_species;           // [Ntot] text labels per site
        thrust::host_vector<int>     int_species;           // [Ntot] type for each site   
        thrust::device_vector<int> d_int_species;           // [Ntot] type for each site

        // Potential field for Ntot sites
        thrust::device_vector<thrust::complex<double>> W;

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
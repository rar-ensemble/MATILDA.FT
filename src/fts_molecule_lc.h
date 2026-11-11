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
        int smear_written = 0;
        
        std::string lc_style;
        std::vector<std::string>    text_species;           // [Ntot] text labels per site
        thrust::host_vector<int>     int_species;           // [Ntot] type for each site
        thrust::host_vector<int>    tensor_flag;            // [Ntot] M(r) for this particle?
        thrust::device_vector<int> d_int_species;           // [Ntot] type for each site

        // [M] Potential field for sites
        thrust::device_vector<thrust::complex<double>> d_W;
        thrust::device_vector<thrust::complex<double>> d_expW;
        thrust::device_vector<thrust::complex<double>> d_E;
        
        // [(Dim*(Dim+1)/2-1)*M] M(r) tensor field
        thrust::device_vector<thrust::complex<double>> d_Mr;

        // [(Dim*(Dim+1)/2-1)*M] orientational order field
        thrust::host_vector<thrust::complex<double>> Stilde;       
        thrust::device_vector<thrust::complex<double>> d_Stilde;
        
        // [(Dim*(Dim+1)/2-1)*M] orientational order field
        thrust::host_vector<thrust::complex<double>> cStilde;
        thrust::device_vector<thrust::complex<double>> d_cStilde;

        // [Nu_tot*M] orientational order field
        thrust::host_vector<thrust::complex<double>> rhou;
        thrust::device_vector<thrust::complex<double>> d_rhou;
        thrust::device_vector<thrust::complex<double>> d_Nru;
        thrust::device_vector<thrust::complex<double>> d_Nr;
        thrust::device_vector<thrust::complex<double>> d_expMddS;

        // Storage for the vectors that translate the Boltzmann weights
        thrust::host_vector<thrust::complex<double>> psi_x, psi_y, psi_z;
        thrust::device_vector<thrust::complex<double>> d_psi_x, d_psi_y, d_psi_z;
        

        ~LCMolec();
        LCMolec(std::istringstream&, FTS_Box*);

        void calcDensity() override;
        void recomputeNmolecs(void) override;
        void computeLinearTerms(void) override;
        void finish_initialization(void) override;
};
#endif
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

        cufftHandle fftBatchPlan;
        
        std::string lc_style;
        std::vector<std::string>    text_species;           // [Ntot] text labels per site
        thrust::host_vector<int>     int_species;           // [Ntot] type for each site
        thrust::host_vector<int>    tensor_flag;            // [Ntot] M(r) for this particle?
        thrust::device_vector<int> d_int_species;           // [Ntot] type for each site

        // The sites may be of different species. Fields and operators are stored
        // once per unique species on the molecule, indexed by its "slot".
        int nUnique;                                        // unique species on the mesogen
        std::vector<int> unique_species;                    // [nUnique] species index of each slot
        std::vector<int> site_slot;                         // [Ntot] slot of each site
        std::vector<int> tens_slot;                         // [nUnique] index into d_expMddS, -1 if no tensor sites
        std::vector<int> needs_Ek;                          // [nUnique] 1 if d_Ek is used for this slot
        int nTensSlots;                                     // slots with tensor-carrying sites
        thrust::device_vector<int> d_site_mask;             // [nUnique*Ntot] 1 if site belongs to slot

        // [M] Potential field scratch
        thrust::device_vector<thrust::complex<double>> d_W;

        // [nUnique*M] exp(-w) for each unique species
        thrust::device_vector<thrust::complex<double>> d_E;
        
        // [nTensSlots*(Dim*(Dim+1)/2-1)*M] smeared M(r) tensor field for each species with tensor sites
        thrust::device_vector<thrust::complex<double>> d_Mr;

        // [nUnique*(Dim*(Dim+1)/2-1)*M] smeared orientational order field
        thrust::host_vector<thrust::complex<double>> Stilde;       
        thrust::device_vector<thrust::complex<double>> d_Stilde;
        
        // [nUnique*(Dim*(Dim+1)/2-1)*M] center orientational order field
        thrust::host_vector<thrust::complex<double>> cStilde;
        thrust::device_vector<thrust::complex<double>> d_cStilde;

        // Functions of (r,u) are generated one orientation at a time and 
        // integrated on the fly, so nothing of size [Nu_tot*M] is stored.

        // [M] N(r,u) and R_j(r,u) at the current orientation, 
        // and [M] N(r) = \int du N(r,u)
        thrust::device_vector<thrust::complex<double>> d_Nu;
        thrust::device_vector<thrust::complex<double>> d_Ru;
        thrust::device_vector<thrust::complex<double>> d_Nr;

        // [nTensSlots*M] exp(-M(r):S(u)) at the current orientation for
        // each species with tensor sites
        thrust::device_vector<thrust::complex<double>> d_expMddS;

        // [M] scratch storage
        thrust::device_vector<thrust::complex<double>> d_tmp;

        // [nUnique*M] k-space Boltzmann weights for sites with no orientation
        // dependence, which can be transformed once per density calculation
        thrust::device_vector<thrust::complex<double>> d_Ek;

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
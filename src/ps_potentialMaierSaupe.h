// Copyright (c) 2025 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"

#ifndef _NBMaier
#define _NBMaier


class NBMaier : public PS_Potential {
    protected:
        int *MS_pair, *d_MS_pair;       // [ns] List of partner particles used to define orientation u
        float *ms_u, *d_ms_u;           // [Dim*ns] orientation vectors for all particles
        float *ms_S, *d_ms_S;           // [Dim*Dim*ns] S tensor for all particles
        float *S_field, *d_S_field;     // [Dim*Dim*M] S tensor field
        float *d_tmp_tensor;            // [Dim*Dim*M] Storage for tensor field manipulations
        float *h_Dim_Dim_tensor, *d_Dim_Dim_tensor;   // [Dim*Dim] Storage for order parameter calculation
        
        
        std::string filename;
        static int num;
        int nms;                        // Number of Maier-Saupe sites 
        float CalculateOrderParameter();
        float CalculateMaxEigenValue();
    public:
        NBMaier();      // Default constructor
        NBMaier(std::istringstream&, PS_Box*);  // Actual used constructor
        ~NBMaier();     // Default destructor

        void initializePotential(void) override;
        void CalcForces(void) override;
        float CalcEnergy(void) override;
        
        float Ao;       // prefactor for Maier-Saupe potential
        float sig2;     // Range of the Gaussian in the potential
};

#endif
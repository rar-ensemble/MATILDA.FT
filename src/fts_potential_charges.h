// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////
// Rob Riggleman         08/06/2026 //
// fts_potential_edwards.h          //
// Instance of fts_potential for    //
// the charges potential            //
// (Model E in ETIP)                //
//////////////////////////////////////

#ifndef _FTS_POTEN_Q
#define _FTS_POTEN_Q

__global__ void d_makeChargeForce(cuDoubleComplex*, const cuDoubleComplex*, 
    const cuDoubleComplex*, const double, const double, const int);

#include "fts_potential.h"

class FTS_Box;

class PotentialCharge : public FTS_Potential {
    protected:

    public:
        PotentialCharge(std::istringstream& iss, FTS_Box*);
        ~PotentialCharge();
        void updateFields() override;
        std::complex<double> calcHamiltonian() override;
        void writeFields(int) override;
        void initLinearCoeffs() override;
        void storePredictorData() override;
        void correctFields() override;
        
        // This field should contain the *smeared* density fields
        thrust::device_vector<thrust::complex<double>> d_rho_q;

        // Vector to store the force term
        thrust::device_vector<thrust::complex<double>> d_dHdw;

        // Noise for w field
        thrust::device_vector<thrust::complex<double>> d_wNoise;

        
        // Variables used in predictor-corrector methods
        thrust::device_vector<thrust::complex<double>> d_dHdwplo;
        thrust::device_vector<thrust::complex<double>> d_wplo;

        
        double delt;    // Size of time step
        double E;       // Dimensionless Bjerrum length
};



#endif
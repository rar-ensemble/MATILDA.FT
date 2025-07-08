// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////
// Rob Riggleman         07/08/2025 //
// fts_potential_edwards.h          //
// Instance of fts_potential for    //
// the Edwards potential            //
// (Model A in ETIP)                //
//////////////////////////////////////

#ifndef _FTS_POTEN_EDW
#define _FTS_POTEN_EDW

__global__ void d_makeEdwardsForce(
    cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*,
    const double, const double, const int);

#include "fts_potential.h"

class FTS_Box;

class PotentialEdwards : public FTS_Potential {
    protected:

    public:
        PotentialEdwards(std::istringstream& iss, FTS_Box*);
        ~PotentialEdwards();
        void updateFields() override;
        std::complex<double> calcHamiltonian() override;
        void writeFields(int) override;
        void initLinearCoeffs() override;
        void storePredictorData() override;
        void correctFields() override;
        
        // This field should contain the *smeared* density fields
        thrust::device_vector<thrust::complex<double>> d_rho_total;

        // Vector to store the force term
        thrust::device_vector<thrust::complex<double>> d_dHdw;

        
        // Variables used in predictor-corrector methods
        thrust::device_vector<thrust::complex<double>> d_dHdwplo;
        thrust::device_vector<thrust::complex<double>> d_wplo;

        
        double delt;                                            // Size of time step
        double B;                                          // Strength of the potential
};



#endif
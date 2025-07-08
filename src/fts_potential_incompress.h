// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////
// Rob Riggleman        12/24/2022  //
// fts_potential_helfand.h          //
// Instance of fts_potential for    //
// the Helfand weakly compressible  //
// potential (Model D in ETIP)      //
//////////////////////////////////////

#ifndef _FTS_POTEN_INC
#define _FTS_POTEN_INC

__global__ void d_makeIncompressForce(
    cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*,
    const double, const double, const int);

#include "fts_potential.h"

class FTS_Box;

class PotentialIncompress : public FTS_Potential {
    protected:

    public:
        PotentialIncompress(std::istringstream& iss, FTS_Box*);
        ~PotentialIncompress();
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
};



#endif
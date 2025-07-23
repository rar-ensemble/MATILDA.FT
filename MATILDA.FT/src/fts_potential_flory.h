// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


// Generate a subclass of FTS_Potential that implements the FLory potential.

#ifndef _FTS_POTEN_FLORY
#define _FTS_POTEN_FLORY


#include "fts_potential.h"

__global__ void d_makeFloryForce(cuDoubleComplex* , cuDoubleComplex* , const cuDoubleComplex* ,
    const cuDoubleComplex* ,const cuDoubleComplex* , const cuDoubleComplex* , const double,
    const double, const double, const int);


class FTS_Box;

class PotentialFlory : public FTS_Potential {

    private:

    public: 
        PotentialFlory();
        PotentialFlory(std::istringstream &iss, FTS_Box*);
        ~PotentialFlory();
        
        void updateFields() override;
        void correctFields() override;
        std::complex<double> calcHamiltonian() override;
        void writeFields(int) override;
        void storePredictorData() override;
        void initLinearCoeffs() override;

        thrust::device_vector<thrust::complex<double>> d_rhoI;
        thrust::device_vector<thrust::complex<double>> d_rhoJ;
        thrust::device_vector<thrust::complex<double>> d_dHdwpl;
        thrust::device_vector<thrust::complex<double>> d_dHdwmi;
        
        // Forces used in predictor-corrector methods
        thrust::device_vector<thrust::complex<double>> d_dHdwplo;
        thrust::device_vector<thrust::complex<double>> d_dHdwmio;
        
        // storage for ``old'' field configurations in P-C methods
        thrust::device_vector<thrust::complex<double>> d_wplo;
        thrust::device_vector<thrust::complex<double>> d_wmio;


        double deltPlus, deltMinus;     // Size of time step on w+, w- fields
        double chiN;                    // Strength of the potential
        std::string typeI, typeJ;       // Types involved in the potential
        int intTypeI, intTypeJ;         // species integer for types I, J

};

#endif
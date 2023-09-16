// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////////////
// fts_potential.h          Rob Riggleman   //
//                          16 Aug 2022     //
// Parent class for FTS potentials. Initial //
// plan is to include Helfand, Flory, and   //
// Edwards potentials. Charges should be    //
// added shortly thereafter.                //
//////////////////////////////////////////////

#ifndef _FTS_POTENTIAL
#define _FTS_POTENTIAL

#include "include_libs.h"
#include "FTS_Box.h"
__global__ void d_fts_updateEM(cuDoubleComplex*, const cuDoubleComplex*, const double, const int);
__global__ void d_fts_update1S(cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, const double, const int);

class FTS_Box;

class FTS_Potential {
    protected:
        std::string input_command;
        FTS_Box* mybox;
        std::string potentialStyle;

    public:

        thrust::complex<double> Hterm;      // Contribution of the potential to the effective Hamiltonian
        std::vector<std::string> actsOn;

        FTS_Potential(std::istringstream&, FTS_Box *);
        virtual ~FTS_Potential();
        bool zeroMean = false;

        std::string updateScheme;
        virtual void updateFields() = 0;       
        virtual std::complex<double> calcHamiltonian() = 0;
        virtual void initLinearCoeffs() = 0;
        std::string printCommand();
        std::string printStyle();
        virtual void writeFields(int) = 0;
        thrust::host_vector<thrust::complex<double>> wpl;       // w_plus for host
        thrust::device_vector<thrust::complex<double>> d_wpl;   // w_plus for device
        thrust::device_vector<thrust::complex<double>> d_Akpl;  // Linear coefficient for 1S, ETD updates

        thrust::host_vector<thrust::complex<double>> wmi;       // w_minus for host
        thrust::device_vector<thrust::complex<double>> d_wmi;   // w_minus for device
        thrust::device_vector<thrust::complex<double>> d_Akmi;  // Linear coefficient for 1S, ETD updates        
};

#endif
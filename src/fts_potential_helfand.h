// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////
// Rob Riggleman        12/24/2022  //
// fts_potential_helfand.h          //
// Instance of fts_potential for    //
// the Helfand weakly compressible  //
// potential (Model D in ETIP)      //
//////////////////////////////////////

#ifndef _FTS_POTEN_HELF
#define _FTS_POTEN_HELF

__global__ void d_makeHelfandForce(
    cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*,
    const double, const double, const double, const int);

#include "fts_potential.h"

class FTS_Box;

class PotentialHelfand : public FTS_Potential {
    protected:

    public:
        PotentialHelfand(std::istringstream& iss, FTS_Box*);
        ~PotentialHelfand();
        void updateFields() override;
        std::complex<double> calcHamiltonian() override;
        void writeFields(int) override;
        void initLinearCoeffs() override;

        double delt;                                            // Size of time step
        double kappaN;                                          // Strength of the potential
};



#endif
// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////////////////////////////////
// fts_potential_maiersaupe.h                                   //
// Field-theoretic Maier-Saupe potential. Owns the symmetric,   //
// traceless tensor field M(r) conjugate to the orientation     //
// tensor density S(r) of one species.                          //
//                                                              //
//   H_MS = C/(2 muN) \int dr M(r):M(r)                         //
//                                                              //
// with M/Nr entering the molecular partition function as       //
// exp( -(M/Nr):(uu - I/d) ). The force on each independent     //
// component is taken with respect to the full tensor,          //
//                                                              //
//   dH/dM_ab = C M_ab / muN + S_ab / Nr,                       //
//                                                              //
// so the saddle point M = -(muN / (C Nr)) S is purely real and //
// the field is relaxed like an exchange (w-) field.            //
//////////////////////////////////////////////////////////////////

#ifndef _FTS_POTEN_MAIERSAUPE
#define _FTS_POTEN_MAIERSAUPE

#include "fts_potential.h"

__global__ void d_makeMaierSaupeForce(cuDoubleComplex*, const cuDoubleComplex*,
    const cuDoubleComplex*, const double, const double, const double, const int);

class FTS_Box;

class PotentialMaierSaupe : public FTS_Potential {

    public: 
        PotentialMaierSaupe(std::istringstream &iss, FTS_Box*);
        ~PotentialMaierSaupe();
        
        void updateFields() override;
        void correctFields() override;
        std::complex<double> calcHamiltonian() override;
        void writeFields(int) override;
        void storePredictorData() override;
        void initLinearCoeffs() override;

        int Ntens;              // Independent tensor components, dim*(dim+1)/2 - 1
        int intSpecies;         // Index of the species this potential acts on
        double muN;             // Strength of the potential
        double delt;            // Size of the time step

        // [Ntens*M] independent components of M(r); in 2D these are xx, xy
        thrust::device_vector<thrust::complex<double>> d_M;

        // [Ntens*M] force, and stored data for predictor-corrector updates
        thrust::device_vector<thrust::complex<double>> d_dHdM;
        thrust::device_vector<thrust::complex<double>> d_dHdMo;
        thrust::device_vector<thrust::complex<double>> d_Mo;

        // [Ntens*M] noise for complex Langevin
        thrust::device_vector<thrust::complex<double>> d_noise;

    private:
        void makeForce();
};

#endif

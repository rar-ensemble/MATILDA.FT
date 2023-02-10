// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _FTS_BOX
#define _FTS_BOX

#include "Box.h"
#include <cufft.h>
#include <cufftXt.h>

class FTS_Species;
class FTS_Molec;
class FTS_Potential;

class FTS_Box : public Box {
    protected:
        std::string ftsStyle;              // "scft" or "cl", maybe also "hpf" in future?
    public:
        ~FTS_Box();
        FTS_Box(std::istringstream&);

        double rho0;        // System density
        double Nr;          // Reference chain length used to non-dimensionalize fields
        double C;           // System concentration
        int chemFieldFreq;  // Frequency to write potential fields
        std::complex<double> Heff;  // Effective Hamiltonian
        
        
        std::vector<FTS_Species> Species;       // Contains the density of each species
        std::vector<FTS_Molec*> Molecs;         // Calculates properties of each species
        std::vector<FTS_Potential*> Potentials; // Stores and updates potentials

        std::ofstream OTP;
        std::string returnFTSstyle();
        void readInput(std::ifstream&);
        void doTimeStep(int);
        void initializeSim() override;
        void writeSpeciesDensityFields();
        void writeData(int) override;
        void computeHamiltonian();
        void writeFields() override;

        std::complex<double> integComplexD(std::complex<double>*);
        thrust::complex<double> integTComplexD(thrust::host_vector<thrust::complex<double>>);
        void writeComplexGridData(std::string, std::vector<std::complex<double>>);
        void writeTComplexGridData(std::string, thrust::host_vector<thrust::complex<double>>);
        void computeHomopolyDebye(thrust::host_vector<thrust::complex<double>>& , const double);
        void computeIntRABlockDebye(thrust::host_vector<thrust::complex<double>>&, const double, const double);
        void computeIntERBlockDebye(thrust::host_vector<thrust::complex<double>>&, const double, 
            const double, const double, const double);
};

#endif // FTS_BOX

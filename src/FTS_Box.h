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
    public:
        ~FTS_Box();
        FTS_Box(std::istringstream&);

        double rho0;        // System density
        double Nr;          // Reference chain length used to non-dimensionalize fields
        double C;           // System concentration
        int chemFieldFreq;  // Frequency to write potential fields
        std::complex<double> Heff;  // Effective Hamiltonian
        bool PCflag;         // Flag for using predictor-corrector methods
        std::string ftsStyle;              // "scft" or "cl"

        int clSampleFreq;   // Steps between CL density-average samples (cl only)
        int nCLSamples;     // Running count of accumulated CL density samples
        int cl_wait;        // Number of steps to wait before computing averages


        // Data for tolerance checks
        double tolerance, Hold;   // Convergence tolerance, old real part of Heff for SCFT simulation
        double error;
        std::string tolMetric;    // Metric to use for convergence
        std::vector<std::complex<double>> tolField, oldValues;
        int tolIndex;       // index for field/potential/phi styles
        void initTolerances(void);
        double convergenceCheck(void);


        cuDoubleComplex *d_cpxGabe, *d_cpxAtmn; // [M] temp storage for device complex arrays
        
        std::vector<FTS_Species> Species;       // Contains the density of each species
        std::vector<FTS_Molec*> Molecs;         // Calculates properties of each species
        std::vector<FTS_Potential*> Potentials; // Stores and updates potentials
        long int speciesTimer, moleculeTimer;  // stores time spent in various algorithms
        long int fieldUpdateTimer;              

        std::ofstream OTP;
        std::string returnFTSstyle();
        void readInput(std::ifstream&);
        void doTimeStep(int);
        void finishInitialization();
        void writeSpeciesDensityFields();
        void writeData(int) override;
        void computeHamiltonian();
        void writeFields() override;
        void writeTime() override;
        void initSmearGaussian(thrust::host_vector<thrust::complex<double>>&, const double, const double);
        int converged(int) override;
        void modifyBox(std::istringstream&) override;

        void NVT(int) override;
        
        void findSpinodal(std::istringstream&) override;
        void storePotentials(std::complex<double>*, std::complex<double>*, double*, double*, const int);
        void restorePotentials(std::complex<double>*, std::complex<double>*, double*, double*, const int);
        void shuffleStorage(std::complex<double>*, std::complex<double>*, double*, double*);

        std::complex<double> integComplexD(std::complex<double>*);
        thrust::complex<double> integTComplexD(thrust::host_vector<thrust::complex<double>>);
        void writeComplexGridData(std::string, std::vector<std::complex<double>>);
        void writeTComplexGridData(std::string, thrust::host_vector<thrust::complex<double>>);
        void computeHomopolyDebye(thrust::host_vector<thrust::complex<double>>& , const double);
        void computeIntRABlockDebye(thrust::host_vector<thrust::complex<double>>&, const double, const double);
        void computeIntERBlockDebye(thrust::host_vector<thrust::complex<double>>&, const double, 
            const double, const double, const double);

        
        // Variables and functions for orientation/unit sphere 
        // In 2D the orientations are Nu_tot equally spaced angles on the unit
        // circle; in 3D they are the points of a Lebedev rule with Nu_tot points.
        // Orientation integrals are \int du f(u) = \sum_m uwts[m] f(u_m), 
        // with \sum_m uwts[m] = Omega (2 pi in 2D, 4 pi in 3D).
        int Ntens;          // dim*(dim+1)/2 - 1, number of ind tensor comps
        int Nu[2], Nu_tot;  // Number of unit sphere grid points
        double Omega;       // Total solid angle
        void initOrientations(void);
        void integrate_u_fixed_r(cuDoubleComplex*, const cuDoubleComplex*);
        
        // uvecs     [Dim*Nu_tot]   unit vectors
        // uwts      [Nu_tot]       quadrature weights
        // Stens     [Ntens*Nu_tot] independent components of S(u) = uu - I/d
        //                          2D: xx, xy.  3D: xx, yy, xy, xz, yz
        // Scontract [Ntens*Nu_tot] coefficients c such that for a symmetric traceless T
        //                          stored by its independent components, 
        //                          T:S(u) = \sum_nt c[nt] * T[nt]
        thrust::host_vector<double>     uvecs,   uwts,   Stens,   Scontract;
        thrust::device_vector<double> d_uvecs, d_uwts, d_Stens, d_Scontract;
};

#endif // FTS_BOX

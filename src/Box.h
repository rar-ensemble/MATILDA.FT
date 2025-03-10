// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _BOX
#define _BOX


#include "include_libs.h"
#include "constants.h"

class Box {
    protected:
        int Dim;                            // System dimensionality
        std::string input_command;          // Command to create this box
        int nTotalBoxes;                    // Number of simulation boxes

    public:
        thrust::host_vector<int> Nx;        // Grid dimensions
        thrust::device_vector<int> d_Nx;    
        int* _d_Nx;

        thrust::host_vector<double> dx;     // grid spacing in each direction
        thrust::device_vector<double> d_dx;
        double* _d_dx;
        
        float *L, *d_L, *Lh, *d_Lh;         // [Dim] Box dimensions, half-box dimensions

        double V;                           // Box volume
        int M;                              // Total number of grid points
        double gvol;                        // Grid volume
        int M_Grid, M_Block;                // GPU Configuration parameters
        int logFreq;                        // Frequency to print energies
        int densityFieldFreq;               // Frequency to write configs
        int maxSteps;                       // Max number of steps to run
        int totSteps;                       // Total elapsed iterations
        long int simTime;                   // Total simulation time
        long int ftTimer;                   // Time spent in FFT routine
        long int ioTimer;                   // Time in I/O routines
        int blockSize;                      // GPU block size

        cufftHandle fftplan, fftplanSingle; // FFT Plans
        void cufftWrapperDouble(thrust::device_vector<thrust::complex<double>>,
            thrust::device_vector<thrust::complex<double>>&, const int);
        void convolveTComplexDouble(thrust::device_vector<thrust::complex<double>>,
            thrust::device_vector<thrust::complex<double>>&, thrust::device_vector<thrust::complex<double>>);

        void cufftWrapperSingle(cuComplex*, cuComplex*, const int);

        Box();
        Box(std::istringstream&);
        virtual ~Box();
        void setDimension(int);
        
        virtual void initializeSim() = 0;               // Subroutine to initial densities/fields prior to a sim
        double get_kD(int, double*);        // Subroutine to compute wavevector corresponding to grid index
        float get_kD(int, float*);        // Subroutine to compute wavevector corresponding to grid index
        void get_r(int, double*);           // Subroutine to compute position corresponding to grid index
        void get_rf(int, float*);           // Subroutine to compute position corresponding to grid index
        // template<typename T> T pbc_dr2(T*, const T*, const T*);
        float pbc_dr2(float*, const float*, const float*);
        double pbc_dr2(double*, const double*, const double*);
        
        void unstack2(int, int*);
        int returnDimension(void);
        std::string printCommand();
        virtual void readInput(std::ifstream&) = 0;
        virtual void writeFields() = 0;
        virtual void writeTime() = 0;
        virtual int converged(int) = 0;

        virtual void NVT(int) = 0;

        virtual void writeData(int) = 0;

        virtual void doTimeStep(int) = 0;

};

#endif // BOX
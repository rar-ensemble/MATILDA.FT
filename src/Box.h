// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _BOX
#define _BOX


#include "include_libs.h"
#include "constants.h"
#include "fft_wrapper.h"

class Box {
    protected:
        int Dim;                            // System dimensionality
        std::string input_command;          // Command to create this box

        void init_fft_plan(cufftType);
        void execute_fft(FFT_COMPLEX * , FFT_COMPLEX *, int );
    public:
        thrust::host_vector<int> Nx;        // Grid dimensions
        thrust::device_vector<int> d_Nx;    

        thrust::host_vector<float> L;       // Box dimensions
        thrust::device_vector<float> d_L;

        thrust::host_vector<double> dx;     // grid spacing in each direction

        thrust::host_vector<float> Lh;         // Half box dimensions
        thrust::device_vector<float> d_Lh;

        double V;                           // Box volume
        int M;                              // Total number of grid points
        double gvol;                        // Grid volume
        int M_Grid, M_Block;                // GPU Configuration parameters
        int logFreq;                        // Frequency to print energies
        int densityFieldFreq;               // Frequency to write configs
        int maxSteps;                       // Max number of steps to run
        long int simTime;                   // Total simulation time
        long int ftTimer;                   // Time spent in FFT routine

        cufftHandle fftplan;                // FFT Plan
        void cufftWrapperDouble(thrust::device_vector<thrust::complex<double>>,
            thrust::device_vector<thrust::complex<double>>&, const int);

        Box();
        Box(std::istringstream&);
        virtual ~Box();
        void setDimension(int);
        
        virtual void initializeSim() = 0;               // Subroutine to initial densities/fields prior to a sim
        double get_kD(int, double*);        // Subroutine to compute wavevector corresponding to grid index
        void get_r(int, double*);           // Subroutine to compute position corresponding to grid index
        void unstack2(int, int*);
        int returnDimension(void);
        std::string printCommand();
        virtual void readInput(std::ifstream&) = 0;
        virtual void writeFields() = 0;
        virtual void writeTime() = 0;

        virtual void writeData(int) = 0;

        virtual void doTimeStep(int) = 0;
};

#endif // BOX
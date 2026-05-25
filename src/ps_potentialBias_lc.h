// Copyright (c) 2025 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"

#ifndef _NBBIASLC
#define _NBBIASLC

// #define EIGEN_NO_CUDA
// #include <Eigen/Dense>
// #include <Eigen/Eigenvalues>

class PS_Box;

class NBBiasLC : public PS_Potential {
    protected:
        int *MS_pair, *d_MS_pair;       // [nstot] partner particle index per site (-1 if none)
        float *ms_u, *d_ms_u;           // [Dim*nstot] unit orientation vectors
        float *ms_S, *d_ms_S;           // [Dim²*nstot] per-particle S tensors
        float *ener, *d_ener;           // [nstot] stores energy per particle

        std::string filename;           // Path to lc_file specifying MS pairs
        
        int nms;                        // Number of active MS pairs read from lc_file

        void  CalcSTensors();
        void  read_lc_file(std::string);

    public:
        NBBiasLC();
        NBBiasLC(std::istringstream&, PS_Box*);
        ~NBBiasLC();

        void initializePotential(void) override;
        void CalcForces(void) override;
        float CalcEnergy(void) override;

        float Ao;           // Gaussian potential prefactor
        float u0[3];        // Bias orientation
        float *S0, *d_S0; // [Dim^2] Bias tensor orientation
};

// Forward declarations for device kernels defined in this file
__global__ void d_calcParticleSTensors(float*, float*, const float*, const int*,
    const float*, const float*, const int, const int);

__global__ void d_mapFieldSTensors(float*, const int*, const float*, const float*,
    const int*, const int, const int, const int);

__global__ void d_extractTensorComponent(cuComplex*, const float*, const int,
    const int, const int, const int);

__global__ void d_storeTensorComponent(float*, const cuComplex*, const int,
    const int, const int, const int);


__global__ void d_accumulateMSBiasForce(float*, const float*, const int*, const float*,
    const float*, const float, const int, const float*, const float*, const int);

__global__ void d_computeLCBiasEnergyPerPartic(float* , const float*, const float*, 
    const int, const int);

__global__ void d_doubleDotTensorFields(float*, const float*, const float*, const int, const int);

__global__ void d_SumAndAverageSTensors(const float*, float*, const int*, const int, const int);



#endif

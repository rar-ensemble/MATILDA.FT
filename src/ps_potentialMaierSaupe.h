// Copyright (c) 2025 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"

#ifndef _NBMaier
#define _NBMaier

// #define EIGEN_NO_CUDA
// #include <Eigen/Dense>
// #include <Eigen/Eigenvalues>

class PS_Box;

class NBMaier : public PS_Potential {
    protected:
        int *MS_pair, *d_MS_pair;       // [nstot] partner particle index per site (-1 if none)
        float *ms_u, *d_ms_u;           // [Dim*nstot] unit orientation vectors
        float *ms_S, *d_ms_S;           // [Dim²*nstot] per-particle S tensors
        float *S_field, *d_S_field;     // [Dim²*M] S tensor field on grid
        float *d_tmp_tensor;            // [Dim²*M] scratch tensor field (device only)
        float *h_Dim_Dim_tensor;        // [Dim²] host copy for eigenvalue solve
        float *d_Dim_Dim_tensor;        // [Dim²] accumulated S tensor for order param

        std::string filename;           // Path to lc_file specifying MS pairs
        static int num;
        int nms;                        // Number of active MS pairs read from lc_file

        float CalculateOrderParameter();
        // float CalculateMaxEigenValue();
        void  CalcSTensors();
        void  read_lc_file(std::string);

    public:
        NBMaier();
        NBMaier(std::istringstream&, PS_Box*);
        ~NBMaier();

        void initializePotential(void) override;
        void CalcForces(void) override;
        float CalcEnergy(void) override;

        float Ao;           // Gaussian potential prefactor
        float sig2;         // Squared range of the Gaussian (stored as sigma^2)
        float orderParam;   // Most recent nematic order parameter (largest eigenvalue * 1.5)
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

__global__ void d_accumulateMSForce1(float*, const int*, const float*, const float*,
    const float*, const int*, const float, const int, const int, const int, const int);

__global__ void d_accumulateMSForce2(float*, const float*, const int*, const float*,
    const float*, const float*, const int*, const float, const int, const int,
    const float*, const float*, const int);

__global__ void d_doubleDotTensorFields(float*, const float*, const float*, const int, const int);

__global__ void d_SumAndAverageSTensors(const float*, float*, const int*, const int, const int);

#endif

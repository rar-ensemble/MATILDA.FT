// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "tensor_potential.h"

#ifndef _TPAIR_MAIERSAUPE
#define _TPAIR_MAIERSAUPE


class MaierSaupe : public TensorPotential, public Potential {
    private:
        int *MS_pair, *d_MS_pair;       // [ns] List of partner particles used to define orientation u
        float *ms_u, *d_ms_u;           // [Dim*ns] orientation vectors for all particles
        float *ms_S, *d_ms_S;           // [Dim*Dim*ns] S tensor for all particles
        float *S_field, *d_S_field;     // [Dim*Dim*M] S tensor field
        float *d_tmp_tensor;            // [Dim*Dim*M] Storage for tensor field manipulations
        float *h_Dim_Dim_tensor, *d_Dim_Dim_tensor;            // [Dim*Dim] Storage for order parameter calculation
        std::string filename;
        static int num;
        int nms;                        // Number of Maier-Saupe sites 
        float CalculateOrderParameter();
        float CalculateMaxEigenValue(float* );
        float MaierSaupe::CalculateOrderParameterGridPoints();
    public:
        MaierSaupe();
        MaierSaupe(std::istringstream& iss);
        ~MaierSaupe();

        void Initialize() override;
        void Allocate(void);
        void CalcVirial()  override { }
        void read_lc_file(std::string);
        void ramp_check_input(std::istringstream& iss);
        void CalcForces(void) override;
        float CalcEnergy(void) override;
        void CalcSTensors(void);
        void ReportEnergies(int&)  override;
};

#endif

__global__ void init_device_gaussian(cufftComplex*, cufftComplex*, 
    float, float, const float*,
    const int, const int*, const int);
__global__ void d_complex2real(cufftComplex*, float*, int);
__global__ void d_calcParticleSTensors(float*, float*, const float*, const int*, const float*,
    const float*, const int, const int);
__global__ void d_zero_float_vector(float*, int);
__global__ void d_mapFieldSTensors( float*, const int*, const float*, const float*, 
    const int*, const int, const int, const int);
__global__ void d_extractTensorComponent(cufftComplex*, const float*, const int,
    const int, const int, const int);
__global__ void d_storeTensorComponent(float*, const cufftComplex*, const int,
    const int, const int, const int);

__global__ void d_prepareForceKSpace(cufftComplex*, cufftComplex*, cufftComplex*,
    const int, const int, const int);

__global__ void d_accumulateMSForce1(float*, const int*, const float*, const float*, const float*, const int*,
    const float, const int, const int, const int, const int);

__global__ void d_accumulateMSForce2(float*, const float*, const int*, const float*, const float*, const float*, 
    const int*, const float, const int, const int, const float*, const float*, const int);

__global__ void d_multiplyComplex(cufftComplex*, cufftComplex*, cufftComplex*, const int);

__global__ void d_doubleDotTensorFields(float*, const float*, const float*, const int, const int);

__global__ void d_SumAndAverageSTensors(
    const float* ms_S,                // [Dim*Dim*ns] stores the particle S-tensor
    float* summed_S_Tensor,             // [Dim*Dim] 
    const int* upartner,        // [ns] partner particle for defining u-vector
    const int Dim,              // Dimensionality
    const int ns );            // number of sites
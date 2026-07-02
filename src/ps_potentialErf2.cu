// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potentialErf2.h"
#include "PS_Box.h"

// Forward declarations for kernels used from device_utils.cu
__global__ void d_floatToCpx(cuComplex*, const float*, const int);
__global__ void d_cpxToFloat(float*, const cuComplex*, const int);

NBErf2::NBErf2() {}
NBErf2::~NBErf2() {}

// Input format: pair_potential erf2 <grpI> <grpJ> <Ao> <Rp> <sigma>
NBErf2::NBErf2(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {

    iss >> grpI;
    iss >> grpJ;
    iss >> Ao;

    float sigma;
    iss >> Rp >> sigma;
    sig2 = sigma * sigma;

    int is_ramping = ramp_check_input(iss, Ao);
}

// Direct k-space initialization matching main branch potential_erf.cu.
// u(k) = Ao * exp(-k²σ²/2) * [4π(sin(Rp·k) - Rp·k·cos(Rp·k)) / (k²k)]²
// (convolution of two identical step functions with a Gaussian smearing)
void NBErf2::initializePotential() {
    std::cout << "Initializing Erf2 potential..." << std::endl;

    PS_Potential::initializePotential();

    std::complex<float> I(0.0f, 1.0f);
    float kv[3];
    int Dim = mybox->returnDimension();
    int M   = mybox->M;

    for (int i = 0; i < M; i++) {
        float k2 = mybox->get_kD(i, kv);
        float k  = sqrtf(k2);

        if (k2 == 0.0f) {
            uk[i] = Ao * (PI4 * Rp*Rp*Rp / 3.0f) * (PI4 * Rp*Rp*Rp / 3.0f);
        } else {
            float temp = PI4 * (sinf(Rp*k) - Rp*k*cosf(Rp*k)) / (k2 * k);
            uk[i] = Ao * expf(-k2 * sig2 * 0.5f) * temp * temp;
        }

        for (int j = 0; j < Dim; j++)
            fk[i*Dim + j] = -I * kv[j] * uk[i];
    }

    cudaMemcpy(d_uk, uk, M * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

    // Inverse FFT d_uk → real-space potential d_ur
    mybox->cufftWrapperSingle(d_uk, mybox->d_cpxAlex, -1);
    d_cpxToFloat<<<mybox->M_Grid, mybox->M_Block>>>(d_ur, mybox->d_cpxAlex, M);
    cudaMemcpy(ur, d_ur, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_fk, fk, M * Dim * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    check_cudaError("NBErf2: initialization");

    std::cout << "  Erf2 initialization completed" << std::endl;
}
